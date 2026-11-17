// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
)

const maxMessage = 16 << 20

const maxHeaderBytes = 8 << 10

const maxHeaderLines = 32

type message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   json.RawMessage `json:"error,omitempty"`
}

const codeParseError = -32700

type parseError struct{ err error }

func (e *parseError) Error() string { return e.err.Error() }

func (e *parseError) Unwrap() error { return e.err }

func (m *message) isResponse() bool { return m.Method == "" }

func (m *message) isRequest() bool { return m.Method != "" && len(m.ID) > 0 }

type reader struct{ r *bufio.Reader }

func newReader(r io.Reader) *reader { return &reader{bufio.NewReaderSize(r, 1<<16)} }

func (rd *reader) read() (*message, error) {
	n, err := rd.contentLength()
	if err != nil {
		return nil, err
	}
	body := make([]byte, 0, min(n, 32768))
	buf := make([]byte, 32768)
	for len(body) < n {
		m, err := io.ReadFull(rd.r, buf[:min(len(buf), n-len(body))])
		body = append(body, buf[:m]...)
		if err != nil {
			return nil, fmt.Errorf("reading message body: %w", midMessage(err))
		}
	}
	m := new(message)
	if err := json.Unmarshal(body, m); err != nil {
		return nil, &parseError{err: fmt.Errorf("decoding message: %w", err)}
	}
	return m, nil
}

func (rd *reader) contentLength() (int, error) {
	n, total := -1, 0
	for range maxHeaderLines {
		line, err := rd.headerLine(maxHeaderBytes - total)
		total += len(line)
		if err != nil {
			if errors.Is(err, io.EOF) && total == 0 {
				return 0, io.EOF
			}
			return 0, fmt.Errorf("reading message header: %w", midMessage(err))
		}
		trimmed := strings.TrimRight(line, "\r\n")
		if trimmed == "" {
			return checkedLength(n)
		}
		if v, ok := headerValue(trimmed, "Content-Length"); ok {
			if n, err = takeLength(n, v); err != nil {
				return 0, err
			}
		}
	}
	return 0, errors.New("message header too large")
}

// headerLine stops at the remaining header budget even without a newline.
// Reading an entire line before checking its length would leave allocation
// and reads unbounded for an incomplete frame.
func (rd *reader) headerLine(remaining int) (string, error) {
	var line strings.Builder
	for range remaining {
		c, err := rd.r.ReadByte()
		if err != nil {
			return line.String(), err
		}
		line.WriteByte(c)
		if c == '\n' {
			return line.String(), nil
		}
	}
	return line.String(), errors.New("message header too large")
}

func checkedLength(n int) (int, error) {
	if n < 0 {
		return 0, errors.New("message header has no Content-Length")
	}
	return n, nil
}

func takeLength(n int, v string) (int, error) {
	if n >= 0 {
		return 0, errors.New("message header has duplicate Content-Length")
	}
	return parseLength(v)
}

func midMessage(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}

func headerValue(line, name string) (string, bool) {
	key, value, ok := strings.Cut(line, ":")
	if !ok || !strings.EqualFold(strings.TrimSpace(key), name) {
		return "", false
	}
	return strings.TrimSpace(value), true
}

func parseLength(v string) (int, error) {
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 || n > maxMessage {
		return 0, fmt.Errorf("invalid Content-Length %q", v)
	}
	return n, nil
}

type writer struct {
	mu sync.Mutex
	w  io.Writer
}

func (w *writer) write(m *message) error {
	m.JSONRPC = "2.0"
	body, err := json.Marshal(m)
	if err != nil {
		return err
	}
	var frame bytes.Buffer
	fmt.Fprintf(&frame, "Content-Length: %d\r\n\r\n", len(body))
	frame.Write(body)
	w.mu.Lock()
	defer w.mu.Unlock()
	_, err = w.w.Write(frame.Bytes())
	return err
}

func (w *writer) notify(method string, params any) error {
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	return w.write(&message{Method: method, Params: raw})
}

func (w *writer) respond(id, result json.RawMessage) error {
	if len(result) == 0 {
		result = nullResult
	}
	return w.write(&message{ID: id, Result: result})
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

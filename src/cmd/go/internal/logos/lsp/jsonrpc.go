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

// maxMessage bounds the body of one message, so a corrupt length header
// cannot ask for all the memory there is.
const maxMessage = 1 << 30

// message is one JSON-RPC 2.0 message. A request has a method and an ID, a
// notification has a method and no ID, and a response has an ID and a
// result or an error.
type message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   json.RawMessage `json:"error,omitempty"`
}

// isResponse reports whether m answers a request.
func (m *message) isResponse() bool { return m.Method == "" }

// isRequest reports whether m is a request.
func (m *message) isRequest() bool { return m.Method != "" && len(m.ID) > 0 }

// reader reads framed messages: a Content-Length header block, a blank
// line, and the JSON body.
type reader struct{ r *bufio.Reader }

// newReader returns a reader of framed messages from r.
func newReader(r io.Reader) *reader { return &reader{bufio.NewReaderSize(r, 1<<16)} }

// read returns the next message. It returns io.EOF at a clean end of input.
func (rd *reader) read() (*message, error) {
	n, err := rd.contentLength()
	if err != nil {
		return nil, err
	}
	body := make([]byte, n)
	if _, err := io.ReadFull(rd.r, body); err != nil {
		return nil, fmt.Errorf("reading message body: %w", midMessage(err))
	}
	m := new(message)
	if err := json.Unmarshal(body, m); err != nil {
		return nil, fmt.Errorf("decoding message: %w", err)
	}
	return m, nil
}

// contentLength consumes one header block and returns the body length.
func (rd *reader) contentLength() (int, error) {
	line, err := rd.r.ReadString('\n')
	if errors.Is(err, io.EOF) && line == "" {
		return 0, io.EOF
	}
	n := -1
	for {
		if err != nil {
			return 0, fmt.Errorf("reading message header: %w", midMessage(err))
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if v, ok := headerValue(line, "Content-Length"); ok {
			if n, err = parseLength(v); err != nil {
				return 0, err
			}
		}
		line, err = rd.r.ReadString('\n')
	}
	if n < 0 {
		return 0, errors.New("message header has no Content-Length")
	}
	return n, nil
}

// midMessage turns the end of input inside a message into
// io.ErrUnexpectedEOF, so only an end between messages reads as a clean
// end of the stream.
func midMessage(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}

// headerValue returns the value of the header line if it is named name.
func headerValue(line, name string) (string, bool) {
	key, value, ok := strings.Cut(line, ":")
	if !ok || !strings.EqualFold(strings.TrimSpace(key), name) {
		return "", false
	}
	return strings.TrimSpace(value), true
}

// parseLength parses a Content-Length value.
func parseLength(v string) (int, error) {
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 || n > maxMessage {
		return 0, fmt.Errorf("invalid Content-Length %q", v)
	}
	return n, nil
}

// writer writes framed messages. Its methods are safe for concurrent use.
type writer struct {
	mu sync.Mutex
	w  io.Writer
}

// write frames and writes m, with the JSON-RPC version filled in.
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

// notify writes a notification.
func (w *writer) notify(method string, params any) error {
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	return w.write(&message{Method: method, Params: raw})
}

// respond writes the successful response of request id. A response always
// carries a result, so an absent one is written as null.
func (w *writer) respond(id, result json.RawMessage) error {
	if len(result) == 0 {
		result = nullResult
	}
	return w.write(&message{ID: id, Result: result})
}

// rpcError is the error object of a JSON-RPC response.
type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// fail writes the error response of request id.
func (w *writer) fail(id json.RawMessage, code int, text string) error {
	raw, err := json.Marshal(rpcError{Code: code, Message: text})
	if err != nil {
		return err
	}
	return w.write(&message{ID: id, Error: raw})
}

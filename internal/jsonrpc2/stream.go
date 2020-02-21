// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
)

// Stream abstracts the transport mechanics from the JSON RPC protocol.
// A Conn reads and writes messages using the stream it was provided on
// construction, and assumes that each call to Read or Write fully transfers
// a single message, or returns an error.
type Stream interface {
	// Read gets the next message from the stream.
	// It is never called concurrently.
	Read(context.Context) ([]byte, int64, error)
	// Write sends a message to the stream.
	// It must be safe for concurrent use.
	Write(context.Context, []byte) (int64, error)
}

// ErrDisconnected signals that the stream or connection exited normally.
var ErrDisconnected = errors.New("disconnected")

// NewStream returns a Stream built on top of an io.Reader and io.Writer
// The messages are sent with no wrapping, and rely on json decode consistency
// to determine message boundaries.
func NewStream(in io.Reader, out io.Writer) Stream {
	return &plainStream{
		in:  json.NewDecoder(in),
		out: out,
	}
}

type plainStream struct {
	in    *json.Decoder
	outMu sync.Mutex
	out   io.Writer
}

func (s *plainStream) Read(ctx context.Context) ([]byte, int64, error) {
	select {
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	default:
	}
	var raw json.RawMessage
	if err := s.in.Decode(&raw); err != nil {
		if err == io.EOF {
			return nil, 0, ErrDisconnected
		}
		return nil, 0, err
	}
	return raw, int64(len(raw)), nil
}

func (s *plainStream) Write(ctx context.Context, data []byte) (int64, error) {
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	default:
	}
	s.outMu.Lock()
	n, err := s.out.Write(data)
	s.outMu.Unlock()
	return int64(n), err
}

// NewHeaderStream returns a Stream built on top of an io.Reader and io.Writer
// The messages are sent with HTTP content length and MIME type headers.
// This is the format used by LSP and others.
func NewHeaderStream(in io.Reader, out io.Writer) Stream {
	return &headerStream{
		in:  bufio.NewReader(in),
		out: out,
	}
}

type headerStream struct {
	in    *bufio.Reader
	outMu sync.Mutex
	out   io.Writer
}

func (s *headerStream) Read(ctx context.Context) ([]byte, int64, error) {
	select {
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	default:
	}
	var total, length int64
	// read the header, stop on the first empty line
	for {
		line, err := s.in.ReadString('\n')
		total += int64(len(line))
		if err == io.EOF {
			// A normal disconnection will terminate with EOF before the next header.
			return nil, total, ErrDisconnected
		}
		if err != nil {
			return nil, total, fmt.Errorf("failed reading header line %q", err)
		}
		line = strings.TrimSpace(line)
		// check we have a header line
		if line == "" {
			break
		}
		colon := strings.IndexRune(line, ':')
		if colon < 0 {
			return nil, total, fmt.Errorf("invalid header line %q", line)
		}
		name, value := line[:colon], strings.TrimSpace(line[colon+1:])
		switch name {
		case "Content-Length":
			if length, err = strconv.ParseInt(value, 10, 32); err != nil {
				return nil, total, fmt.Errorf("failed parsing Content-Length: %v", value)
			}
			if length <= 0 {
				return nil, total, fmt.Errorf("invalid Content-Length: %v", length)
			}
		default:
			// ignoring unknown headers
		}
	}
	if length == 0 {
		return nil, total, fmt.Errorf("missing Content-Length header")
	}
	data := make([]byte, length)
	if _, err := io.ReadFull(s.in, data); err != nil {
		return nil, total, err
	}
	total += length
	return data, total, nil
}

func (s *headerStream) Write(ctx context.Context, data []byte) (int64, error) {
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	default:
	}
	s.outMu.Lock()
	defer s.outMu.Unlock()
	n, err := fmt.Fprintf(s.out, "Content-Length: %v\r\n\r\n", len(data))
	total := int64(n)
	if err == nil {
		n, err = s.out.Write(data)
		total += int64(n)
	}
	return total, err
}

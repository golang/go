// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"bufio"
	"context"
	"encoding/json"
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
	Read(context.Context) ([]byte, error)
	// Write sends a message to the stream.
	// It must be safe for concurrent use.
	Write(context.Context, []byte) error
}

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

func (s *plainStream) Read(ctx context.Context) ([]byte, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	var raw json.RawMessage
	if err := s.in.Decode(&raw); err != nil {
		return nil, err
	}
	return raw, nil
}

func (s *plainStream) Write(ctx context.Context, data []byte) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	s.outMu.Lock()
	_, err := s.out.Write(data)
	s.outMu.Unlock()
	return err
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

func (s *headerStream) Read(ctx context.Context) ([]byte, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	var length int64
	// read the header, stop on the first empty line
	for {
		line, err := s.in.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("failed reading header line %q", err)
		}
		line = strings.TrimSpace(line)
		// check we have a header line
		if line == "" {
			break
		}
		colon := strings.IndexRune(line, ':')
		if colon < 0 {
			return nil, fmt.Errorf("invalid header line %q", line)
		}
		name, value := line[:colon], strings.TrimSpace(line[colon+1:])
		switch name {
		case "Content-Length":
			if length, err = strconv.ParseInt(value, 10, 32); err != nil {
				return nil, fmt.Errorf("failed parsing Content-Length: %v", value)
			}
			if length <= 0 {
				return nil, fmt.Errorf("invalid Content-Length: %v", length)
			}
		default:
			// ignoring unknown headers
		}
	}
	if length == 0 {
		return nil, fmt.Errorf("missing Content-Length header")
	}
	data := make([]byte, length)
	if _, err := io.ReadFull(s.in, data); err != nil {
		return nil, err
	}
	return data, nil
}

func (s *headerStream) Write(ctx context.Context, data []byte) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	s.outMu.Lock()
	_, err := fmt.Fprintf(s.out, "Content-Length: %v\r\n\r\n", len(data))
	if err == nil {
		_, err = s.out.Write(data)
	}
	s.outMu.Unlock()
	return err
}

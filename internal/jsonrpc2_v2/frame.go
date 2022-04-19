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
)

// Reader abstracts the transport mechanics from the JSON RPC protocol.
// A Conn reads messages from the reader it was provided on construction,
// and assumes that each call to Read fully transfers a single message,
// or returns an error.
// A reader is not safe for concurrent use, it is expected it will be used by
// a single Conn in a safe manner.
type Reader interface {
	// Read gets the next message from the stream.
	Read(context.Context) (Message, int64, error)
}

// Writer abstracts the transport mechanics from the JSON RPC protocol.
// A Conn writes messages using the writer it was provided on construction,
// and assumes that each call to Write fully transfers a single message,
// or returns an error.
// A writer is not safe for concurrent use, it is expected it will be used by
// a single Conn in a safe manner.
type Writer interface {
	// Write sends a message to the stream.
	Write(context.Context, Message) (int64, error)
}

// Framer wraps low level byte readers and writers into jsonrpc2 message
// readers and writers.
// It is responsible for the framing and encoding of messages into wire form.
type Framer interface {
	// Reader wraps a byte reader into a message reader.
	Reader(rw io.Reader) Reader
	// Writer wraps a byte writer into a message writer.
	Writer(rw io.Writer) Writer
}

// RawFramer returns a new Framer.
// The messages are sent with no wrapping, and rely on json decode consistency
// to determine message boundaries.
func RawFramer() Framer { return rawFramer{} }

type rawFramer struct{}
type rawReader struct{ in *json.Decoder }
type rawWriter struct{ out io.Writer }

func (rawFramer) Reader(rw io.Reader) Reader {
	return &rawReader{in: json.NewDecoder(rw)}
}

func (rawFramer) Writer(rw io.Writer) Writer {
	return &rawWriter{out: rw}
}

func (r *rawReader) Read(ctx context.Context) (Message, int64, error) {
	select {
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	default:
	}
	var raw json.RawMessage
	if err := r.in.Decode(&raw); err != nil {
		return nil, 0, err
	}
	msg, err := DecodeMessage(raw)
	return msg, int64(len(raw)), err
}

func (w *rawWriter) Write(ctx context.Context, msg Message) (int64, error) {
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	default:
	}
	data, err := EncodeMessage(msg)
	if err != nil {
		return 0, fmt.Errorf("marshaling message: %v", err)
	}
	n, err := w.out.Write(data)
	return int64(n), err
}

// HeaderFramer returns a new Framer.
// The messages are sent with HTTP content length and MIME type headers.
// This is the format used by LSP and others.
func HeaderFramer() Framer { return headerFramer{} }

type headerFramer struct{}
type headerReader struct{ in *bufio.Reader }
type headerWriter struct{ out io.Writer }

func (headerFramer) Reader(rw io.Reader) Reader {
	return &headerReader{in: bufio.NewReader(rw)}
}

func (headerFramer) Writer(rw io.Writer) Writer {
	return &headerWriter{out: rw}
}

func (r *headerReader) Read(ctx context.Context) (Message, int64, error) {
	select {
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	default:
	}
	var total, length int64
	// read the header, stop on the first empty line
	for {
		line, err := r.in.ReadString('\n')
		total += int64(len(line))
		if err != nil {
			return nil, total, fmt.Errorf("failed reading header line: %w", err)
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
	n, err := io.ReadFull(r.in, data)
	total += int64(n)
	if err != nil {
		return nil, total, err
	}
	msg, err := DecodeMessage(data)
	return msg, total, err
}

func (w *headerWriter) Write(ctx context.Context, msg Message) (int64, error) {
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	default:
	}
	data, err := EncodeMessage(msg)
	if err != nil {
		return 0, fmt.Errorf("marshaling message: %v", err)
	}
	n, err := fmt.Fprintf(w.out, "Content-Length: %v\r\n\r\n", len(data))
	total := int64(n)
	if err == nil {
		n, err = w.out.Write(data)
		total += int64(n)
	}
	return total, err
}

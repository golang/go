// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"fmt"
	"net"
	"time"
)

// A ResponseController is used by an HTTP handler to control the response.
//
// A ResponseController may not be used after the [Handler.ServeHTTP] method has returned.
type ResponseController struct {
	rw ResponseWriter
}

// NewResponseController creates a [ResponseController] for a request.
//
// The ResponseWriter should be the original value passed to the [Handler.ServeHTTP] method,
// or have an Unwrap method returning the original ResponseWriter.
//
// If the ResponseWriter implements any of the following methods, the ResponseController
// will call them as appropriate:
//
//	Flush()
//	FlushError() error // alternative Flush returning an error
//	Hijack() (net.Conn, *bufio.ReadWriter, error)
//	SetReadDeadline(deadline time.Time) error
//	SetWriteDeadline(deadline time.Time) error
//	EnableFullDuplex() error
//
// If the ResponseWriter does not support a method, ResponseController returns
// an error matching [ErrNotSupported].
func NewResponseController(rw ResponseWriter) *ResponseController {
	return &ResponseController{rw}
}

type rwUnwrapper interface {
	Unwrap() ResponseWriter
}

// Flush flushes buffered data to the client.
func (c *ResponseController) Flush() error {
	rw := c.rw
	for {
		switch t := rw.(type) {
		case interface{ FlushError() error }:
			return t.FlushError()
		case Flusher:
			t.Flush()
			return nil
		case rwUnwrapper:
			rw = t.Unwrap()
		default:
			return errNotSupported()
		}
	}
}

// Hijack lets the caller take over the connection.
// See the Hijacker interface for details.
func (c *ResponseController) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	rw := c.rw
	for {
		switch t := rw.(type) {
		case Hijacker:
			return t.Hijack()
		case rwUnwrapper:
			rw = t.Unwrap()
		default:
			return nil, nil, errNotSupported()
		}
	}
}

// SetReadDeadline sets the deadline for reading the entire request, including the body.
// Reads from the request body after the deadline has been exceeded will return an error.
// A zero value means no deadline.
//
// Setting the read deadline after it has been exceeded will not extend it.
func (c *ResponseController) SetReadDeadline(deadline time.Time) error {
	rw := c.rw
	for {
		switch t := rw.(type) {
		case interface{ SetReadDeadline(time.Time) error }:
			return t.SetReadDeadline(deadline)
		case rwUnwrapper:
			rw = t.Unwrap()
		default:
			return errNotSupported()
		}
	}
}

// SetWriteDeadline sets the deadline for writing the response.
// Writes to the response body after the deadline has been exceeded will not block,
// but may succeed if the data has been buffered.
// A zero value means no deadline.
//
// Setting the write deadline after it has been exceeded will not extend it.
func (c *ResponseController) SetWriteDeadline(deadline time.Time) error {
	rw := c.rw
	for {
		switch t := rw.(type) {
		case interface{ SetWriteDeadline(time.Time) error }:
			return t.SetWriteDeadline(deadline)
		case rwUnwrapper:
			rw = t.Unwrap()
		default:
			return errNotSupported()
		}
	}
}

// EnableFullDuplex indicates that the request handler will interleave reads from [Request.Body]
// with writes to the [ResponseWriter].
//
// For HTTP/1 requests, the Go HTTP server by default consumes any unread portion of
// the request body before beginning to write the response, preventing handlers from
// concurrently reading from the request and writing the response.
// Calling EnableFullDuplex disables this behavior and permits handlers to continue to read
// from the request while concurrently writing the response.
//
// For HTTP/2 requests, the Go HTTP server always permits concurrent reads and responses.
func (c *ResponseController) EnableFullDuplex() error {
	rw := c.rw
	for {
		switch t := rw.(type) {
		case interface{ EnableFullDuplex() error }:
			return t.EnableFullDuplex()
		case rwUnwrapper:
			rw = t.Unwrap()
		default:
			return errNotSupported()
		}
	}
}

// errNotSupported returns an error that Is ErrNotSupported,
// but is not == to it.
func errNotSupported() error {
	return fmt.Errorf("%w", ErrNotSupported)
}

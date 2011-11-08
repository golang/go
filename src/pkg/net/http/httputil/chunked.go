// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httputil

import (
	"bufio"
	"io"
	"net/http"
	"strconv"
	"strings"
)

// NewChunkedWriter returns a new writer that translates writes into HTTP
// "chunked" format before writing them to w. Closing the returned writer
// sends the final 0-length chunk that marks the end of the stream.
//
// NewChunkedWriter is not needed by normal applications. The http
// package adds chunking automatically if handlers don't set a
// Content-Length header. Using NewChunkedWriter inside a handler
// would result in double chunking or chunking with a Content-Length
// length, both of which are wrong.
func NewChunkedWriter(w io.Writer) io.WriteCloser {
	return &chunkedWriter{w}
}

// Writing to ChunkedWriter translates to writing in HTTP chunked Transfer
// Encoding wire format to the underlying Wire writer.
type chunkedWriter struct {
	Wire io.Writer
}

// Write the contents of data as one chunk to Wire.
// NOTE: Note that the corresponding chunk-writing procedure in Conn.Write has
// a bug since it does not check for success of io.WriteString
func (cw *chunkedWriter) Write(data []byte) (n int, err error) {

	// Don't send 0-length data. It looks like EOF for chunked encoding.
	if len(data) == 0 {
		return 0, nil
	}

	head := strconv.Itob(len(data), 16) + "\r\n"

	if _, err = io.WriteString(cw.Wire, head); err != nil {
		return 0, err
	}
	if n, err = cw.Wire.Write(data); err != nil {
		return
	}
	if n != len(data) {
		err = io.ErrShortWrite
		return
	}
	_, err = io.WriteString(cw.Wire, "\r\n")

	return
}

func (cw *chunkedWriter) Close() error {
	_, err := io.WriteString(cw.Wire, "0\r\n")
	return err
}

// NewChunkedReader returns a new reader that translates the data read from r
// out of HTTP "chunked" format before returning it. 
// The reader returns io.EOF when the final 0-length chunk is read.
//
// NewChunkedReader is not needed by normal applications. The http package
// automatically decodes chunking when reading response bodies.
func NewChunkedReader(r io.Reader) io.Reader {
	// This is a bit of a hack so we don't have to copy chunkedReader into
	// httputil.  It's a bit more complex than chunkedWriter, which is copied
	// above.
	req, err := http.ReadRequest(bufio.NewReader(io.MultiReader(
		strings.NewReader("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"),
		r,
		strings.NewReader("\r\n"))))
	if err != nil {
		panic("bad fake request: " + err.Error())
	}
	return req.Body
}

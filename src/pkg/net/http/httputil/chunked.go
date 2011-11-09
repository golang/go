// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The wire protocol for HTTP's "chunked" Transfer-Encoding.

// This code is a duplicate of ../chunked.go with these edits:
//	s/newChunked/NewChunked/g
//	s/package http/package httputil/
// Please make any changes in both files.

package httputil

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"strconv"
)

const maxLineLength = 4096 // assumed <= bufio.defaultBufSize

var ErrLineTooLong = errors.New("header line too long")

// NewChunkedReader returns a new chunkedReader that translates the data read from r
// out of HTTP "chunked" format before returning it. 
// The chunkedReader returns io.EOF when the final 0-length chunk is read.
//
// NewChunkedReader is not needed by normal applications. The http package
// automatically decodes chunking when reading response bodies.
func NewChunkedReader(r io.Reader) io.Reader {
	br, ok := r.(*bufio.Reader)
	if !ok {
		br = bufio.NewReader(r)
	}
	return &chunkedReader{r: br}
}

type chunkedReader struct {
	r   *bufio.Reader
	n   uint64 // unread bytes in chunk
	err error
}

func (cr *chunkedReader) beginChunk() {
	// chunk-size CRLF
	var line string
	line, cr.err = readLine(cr.r)
	if cr.err != nil {
		return
	}
	cr.n, cr.err = strconv.Btoui64(line, 16)
	if cr.err != nil {
		return
	}
	if cr.n == 0 {
		cr.err = io.EOF
	}
}

func (cr *chunkedReader) Read(b []uint8) (n int, err error) {
	if cr.err != nil {
		return 0, cr.err
	}
	if cr.n == 0 {
		cr.beginChunk()
		if cr.err != nil {
			return 0, cr.err
		}
	}
	if uint64(len(b)) > cr.n {
		b = b[0:cr.n]
	}
	n, cr.err = cr.r.Read(b)
	cr.n -= uint64(n)
	if cr.n == 0 && cr.err == nil {
		// end of chunk (CRLF)
		b := make([]byte, 2)
		if _, cr.err = io.ReadFull(cr.r, b); cr.err == nil {
			if b[0] != '\r' || b[1] != '\n' {
				cr.err = errors.New("malformed chunked encoding")
			}
		}
	}
	return n, cr.err
}

// Read a line of bytes (up to \n) from b.
// Give up if the line exceeds maxLineLength.
// The returned bytes are a pointer into storage in
// the bufio, so they are only valid until the next bufio read.
func readLineBytes(b *bufio.Reader) (p []byte, err error) {
	if p, err = b.ReadSlice('\n'); err != nil {
		// We always know when EOF is coming.
		// If the caller asked for a line, there should be a line.
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		} else if err == bufio.ErrBufferFull {
			err = ErrLineTooLong
		}
		return nil, err
	}
	if len(p) >= maxLineLength {
		return nil, ErrLineTooLong
	}

	// Chop off trailing white space.
	p = bytes.TrimRight(p, " \r\t\n")

	return p, nil
}

// readLineBytes, but convert the bytes into a string.
func readLine(b *bufio.Reader) (s string, err error) {
	p, e := readLineBytes(b)
	if e != nil {
		return "", e
	}
	return string(p), nil
}

// NewChunkedWriter returns a new chunkedWriter that translates writes into HTTP
// "chunked" format before writing them to w. Closing the returned chunkedWriter
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

// Writing to chunkedWriter translates to writing in HTTP chunked Transfer
// Encoding wire format to the underlying Wire chunkedWriter.
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

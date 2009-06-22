// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package provides basic interfaces to I/O primitives.
// Its primary job is to wrap existing implementations of such primitives,
// such as those in package os, into shared public interfaces that
// abstract the functionality.
// It also provides buffering primitives and some other basic operations.
package io

import (
	"bytes";
	"os";
)

// Error represents an unexpected I/O behavior.
type Error struct {
	os.ErrorString
}

// ErrShortWrite means that a write accepted fewer bytes than requested
// but failed to return an explicit error.
var ErrShortWrite os.Error = &Error{"short write"}

// ErrUnexpectedEOF means that os.EOF was encountered in the
// middle of reading a fixed-size block or data structure.
var ErrUnexpectedEOF os.Error = &Error{"unexpected EOF"}

// Reader is the interface that wraps the basic Read method.
//
// Read reads up to len(p) bytes into p.  It returns the number of bytes
// read (0 <= n <= len(p)) and any error encountered.
// Even if Read returns n < len(p),
// it may use all of p as scratch space during the call.
// If some data is available but not len(p) bytes, Read conventionally
// returns what is available rather than block waiting for more.
//
// At the end of the input stream, Read returns 0, os.EOF.
// Read may return a non-zero number of bytes with a non-nil err.
// In particular, a Read that exhausts the input may return n > 0, os.EOF.
type Reader interface {
	Read(p []byte) (n int, err os.Error);
}

// Writer is the interface that wraps the basic Write method.
//
// Write writes len(p) bytes from p to the underlying data stream.
// It returns the number of bytes written from p (0 <= n <= len(p))
// and any error encountered that caused the write to stop early.
// Write must return a non-nil error if it returns n < len(p).
type Writer interface {
	Write(p []byte) (n int, err os.Error);
}

// Closer is the interface that wraps the basic Close method.
type Closer interface {
	Close() os.Error;
}

// ReadWrite is the interface that groups the basic Read and Write methods.
type ReadWriter interface {
	Reader;
	Writer;
}

// ReadCloser is the interface that groups the basic Read and Close methods.
type ReadCloser interface {
	Reader;
	Closer;
}

// WriteCloser is the interface that groups the basic Write and Close methods.
type WriteCloser interface {
	Writer;
	Closer;
}

// ReadWriteCloser is the interface that groups the basic Read, Write and Close methods.
type ReadWriteCloser interface {
	Reader;
	Writer;
	Closer;
}

// Convert a string to an array of bytes for easy marshaling.
func StringBytes(s string) []byte {
	b := make([]byte, len(s));
	for i := 0; i < len(s); i++ {
		b[i] = s[i];
	}
	return b;
}

// WriteString writes the contents of the string s to w, which accepts an array of bytes.
func WriteString(w Writer, s string) (n int, err os.Error) {
	return w.Write(StringBytes(s))
}

// ReadAtLeast reads from r into buf until it has read at least min bytes.
// It returns the number of bytes copied and an error if fewer bytes were read.
// The error is os.EOF only if no bytes were read.
// If an EOF happens after reading fewer than min bytes,
// ReadAtLeast returns ErrUnexpectedEOF.
func ReadAtLeast(r Reader, buf []byte, min int) (n int, err os.Error) {
	n = 0;
	for n < min {
		nn, e := r.Read(buf[n:len(buf)]);
		if nn > 0 {
			n += nn
		}
		if e != nil {
			if e == os.EOF && n > 0 {
				e = ErrUnexpectedEOF;
			}
			return n, e
		}
	}
	return n, nil
}

// ReadFull reads exactly len(buf) bytes from r into buf.
// It returns the number of bytes copied and an error if fewer bytes were read.
// The error is os.EOF only if no bytes were read.
// If an EOF happens after reading some but not all the bytes,
// ReadFull returns ErrUnexpectedEOF.
func ReadFull(r Reader, buf []byte) (n int, err os.Error) {
	// TODO(rsc): 6g bug keeps us from writing the obvious 1-liner
	n, err = ReadAtLeast(r, buf, len(buf));
	return;
}

// Copyn copies n bytes (or until an error) from src to dst.
// It returns the number of bytes copied and the error, if any.
func Copyn(src Reader, dst Writer, n int64) (written int64, err os.Error) {
	buf := make([]byte, 32*1024);
	for written < n {
		l := len(buf);
		if d := n - written; d < int64(l) {
			l = int(d);
		}
		nr, er := src.Read(buf[0 : l]);
		if nr > 0 {
			nw, ew := dst.Write(buf[0 : nr]);
			if nw > 0 {
				written += int64(nw);
			}
			if ew != nil {
				err = ew;
				break;
			}
			if nr != nw {
				err = ErrShortWrite;
				break;
			}
		}
		if er != nil {
			err = er;
			break;
		}
	}
	return written, err
}

// Copy copies from src to dst until either EOF is reached
// on src or an error occurs.  It returns the number of bytes
// copied and the error, if any.
func Copy(src Reader, dst Writer) (written int64, err os.Error) {
	buf := make([]byte, 32*1024);
	for {
		nr, er := src.Read(buf);
		if nr > 0 {
			nw, ew := dst.Write(buf[0:nr]);
			if nw > 0 {
				written += int64(nw);
			}
			if ew != nil {
				err = ew;
				break;
			}
			if nr != nw {
				err = ErrShortWrite;
				break;
			}
		}
		if er == os.EOF {
			break;
		}
		if er != nil {
			err = er;
			break;
		}
	}
	return written, err
}

// A ByteReader satisfies Reads by consuming data from a slice of bytes.
// Clients can call NewByteReader to create one or wrap pointers
// to their own slices: r := ByteReader{&data}.
type ByteReader struct {
	Data *[]byte
}

func (r ByteReader) Read(p []byte) (int, os.Error) {
	n := len(p);
	b := *r.Data;
	if len(b) == 0 {
		return 0, os.EOF;
	}
	if n > len(b) {
		n = len(b);
	}
	bytes.Copy(p, b[0:n]);
	*r.Data = b[n:len(b)];
	return n, nil;
}

// NewByteReader returns a new ByteReader reading from data.
func NewByteReader(data []byte) ByteReader {
	return ByteReader{ &data };
}


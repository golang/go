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

// ErrEOF means that data was expected, but a read got EOF instead.
var ErrEOF os.Error = &Error{"EOF"}

// ErrShortWrite means that a write accepted fewer bytes than requested
// but failed to return an explicit error.
var ErrShortWrite os.Error = &Error{"short write"}


// Reader is the interface that wraps the basic Read method.
// An implementation of Read is allowed to use all of p for
// scratch space during the call, even if it eventually returns
// n < len(p).
type Reader interface {
	Read(p []byte) (n int, err os.Error);
}

// Writer is the interface that wraps the basic Write method.
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

// ReadAtLeast reads r into buf until at least min bytes have been read,
// or until EOF or error.
func ReadAtLeast(r Reader, buf []byte, min int) (n int, err os.Error) {
	n = 0;
	for n < min {
		nn, e := r.Read(buf[n:len(buf)]);
		if nn > 0 {
			n += nn
		}
		if e != nil {
			return n, e
		}
		if nn <= 0 {
			return n, ErrEOF	// no error but insufficient data
		}
	}
	return n, nil
}

// FullRead reads r until the buffer buf is full, or until EOF or error.
func FullRead(r Reader, buf []byte) (n int, err os.Error) {
	// TODO(rsc): 6g bug prevents obvious return
	n, err = ReadAtLeast(r, buf, len(buf));
	return;
}

// Convert something that implements Read into something
// whose Reads are always FullReads
type fullRead struct {
	r	Reader;
}

func (fr *fullRead) Read(p []byte) (n int, err os.Error) {
	n, err = FullRead(fr.r, p);
	return n, err
}

// MakeFullReader takes r, an implementation of Read, and returns an object
// that still implements Read but always calls FullRead underneath.
func MakeFullReader(r Reader) Reader {
	if fr, ok := r.(*fullRead); ok {
		// already a fullRead
		return r
	}
	return &fullRead{r}
}

// Copy n copies n bytes (or until EOF is reached) from src to dst.
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
				err = os.EIO;
				break;
			}
		}
		if er != nil {
			err = er;
			break;
		}
		if nr == 0 {
			err = ErrEOF;
			break;
		}
	}
	return written, err
}

// Copy copies from src to dst until EOF is reached.
// It returns the number of bytes copied and the error, if any.
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
				err = os.EIO;
				break;
			}
		}
		if er != nil {
			err = er;
			break;
		}
		if nr == 0 {
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
	b := r.Data;
	if n > len(b) {
		n = len(b);
	}
	bytes.Copy(p, b[0:n]);
	*b = b[n:len(b)];
	return n, nil;
}

// NewByteReader returns a new ByteReader reading from data.
func NewByteReader(data []byte) ByteReader {
	return ByteReader{ &data };
}


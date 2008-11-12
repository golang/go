// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io

import (
	"os";
	"syscall";
)

export var ErrEOF = os.NewError("EOF")

export type Read interface {
	Read(p *[]byte) (n int, err *os.Error);
}

export type Write interface {
	Write(p *[]byte) (n int, err *os.Error);
}

export type ReadWrite interface {
	Read(p *[]byte) (n int, err *os.Error);
	Write(p *[]byte) (n int, err *os.Error);
}

export type ReadWriteClose interface {
	Read(p *[]byte) (n int, err *os.Error);
	Write(p *[]byte) (n int, err *os.Error);
	Close() *os.Error;
}

export func WriteString(w Write, s string) (n int, err *os.Error) {
	b := new([]byte, len(s)+1);
	if !syscall.StringToBytes(b, s) {
		return -1, os.EINVAL
	}
	// BUG return w.Write(b[0:len(s)])
	r, e := w.Write(b[0:len(s)]);
	return r, e
}

// Read until buffer is full,  EOF, or error
export func Readn(fd Read, buf *[]byte) (n int, err *os.Error) {
	n = 0;
	for n < len(buf) {
		nn, e := fd.Read(buf[n:len(buf)]);
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

// Convert something that implements Read into something
// whose Reads are always Readn
type FullRead struct {
	fd	Read;
}

func (fd *FullRead) Read(p *[]byte) (n int, err *os.Error) {
	n, err = Readn(fd, p);
	return n, err
}

export func MakeFullReader(fd Read) Read {
	if fr, ok := fd.(*FullRead); ok {
		// already a FullRead
		return fd
	}
	return &FullRead{fd}
}

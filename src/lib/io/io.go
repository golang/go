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
	Read(p []byte) (n int, err *os.Error);
}

export type Write interface {
	Write(p []byte) (n int, err *os.Error);
}

export type ReadWrite interface {
	Read(p []byte) (n int, err *os.Error);
	Write(p []byte) (n int, err *os.Error);
}

export type ReadWriteClose interface {
	Read(p []byte) (n int, err *os.Error);
	Write(p []byte) (n int, err *os.Error);
	Close() *os.Error;
}

export func WriteString(w Write, s string) (n int, err *os.Error) {
	b := make([]byte, len(s)+1);
	if !syscall.StringToBytes(b, s) {
		return -1, os.EINVAL
	}
	// BUG return w.Write(b[0:len(s)])
	r, e := w.Write(b[0:len(s)]);
	return r, e
}

// Read until buffer is full, EOF, or error
export func Readn(fd Read, buf []byte) (n int, err *os.Error) {
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
type _FullRead struct {
	fd	Read;
}

func (fd *_FullRead) Read(p []byte) (n int, err *os.Error) {
	n, err = Readn(fd.fd, p);
	return n, err
}

export func Make_FullReader(fd Read) Read {
	if fr, ok := fd.(*_FullRead); ok {
		// already a _FullRead
		return fd
	}
	return &_FullRead{fd}
}

// Copies n bytes (or until EOF is reached) from src to dst.
// Returns the number of bytes copied and the error, if any.
export func Copyn(src Read, dst Write, n int64) (written int64, err *os.Error) {
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

// Copies from src to dst until EOF is reached.
// Returns the number of bytes copied and the error, if any.
export func Copy(src Read, dst Write) (written int64, err *os.Error) {
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

// Convert a string to an array of bytes for easy marshaling.
// Could fill with syscall.StringToBytes but it adds an unnecessary \000
// so the length would be wrong.
export func StringBytes(s string) []byte {
	b := make([]byte, len(s));
	for i := 0; i < len(s); i++ {
		b[i] = s[i];
	}
	return b;
}

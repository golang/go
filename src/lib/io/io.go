// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io

import (
	"os";
	"syscall";
)

var ErrEOF = os.NewError("EOF")

type Read interface {
	Read(p []byte) (n int, err *os.Error);
}

type Write interface {
	Write(p []byte) (n int, err *os.Error);
}

type Close interface {
	Close() *os.Error;
}

type ReadWrite interface {
	Read(p []byte) (n int, err *os.Error);
	Write(p []byte) (n int, err *os.Error);
}

type ReadClose interface {
	Read(p []byte) (n int, err *os.Error);
	Close() *os.Error;
}

type WriteClose interface {
	Write(p []byte) (n int, err *os.Error);
	Close() *os.Error;
}

type ReadWriteClose interface {
	Read(p []byte) (n int, err *os.Error);
	Write(p []byte) (n int, err *os.Error);
	Close() *os.Error;
}

// Convert a string to an array of bytes for easy marshaling.
// Could fill with syscall.StringToBytes but it adds an unnecessary \000
// so the length would be wrong.
func StringBytes(s string) []byte {
	b := make([]byte, len(s));
	for i := 0; i < len(s); i++ {
		b[i] = s[i];
	}
	return b;
}

func WriteString(w Write, s string) (n int, err *os.Error) {
	return w.Write(StringBytes(s))
}

// Read until buffer is full, EOF, or error
func Readn(fd Read, buf []byte) (n int, err *os.Error) {
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
type fullRead struct {
	fd	Read;
}

func (fd *fullRead) Read(p []byte) (n int, err *os.Error) {
	n, err = Readn(fd.fd, p);
	return n, err
}

func MakeFullReader(fd Read) Read {
	if fr, ok := fd.(*fullRead); ok {
		// already a fullRead
		return fd
	}
	return &fullRead(fd)
}

// Copies n bytes (or until EOF is reached) from src to dst.
// Returns the number of bytes copied and the error, if any.
func Copyn(src Read, dst Write, n int64) (written int64, err *os.Error) {
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
func Copy(src Read, dst Write) (written int64, err *os.Error) {
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

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package provides basic interfaces to I/O primitives.
// Its primary job is to wrap existing implementations of such primitives,
// such as those in package os, into shared public interfaces that
// abstract the functionality, plus some other related primitives.
package io

import "os"

// Error represents an unexpected I/O behavior.
type Error struct {
	os.ErrorString
}

// ErrShortWrite means that a write accepted fewer bytes than requested
// but failed to return an explicit error.
var ErrShortWrite os.Error = &Error{"short write"}

// ErrShortBuffer means that a read required a longer buffer than was provided.
var ErrShortBuffer os.Error = &Error{"short buffer"}

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
	Read(p []byte) (n int, err os.Error)
}

// Writer is the interface that wraps the basic Write method.
//
// Write writes len(p) bytes from p to the underlying data stream.
// It returns the number of bytes written from p (0 <= n <= len(p))
// and any error encountered that caused the write to stop early.
// Write must return a non-nil error if it returns n < len(p).
type Writer interface {
	Write(p []byte) (n int, err os.Error)
}

// Closer is the interface that wraps the basic Close method.
type Closer interface {
	Close() os.Error
}

// Seeker is the interface that wraps the basic Seek method.
//
// Seek sets the offset for the next Read or Write to offset,
// interpreted according to whence: 0 means relative to the origin of
// the file, 1 means relative to the current offset, and 2 means
// relative to the end.  Seek returns the new offset and an Error, if
// any.
type Seeker interface {
	Seek(offset int64, whence int) (ret int64, err os.Error)
}

// ReadWriter is the interface that groups the basic Read and Write methods.
type ReadWriter interface {
	Reader
	Writer
}

// ReadCloser is the interface that groups the basic Read and Close methods.
type ReadCloser interface {
	Reader
	Closer
}

// WriteCloser is the interface that groups the basic Write and Close methods.
type WriteCloser interface {
	Writer
	Closer
}

// ReadWriteCloser is the interface that groups the basic Read, Write and Close methods.
type ReadWriteCloser interface {
	Reader
	Writer
	Closer
}

// ReadSeeker is the interface that groups the basic Read and Seek methods.
type ReadSeeker interface {
	Reader
	Seeker
}

// WriteSeeker is the interface that groups the basic Write and Seek methods.
type WriteSeeker interface {
	Writer
	Seeker
}

// ReadWriteSeeker is the interface that groups the basic Read, Write and Seek methods.
type ReadWriteSeeker interface {
	Reader
	Writer
	Seeker
}

// ReaderFrom is the interface that wraps the ReadFrom method.
type ReaderFrom interface {
	ReadFrom(r Reader) (n int64, err os.Error)
}

// WriterTo is the interface that wraps the WriteTo method.
type WriterTo interface {
	WriteTo(w Writer) (n int64, err os.Error)
}

// ReaderAt is the interface that wraps the basic ReadAt method.
//
// ReadAt reads len(p) bytes into p starting at offset off in the
// underlying data stream.  It returns the number of bytes
// read (0 <= n <= len(p)) and any error encountered.
//
// Even if ReadAt returns n < len(p),
// it may use all of p as scratch space during the call.
// If some data is available but not len(p) bytes, ReadAt blocks
// until either all the data is available or an error occurs.
//
// At the end of the input stream, ReadAt returns 0, os.EOF.
// ReadAt may return a non-zero number of bytes with a non-nil err.
// In particular, a ReadAt that exhausts the input may return n > 0, os.EOF.
type ReaderAt interface {
	ReadAt(p []byte, off int64) (n int, err os.Error)
}

// WriterAt is the interface that wraps the basic WriteAt method.
//
// WriteAt writes len(p) bytes from p to the underlying data stream
// at offset off.  It returns the number of bytes written from p (0 <= n <= len(p))
// and any error encountered that caused the write to stop early.
// WriteAt must return a non-nil error if it returns n < len(p).
type WriterAt interface {
	WriteAt(p []byte, off int64) (n int, err os.Error)
}

// ReadByter is the interface that wraps the ReadByte method.
//
// ReadByte reads and returns the next byte from the input.
// If no byte is available, err will be set.
type ReadByter interface {
	ReadByte() (c byte, err os.Error)
}

// WriteString writes the contents of the string s to w, which accepts an array of bytes.
func WriteString(w Writer, s string) (n int, err os.Error) {
	return w.Write([]byte(s))
}

// ReadAtLeast reads from r into buf until it has read at least min bytes.
// It returns the number of bytes copied and an error if fewer bytes were read.
// The error is os.EOF only if no bytes were read.
// If an EOF happens after reading fewer than min bytes,
// ReadAtLeast returns ErrUnexpectedEOF.
// If min is greater than the length of buf, ReadAtLeast returns ErrShortBuffer.
func ReadAtLeast(r Reader, buf []byte, min int) (n int, err os.Error) {
	if len(buf) < min {
		return 0, ErrShortBuffer
	}
	for n < min {
		nn, e := r.Read(buf[n:])
		if nn > 0 {
			n += nn
		}
		if e != nil {
			if e == os.EOF && n > 0 {
				e = ErrUnexpectedEOF
			}
			return n, e
		}
	}
	return
}

// ReadFull reads exactly len(buf) bytes from r into buf.
// It returns the number of bytes copied and an error if fewer bytes were read.
// The error is os.EOF only if no bytes were read.
// If an EOF happens after reading some but not all the bytes,
// ReadFull returns ErrUnexpectedEOF.
func ReadFull(r Reader, buf []byte) (n int, err os.Error) {
	return ReadAtLeast(r, buf, len(buf))
}

// Copyn copies n bytes (or until an error) from src to dst.
// It returns the number of bytes copied and the error, if any.
//
// If dst implements the ReaderFrom interface,
// the copy is implemented by calling dst.ReadFrom(src).
func Copyn(dst Writer, src Reader, n int64) (written int64, err os.Error) {
	// If the writer has a ReadFrom method, use it to do the copy.
	// Avoids a buffer allocation and a copy.
	if rt, ok := dst.(ReaderFrom); ok {
		written, err = rt.ReadFrom(LimitReader(src, n))
		if written < n && err == nil {
			// rt stopped early; must have been EOF.
			err = os.EOF
		}
		return
	}
	buf := make([]byte, 32*1024)
	for written < n {
		l := len(buf)
		if d := n - written; d < int64(l) {
			l = int(d)
		}
		nr, er := src.Read(buf[0:l])
		if nr > 0 {
			nw, ew := dst.Write(buf[0:nr])
			if nw > 0 {
				written += int64(nw)
			}
			if ew != nil {
				err = ew
				break
			}
			if nr != nw {
				err = ErrShortWrite
				break
			}
		}
		if er != nil {
			err = er
			break
		}
	}
	return written, err
}

// Copy copies from src to dst until either EOF is reached
// on src or an error occurs.  It returns the number of bytes
// copied and the error, if any.
//
// If dst implements the ReaderFrom interface,
// the copy is implemented by calling dst.ReadFrom(src).
// Otherwise, if src implements the WriterTo interface,
// the copy is implemented by calling src.WriteTo(dst).
func Copy(dst Writer, src Reader) (written int64, err os.Error) {
	// If the writer has a ReadFrom method, use it to do the copy.
	// Avoids an allocation and a copy.
	if rt, ok := dst.(ReaderFrom); ok {
		return rt.ReadFrom(src)
	}
	// Similarly, if the reader has a WriteTo method, use it to do the copy.
	if wt, ok := src.(WriterTo); ok {
		return wt.WriteTo(dst)
	}
	buf := make([]byte, 32*1024)
	for {
		nr, er := src.Read(buf)
		if nr > 0 {
			nw, ew := dst.Write(buf[0:nr])
			if nw > 0 {
				written += int64(nw)
			}
			if ew != nil {
				err = ew
				break
			}
			if nr != nw {
				err = ErrShortWrite
				break
			}
		}
		if er == os.EOF {
			break
		}
		if er != nil {
			err = er
			break
		}
	}
	return written, err
}

// LimitReader returns a Reader that reads from r
// but stops with os.EOF after n bytes.
func LimitReader(r Reader, n int64) Reader { return &limitedReader{r, n} }

type limitedReader struct {
	r Reader
	n int64
}

func (l *limitedReader) Read(p []byte) (n int, err os.Error) {
	if l.n <= 0 {
		return 0, os.EOF
	}
	if int64(len(p)) > l.n {
		p = p[0:l.n]
	}
	n, err = l.r.Read(p)
	l.n -= int64(n)
	return
}

// NewSectionReader returns a SectionReader that reads from r
// starting at offset off and stops with os.EOF after n bytes.
func NewSectionReader(r ReaderAt, off int64, n int64) *SectionReader {
	return &SectionReader{r, off, off, off + n}
}

// SectionReader implements Read, Seek, and ReadAt on a section
// of an underlying ReaderAt.
type SectionReader struct {
	r     ReaderAt
	base  int64
	off   int64
	limit int64
}

func (s *SectionReader) Read(p []byte) (n int, err os.Error) {
	if s.off >= s.limit {
		return 0, os.EOF
	}
	if max := s.limit - s.off; int64(len(p)) > max {
		p = p[0:max]
	}
	n, err = s.r.ReadAt(p, s.off)
	s.off += int64(n)
	return
}

func (s *SectionReader) Seek(offset int64, whence int) (ret int64, err os.Error) {
	switch whence {
	default:
		return 0, os.EINVAL
	case 0:
		offset += s.base
	case 1:
		offset += s.off
	case 2:
		offset += s.limit
	}
	if offset < s.base || offset > s.limit {
		return 0, os.EINVAL
	}
	s.off = offset
	return offset - s.base, nil
}

func (s *SectionReader) ReadAt(p []byte, off int64) (n int, err os.Error) {
	if off < 0 || off >= s.limit-s.base {
		return 0, os.EOF
	}
	off += s.base
	if max := s.limit - off; int64(len(p)) > max {
		p = p[0:max]
	}
	return s.r.ReadAt(p, off)
}

// Size returns the size of the section in bytes.
func (s *SectionReader) Size() int64 { return s.limit - s.base }

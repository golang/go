// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package io provides basic interfaces to I/O primitives.
// Its primary job is to wrap existing implementations of such primitives,
// such as those in package os, into shared public interfaces that
// abstract the functionality, plus some other related primitives.
package io

import "os"

// Error represents an unexpected I/O behavior.
type Error struct {
	ErrorString string
}

func (err *Error) String() string { return err.ErrorString }

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
// read (0 <= n <= len(p)) and any error encountered.  Even if Read
// returns n < len(p), it may use all of p as scratch space during the call.
// If some data is available but not len(p) bytes, Read conventionally
// returns what is available instead of waiting for more.
//
// When Read encounters an error or end-of-file condition after
// successfully reading n > 0 bytes, it returns the number of
// bytes read.  It may return the (non-nil) error from the same call
// or return the error (and n == 0) from a subsequent call.
// An instance of this general case is that a Reader returning
// a non-zero number of bytes at the end of the input stream may
// return either err == os.EOF or err == nil.  The next Read should
// return 0, os.EOF regardless.
//
// Callers should always process the n > 0 bytes returned before
// considering the error err.  Doing so correctly handles I/O errors
// that happen after reading some bytes and also both of the
// allowed EOF behaviors.
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
// underlying input source.  It returns the number of bytes
// read (0 <= n <= len(p)) and any error encountered.
//
// When ReadAt returns n < len(p), it returns a non-nil error
// explaining why more bytes were not returned.  In this respect,
// ReadAt is stricter than Read.
//
// Even if ReadAt returns n < len(p), it may use all of p as scratch
// space during the call.  If some data is available but not len(p) bytes,
// ReadAt blocks until either all the data is available or an error occurs.
// In this respect ReadAt is different from Read.
//
// If the n = len(p) bytes returned by ReadAt are at the end of the
// input source, ReadAt may return either err == os.EOF or err == nil.
//
// If ReadAt is reading from an input source with a seek offset,
// ReadAt should not affect nor be affected by the underlying
// seek offset.
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

// ByteReader is the interface that wraps the ReadByte method.
//
// ReadByte reads and returns the next byte from the input.
// If no byte is available, err will be set.
type ByteReader interface {
	ReadByte() (c byte, err os.Error)
}

// ByteScanner is the interface that adds the UnreadByte method to the
// basic ReadByte method.
//
// UnreadByte causes the next call to ReadByte to return the same byte
// as the previous call to ReadByte.
// It may be an error to call UnreadByte twice without an intervening
// call to ReadByte.
type ByteScanner interface {
	ByteReader
	UnreadByte() os.Error
}

// RuneReader is the interface that wraps the ReadRune method.
//
// ReadRune reads a single UTF-8 encoded Unicode character
// and returns the rune and its size in bytes. If no character is
// available, err will be set.
type RuneReader interface {
	ReadRune() (r rune, size int, err os.Error)
}

// RuneScanner is the interface that adds the UnreadRune method to the
// basic ReadRune method.
//
// UnreadRune causes the next call to ReadRune to return the same rune
// as the previous call to ReadRune.
// It may be an error to call UnreadRune twice without an intervening
// call to ReadRune.
type RuneScanner interface {
	RuneReader
	UnreadRune() os.Error
}

// stringWriter is the interface that wraps the WriteString method.
type stringWriter interface {
	WriteString(s string) (n int, err os.Error)
}

// WriteString writes the contents of the string s to w, which accepts an array of bytes.
func WriteString(w Writer, s string) (n int, err os.Error) {
	if sw, ok := w.(stringWriter); ok {
		return sw.WriteString(s)
	}
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
	for n < min && err == nil {
		var nn int
		nn, err = r.Read(buf[n:])
		n += nn
	}
	if err == os.EOF {
		if n >= min {
			err = nil
		} else if n > 0 {
			err = ErrUnexpectedEOF
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

// CopyN copies n bytes (or until an error) from src to dst.
// It returns the number of bytes copied and the earliest
// error encountered while copying.  Because Read can
// return the full amount requested as well as an error
// (including os.EOF), so can CopyN.
//
// If dst implements the ReaderFrom interface,
// the copy is implemented by calling dst.ReadFrom(src).
func CopyN(dst Writer, src Reader, n int64) (written int64, err os.Error) {
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
// copied and the first error encountered while copying, if any.
//
// A successful Copy returns err == nil, not err == os.EOF.
// Because Copy is defined to read from src until EOF, it does
// not treat an EOF from Read as an error to be reported.
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
// The underlying implementation is a *LimitedReader.
func LimitReader(r Reader, n int64) Reader { return &LimitedReader{r, n} }

// A LimitedReader reads from R but limits the amount of
// data returned to just N bytes. Each call to Read
// updates N to reflect the new amount remaining.
type LimitedReader struct {
	R Reader // underlying reader
	N int64  // max bytes remaining
}

func (l *LimitedReader) Read(p []byte) (n int, err os.Error) {
	if l.N <= 0 {
		return 0, os.EOF
	}
	if int64(len(p)) > l.N {
		p = p[0:l.N]
	}
	n, err = l.R.Read(p)
	l.N -= int64(n)
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

// TeeReader returns a Reader that writes to w what it reads from r.
// All reads from r performed through it are matched with
// corresponding writes to w.  There is no internal buffering -
// the write must complete before the read completes.
// Any error encountered while writing is reported as a read error.
func TeeReader(r Reader, w Writer) Reader {
	return &teeReader{r, w}
}

type teeReader struct {
	r Reader
	w Writer
}

func (t *teeReader) Read(p []byte) (n int, err os.Error) {
	n, err = t.r.Read(p)
	if n > 0 {
		if n, err := t.w.Write(p[:n]); err != nil {
			return n, err
		}
	}
	return
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements buffered I/O.  It wraps an io.Reader or io.Writer
// object, creating another object (Reader or Writer) that also implements
// the interface but provides buffering and some help for textual I/O.
package bufio

import (
	"bytes"
	"io"
	"os"
	"strconv"
	"utf8"
)


const (
	defaultBufSize = 4096
)

// Errors introduced by this package.
type Error struct {
	os.ErrorString
}

var (
	ErrInvalidUnreadByte os.Error = &Error{"bufio: invalid use of UnreadByte"}
	ErrBufferFull        os.Error = &Error{"bufio: buffer full"}
	errInternal          os.Error = &Error{"bufio: internal error"}
)

// BufSizeError is the error representing an invalid buffer size.
type BufSizeError int

func (b BufSizeError) String() string {
	return "bufio: bad buffer size " + strconv.Itoa(int(b))
}


// Buffered input.

// Reader implements buffering for an io.Reader object.
type Reader struct {
	buf      []byte
	rd       io.Reader
	r, w     int
	err      os.Error
	lastbyte int
}

// NewReaderSize creates a new Reader whose buffer has the specified size,
// which must be greater than zero.  If the argument io.Reader is already a
// Reader with large enough size, it returns the underlying Reader.
// It returns the Reader and any error.
func NewReaderSize(rd io.Reader, size int) (*Reader, os.Error) {
	if size <= 0 {
		return nil, BufSizeError(size)
	}
	// Is it already a Reader?
	b, ok := rd.(*Reader)
	if ok && len(b.buf) >= size {
		return b, nil
	}
	b = new(Reader)
	b.buf = make([]byte, size)
	b.rd = rd
	b.lastbyte = -1
	return b, nil
}

// NewReader returns a new Reader whose buffer has the default size.
func NewReader(rd io.Reader) *Reader {
	b, err := NewReaderSize(rd, defaultBufSize)
	if err != nil {
		// cannot happen - defaultBufSize is a valid size
		panic("bufio: NewReader: ", err.String())
	}
	return b
}

// fill reads a new chunk into the buffer.
func (b *Reader) fill() {
	// Slide existing data to beginning.
	if b.w > b.r {
		copy(b.buf[0:b.w-b.r], b.buf[b.r:b.w])
		b.w -= b.r
	} else {
		b.w = 0
	}
	b.r = 0

	// Read new data.
	n, e := b.rd.Read(b.buf[b.w:])
	b.w += n
	if e != nil {
		b.err = e
	}
}

// Read reads data into p.
// It returns the number of bytes read into p.
// If nn < len(p), also returns an error explaining
// why the read is short.  At EOF, the count will be
// zero and err will be os.EOF.
func (b *Reader) Read(p []byte) (nn int, err os.Error) {
	nn = 0
	for len(p) > 0 {
		n := len(p)
		if b.w == b.r {
			if b.err != nil {
				return nn, b.err
			}
			if len(p) >= len(b.buf) {
				// Large read, empty buffer.
				// Read directly into p to avoid copy.
				n, b.err = b.rd.Read(p)
				if n > 0 {
					b.lastbyte = int(p[n-1])
				}
				p = p[n:]
				nn += n
				continue
			}
			b.fill()
			continue
		}
		if n > b.w-b.r {
			n = b.w - b.r
		}
		copy(p[0:n], b.buf[b.r:b.r+n])
		p = p[n:]
		b.r += n
		b.lastbyte = int(b.buf[b.r-1])
		nn += n
	}
	return nn, nil
}

// ReadByte reads and returns a single byte.
// If no byte is available, returns an error.
func (b *Reader) ReadByte() (c byte, err os.Error) {
	for b.w == b.r {
		if b.err != nil {
			return 0, b.err
		}
		b.fill()
	}
	c = b.buf[b.r]
	b.r++
	b.lastbyte = int(c)
	return c, nil
}

// UnreadByte unreads the last byte.  Only the most recently read byte can be unread.
func (b *Reader) UnreadByte() os.Error {
	if b.r == b.w && b.lastbyte >= 0 {
		b.w = 1
		b.r = 0
		b.buf[0] = byte(b.lastbyte)
		b.lastbyte = -1
		return nil
	}
	if b.r <= 0 {
		return ErrInvalidUnreadByte
	}
	b.r--
	b.lastbyte = -1
	return nil
}

// ReadRune reads a single UTF-8 encoded Unicode character and returns the
// rune and its size in bytes.
func (b *Reader) ReadRune() (rune int, size int, err os.Error) {
	for b.r+utf8.UTFMax > b.w && !utf8.FullRune(b.buf[b.r:b.w]) && b.err == nil {
		b.fill()
	}
	if b.r == b.w {
		return 0, 0, b.err
	}
	rune, size = int(b.buf[b.r]), 1
	if rune >= 0x80 {
		rune, size = utf8.DecodeRune(b.buf[b.r:b.w])
	}
	b.r += size
	b.lastbyte = int(b.buf[b.r-1])
	return rune, size, nil
}

// Buffered returns the number of bytes that can be read from the current buffer.
func (b *Reader) Buffered() int { return b.w - b.r }

// ReadSlice reads until the first occurrence of delim in the input,
// returning a slice pointing at the bytes in the buffer.
// The bytes stop being valid at the next read call.
// If ReadSlice encounters an error before finding a delimiter,
// it returns all the data in the buffer and the error itself (often os.EOF).
// ReadSlice fails with error ErrBufferFull if the buffer fills without a delim.
// Because the data returned from ReadSlice will be overwritten
// by the next I/O operation, most clients should use
// ReadBytes or ReadString instead.
// ReadSlice returns err != nil if and only if line does not end in delim.
func (b *Reader) ReadSlice(delim byte) (line []byte, err os.Error) {
	// Look in buffer.
	if i := bytes.IndexByte(b.buf[b.r:b.w], delim); i >= 0 {
		line1 := b.buf[b.r : b.r+i+1]
		b.r += i + 1
		return line1, nil
	}

	// Read more into buffer, until buffer fills or we find delim.
	for {
		if b.err != nil {
			line := b.buf[b.r:b.w]
			b.r = b.w
			return line, b.err
		}

		n := b.Buffered()
		b.fill()

		// Search new part of buffer
		if i := bytes.IndexByte(b.buf[n:b.w], delim); i >= 0 {
			line := b.buf[0 : n+i+1]
			b.r = n + i + 1
			return line, nil
		}

		// Buffer is full?
		if b.Buffered() >= len(b.buf) {
			return nil, ErrBufferFull
		}
	}
	panic("not reached")
}

// ReadBytes reads until the first occurrence of delim in the input,
// returning a string containing the data up to and including the delimiter.
// If ReadBytes encounters an error before finding a delimiter,
// it returns the data read before the error and the error itself (often os.EOF).
// ReadBytes returns err != nil if and only if line does not end in delim.
func (b *Reader) ReadBytes(delim byte) (line []byte, err os.Error) {
	// Use ReadSlice to look for array,
	// accumulating full buffers.
	var frag []byte
	var full [][]byte
	nfull := 0
	err = nil

	for {
		var e os.Error
		frag, e = b.ReadSlice(delim)
		if e == nil { // got final fragment
			break
		}
		if e != ErrBufferFull { // unexpected error
			err = e
			break
		}

		// Read bytes out of buffer.
		buf := make([]byte, b.Buffered())
		var n int
		n, e = b.Read(buf)
		if e != nil {
			frag = buf[0:n]
			err = e
			break
		}
		if n != len(buf) {
			frag = buf[0:n]
			err = errInternal
			break
		}

		// Grow list if needed.
		if full == nil {
			full = make([][]byte, 16)
		} else if nfull >= len(full) {
			newfull := make([][]byte, len(full)*2)
			for i := 0; i < len(full); i++ {
				newfull[i] = full[i]
			}
			full = newfull
		}

		// Save buffer
		full[nfull] = buf
		nfull++
	}

	// Allocate new buffer to hold the full pieces and the fragment.
	n := 0
	for i := 0; i < nfull; i++ {
		n += len(full[i])
	}
	n += len(frag)

	// Copy full pieces and fragment in.
	buf := make([]byte, n)
	n = 0
	for i := 0; i < nfull; i++ {
		copy(buf[n:n+len(full[i])], full[i])
		n += len(full[i])
	}
	copy(buf[n:n+len(frag)], frag)
	return buf, err
}

// ReadString reads until the first occurrence of delim in the input,
// returning a string containing the data up to and including the delimiter.
// If ReadString encounters an error before finding a delimiter,
// it returns the data read before the error and the error itself (often os.EOF).
// ReadString returns err != nil if and only if line does not end in delim.
func (b *Reader) ReadString(delim byte) (line string, err os.Error) {
	bytes, e := b.ReadBytes(delim)
	return string(bytes), e
}


// buffered output

// Writer implements buffering for an io.Writer object.
type Writer struct {
	err os.Error
	buf []byte
	n   int
	wr  io.Writer
}

// NewWriterSize creates a new Writer whose buffer has the specified size,
// which must be greater than zero. If the argument io.Writer is already a
// Writer with large enough size, it returns the underlying Writer.
// It returns the Writer and any error.
func NewWriterSize(wr io.Writer, size int) (*Writer, os.Error) {
	if size <= 0 {
		return nil, BufSizeError(size)
	}
	// Is it already a Writer?
	b, ok := wr.(*Writer)
	if ok && len(b.buf) >= size {
		return b, nil
	}
	b = new(Writer)
	b.buf = make([]byte, size)
	b.wr = wr
	return b, nil
}

// NewWriter returns a new Writer whose buffer has the default size.
func NewWriter(wr io.Writer) *Writer {
	b, err := NewWriterSize(wr, defaultBufSize)
	if err != nil {
		// cannot happen - defaultBufSize is valid size
		panic("bufio: NewWriter: ", err.String())
	}
	return b
}

// Flush writes any buffered data to the underlying io.Writer.
func (b *Writer) Flush() os.Error {
	if b.err != nil {
		return b.err
	}
	n, e := b.wr.Write(b.buf[0:b.n])
	if n < b.n && e == nil {
		e = io.ErrShortWrite
	}
	if e != nil {
		if n > 0 && n < b.n {
			copy(b.buf[0:b.n-n], b.buf[n:b.n])
		}
		b.n -= n
		b.err = e
		return e
	}
	b.n = 0
	return nil
}

// Available returns how many bytes are unused in the buffer.
func (b *Writer) Available() int { return len(b.buf) - b.n }

// Buffered returns the number of bytes that have been written into the current buffer.
func (b *Writer) Buffered() int { return b.n }

// Write writes the contents of p into the buffer.
// It returns the number of bytes written.
// If nn < len(p), also returns an error explaining
// why the write is short.
func (b *Writer) Write(p []byte) (nn int, err os.Error) {
	if b.err != nil {
		return 0, b.err
	}
	nn = 0
	for len(p) > 0 {
		n := b.Available()
		if n <= 0 {
			if b.Flush(); b.err != nil {
				break
			}
			n = b.Available()
		}
		if b.Available() == 0 && len(p) >= len(b.buf) {
			// Large write, empty buffer.
			// Write directly from p to avoid copy.
			n, b.err = b.wr.Write(p)
			nn += n
			p = p[n:]
			if b.err != nil {
				break
			}
			continue
		}
		if n > len(p) {
			n = len(p)
		}
		copy(b.buf[b.n:b.n+n], p[0:n])
		b.n += n
		nn += n
		p = p[n:]
	}
	return nn, b.err
}

// WriteByte writes a single byte.
func (b *Writer) WriteByte(c byte) os.Error {
	if b.err != nil {
		return b.err
	}
	if b.Available() <= 0 && b.Flush() != nil {
		return b.err
	}
	b.buf[b.n] = c
	b.n++
	return nil
}

// WriteString writes a string.
func (b *Writer) WriteString(s string) os.Error {
	if b.err != nil {
		return b.err
	}
	// Common case, worth making fast.
	if b.Available() >= len(s) || len(b.buf) >= len(s) && b.Flush() == nil {
		for i := 0; i < len(s); i++ { // loop over bytes, not runes.
			b.buf[b.n] = s[i]
			b.n++
		}
		return nil
	}
	for i := 0; i < len(s); i++ { // loop over bytes, not runes.
		b.WriteByte(s[i])
	}
	return b.err
}

// buffered input and output

// ReadWriter stores pointers to a Reader and a Writer.
// It implements io.ReadWriter.
type ReadWriter struct {
	*Reader
	*Writer
}

// NewReadWriter allocates a new ReadWriter that dispatches to r and w.
func NewReadWriter(r *Reader, w *Writer) *ReadWriter {
	return &ReadWriter{r, w}
}

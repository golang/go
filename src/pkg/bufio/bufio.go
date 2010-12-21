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
	ErrInvalidUnreadRune os.Error = &Error{"bufio: invalid use of UnreadRune"}
	ErrBufferFull        os.Error = &Error{"bufio: buffer full"}
	ErrNegativeCount     os.Error = &Error{"bufio: negative count"}
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
	buf          []byte
	rd           io.Reader
	r, w         int
	err          os.Error
	lastByte     int
	lastRuneSize int
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
	b.lastByte = -1
	b.lastRuneSize = -1
	return b, nil
}

// NewReader returns a new Reader whose buffer has the default size.
func NewReader(rd io.Reader) *Reader {
	b, err := NewReaderSize(rd, defaultBufSize)
	if err != nil {
		// cannot happen - defaultBufSize is a valid size
		panic(err)
	}
	return b
}

// fill reads a new chunk into the buffer.
func (b *Reader) fill() {
	// Slide existing data to beginning.
	if b.r > 0 {
		copy(b.buf, b.buf[b.r:b.w])
		b.w -= b.r
		b.r = 0
	}

	// Read new data.
	n, e := b.rd.Read(b.buf[b.w:])
	b.w += n
	if e != nil {
		b.err = e
	}
}

// Peek returns the next n bytes without advancing the reader. The bytes stop
// being valid at the next read call. If Peek returns fewer than n bytes, it
// also returns an error explaining why the read is short. The error is
// ErrBufferFull if n is larger than b's buffer size.
func (b *Reader) Peek(n int) ([]byte, os.Error) {
	if n < 0 {
		return nil, ErrNegativeCount
	}
	if n > len(b.buf) {
		return nil, ErrBufferFull
	}
	for b.w-b.r < n && b.err == nil {
		b.fill()
	}
	m := b.w - b.r
	if m > n {
		m = n
	}
	err := b.err
	if m < n && err == nil {
		err = ErrBufferFull
	}
	return b.buf[b.r : b.r+m], err
}

// Read reads data into p.
// It returns the number of bytes read into p.
// It calls Read at most once on the underlying Reader,
// hence n may be less than len(p).
// At EOF, the count will be zero and err will be os.EOF.
func (b *Reader) Read(p []byte) (n int, err os.Error) {
	n = len(p)
	if n == 0 {
		return 0, b.err
	}
	if b.w == b.r {
		if b.err != nil {
			return 0, b.err
		}
		if len(p) >= len(b.buf) {
			// Large read, empty buffer.
			// Read directly into p to avoid copy.
			n, b.err = b.rd.Read(p)
			if n > 0 {
				b.lastByte = int(p[n-1])
				b.lastRuneSize = -1
			}
			return n, b.err
		}
		b.fill()
		if b.w == b.r {
			return 0, b.err
		}
	}

	if n > b.w-b.r {
		n = b.w - b.r
	}
	copy(p[0:n], b.buf[b.r:])
	b.r += n
	b.lastByte = int(b.buf[b.r-1])
	b.lastRuneSize = -1
	return n, nil
}

// ReadByte reads and returns a single byte.
// If no byte is available, returns an error.
func (b *Reader) ReadByte() (c byte, err os.Error) {
	b.lastRuneSize = -1
	for b.w == b.r {
		if b.err != nil {
			return 0, b.err
		}
		b.fill()
	}
	c = b.buf[b.r]
	b.r++
	b.lastByte = int(c)
	return c, nil
}

// UnreadByte unreads the last byte.  Only the most recently read byte can be unread.
func (b *Reader) UnreadByte() os.Error {
	b.lastRuneSize = -1
	if b.r == b.w && b.lastByte >= 0 {
		b.w = 1
		b.r = 0
		b.buf[0] = byte(b.lastByte)
		b.lastByte = -1
		return nil
	}
	if b.r <= 0 {
		return ErrInvalidUnreadByte
	}
	b.r--
	b.lastByte = -1
	return nil
}

// ReadRune reads a single UTF-8 encoded Unicode character and returns the
// rune and its size in bytes.
func (b *Reader) ReadRune() (rune int, size int, err os.Error) {
	for b.r+utf8.UTFMax > b.w && !utf8.FullRune(b.buf[b.r:b.w]) && b.err == nil {
		b.fill()
	}
	b.lastRuneSize = -1
	if b.r == b.w {
		return 0, 0, b.err
	}
	rune, size = int(b.buf[b.r]), 1
	if rune >= 0x80 {
		rune, size = utf8.DecodeRune(b.buf[b.r:b.w])
	}
	b.r += size
	b.lastByte = int(b.buf[b.r-1])
	b.lastRuneSize = size
	return rune, size, nil
}

// UnreadRune unreads the last rune.  If the most recent read operation on
// the buffer was not a ReadRune, UnreadRune returns an error.  (In this
// regard it is stricter than UnreadByte, which will unread the last byte
// from any read operation.)
func (b *Reader) UnreadRune() os.Error {
	if b.lastRuneSize < 0 || b.r == 0 {
		return ErrInvalidUnreadRune
	}
	b.r -= b.lastRuneSize
	b.lastByte = -1
	b.lastRuneSize = -1
	return nil
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
			b.r = b.w
			return b.buf, ErrBufferFull
		}
	}
	panic("not reached")
}

// ReadBytes reads until the first occurrence of delim in the input,
// returning a slice containing the data up to and including the delimiter.
// If ReadBytes encounters an error before finding a delimiter,
// it returns the data read before the error and the error itself (often os.EOF).
// ReadBytes returns err != nil if and only if line does not end in delim.
func (b *Reader) ReadBytes(delim byte) (line []byte, err os.Error) {
	// Use ReadSlice to look for array,
	// accumulating full buffers.
	var frag []byte
	var full [][]byte
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

		// Make a copy of the buffer.
		buf := make([]byte, len(frag))
		copy(buf, frag)
		full = append(full, buf)
	}

	// Allocate new buffer to hold the full pieces and the fragment.
	n := 0
	for i := range full {
		n += len(full[i])
	}
	n += len(frag)

	// Copy full pieces and fragment in.
	buf := make([]byte, n)
	n = 0
	for i := range full {
		n += copy(buf[n:], full[i])
	}
	copy(buf[n:], frag)
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
		panic(err)
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
// If nn < len(p), it also returns an error explaining
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
		if b.Buffered() == 0 && len(p) >= len(b.buf) {
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

// WriteRune writes a single Unicode code point, returning
// the number of bytes written and any error.
func (b *Writer) WriteRune(rune int) (size int, err os.Error) {
	if rune < utf8.RuneSelf {
		err = b.WriteByte(byte(rune))
		if err != nil {
			return 0, err
		}
		return 1, nil
	}
	if b.err != nil {
		return 0, b.err
	}
	n := b.Available()
	if n < utf8.UTFMax {
		if b.Flush(); b.err != nil {
			return 0, b.err
		}
		n = b.Available()
		if n < utf8.UTFMax {
			// Can only happen if buffer is silly small.
			return b.WriteString(string(rune))
		}
	}
	size = utf8.EncodeRune(b.buf[b.n:], rune)
	b.n += size
	return size, nil
}

// WriteString writes a string.
// It returns the number of bytes written.
// If the count is less than len(s), it also returns an error explaining
// why the write is short.
func (b *Writer) WriteString(s string) (int, os.Error) {
	if b.err != nil {
		return 0, b.err
	}
	// Common case, worth making fast.
	if b.Available() >= len(s) || len(b.buf) >= len(s) && b.Flush() == nil {
		for i := 0; i < len(s); i++ { // loop over bytes, not runes.
			b.buf[b.n] = s[i]
			b.n++
		}
		return len(s), nil
	}
	for i := 0; i < len(s); i++ { // loop over bytes, not runes.
		b.WriteByte(s[i])
		if b.err != nil {
			return i, b.err
		}
	}
	return len(s), nil
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

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements buffered I/O.  It wraps an io.Reader or io.Writer
// object, creating another object (Reader or Writer) that also implements
// the interface but provides buffering and some help for textual I/O.
package bufio

import (
	"io";
	"os";
	"utf8";
)


// TODO:
//	- maybe define an interface
//	- Reader: ReadRune, UnreadRune ?
//		could make ReadRune generic if we dropped UnreadRune
//	- buffered output

const (
	defaultBufSize = 4096
)

// Errors introduced by this package.
type Error struct {
	os.ErrorString;
}

var (
	PhaseError os.Error = &Error{"bufio: phase error"};
	BufferFull os.Error = &Error{"bufio: buffer full"};
	InternalError os.Error = &Error{"bufio: internal error"};
	BadBufSize os.Error = &Error{"bufio: bad buffer size"};
)

func copySlice(dst []byte, src []byte) {
	for i := 0; i < len(dst); i++ {
		dst[i] = src[i]
	}
}


// Buffered input.

// Reader implements buffering for an io.Reader object.
type Reader struct {
	buf []byte;
	rd io.Reader;
	r, w int;
	err os.Error;
	lastbyte int;
}

// NewReaderSize creates a new Reader whose buffer has the specified size,
// which must be greater than zero.  If the argument io.Reader is already a
// Reader with large enough size, it returns the underlying Reader.
// It returns the Reader and any error.
func NewReaderSize(rd io.Reader, size int) (*Reader, os.Error) {
	if size <= 0 {
		return nil, BadBufSize
	}
	// Is it already a Reader?
	b, ok := rd.(*Reader);
	if ok && len(b.buf) >= size {
		return b, nil
	}
	b = new(Reader);
	b.buf = make([]byte, size);
	b.rd = rd;
	b.lastbyte = -1;
	return b, nil
}

// NewReader returns a new Reader whose buffer has the default size.
func NewReader(rd io.Reader) *Reader {
	b, err := NewReaderSize(rd, defaultBufSize);
	if err != nil {
		// cannot happen - defaultBufSize is a valid size
		panic("bufio: NewReader: ", err.String());
	}
	return b;
}

//.fill reads a new chunk into the buffer.
func (b *Reader) fill() os.Error {
	if b.err != nil {
		return b.err
	}

	// Slide existing data to beginning.
	if b.w > b.r {
		copySlice(b.buf[0:b.w-b.r], b.buf[b.r:b.w]);
		b.w -= b.r;
	} else {
		b.w = 0
	}
	b.r = 0;

	// Read new data.
	n, e := b.rd.Read(b.buf[b.w:len(b.buf)]);
	if e != nil {
		b.err = e;
		return e
	}
	b.w += n;
	return nil
}

// Read reads data into p.
// It returns the number of bytes read into p.
// If nn < len(p), also returns an error explaining
// why the read is short.  At EOF, the count will be
// zero and err will be io.ErrEOF.
func (b *Reader) Read(p []byte) (nn int, err os.Error) {
	nn = 0;
	for len(p) > 0 {
		n := len(p);
		if b.w == b.r {
			if len(p) >= len(b.buf) {
				// Large read, empty buffer.
				// Read directly into p to avoid copy.
				n, b.err = b.rd.Read(p);
				if n > 0 {
					b.lastbyte = int(p[n-1]);
				}
				p = p[n:len(p)];
				nn += n;
				if b.err != nil {
					return nn, b.err
				}
				if n == 0 {
					return nn, io.ErrEOF
				}
				continue;
			}
			b.fill();
			if b.err != nil {
				return nn, b.err
			}
			if b.w == b.r {
				return nn, io.ErrEOF
			}
		}
		if n > b.w - b.r {
			n = b.w - b.r
		}
		copySlice(p[0:n], b.buf[b.r:b.r+n]);
		p = p[n:len(p)];
		b.r += n;
		b.lastbyte = int(b.buf[b.r-1]);
		nn += n
	}
	return nn, nil
}

// ReadByte reads and returns a single byte.
// If no byte is available, returns an error.
func (b *Reader) ReadByte() (c byte, err os.Error) {
	if b.w == b.r {
		b.fill();
		if b.err != nil {
			return 0, b.err
		}
		if b.w == b.r {
			return 0, io.ErrEOF
		}
	}
	c = b.buf[b.r];
	b.r++;
	b.lastbyte = int(c);
	return c, nil
}

// UnreadByte unreads the last byte.  Only one byte may be unread at a given time.
func (b *Reader) UnreadByte() os.Error {
	if b.err != nil {
		return b.err
	}
	if b.r == b.w && b.lastbyte >= 0 {
		b.w = 1;
		b.r = 0;
		b.buf[0] = byte(b.lastbyte);
		b.lastbyte = -1;
		return nil;
	}
	if b.r <= 0 {
		return PhaseError
	}
	b.r--;
	b.lastbyte = -1;
	return nil
}

// ReadRune reads a single UTF-8 encoded Unicode character and returns the
// rune and its size in bytes.
func (b *Reader) ReadRune() (rune int, size int, err os.Error) {
	for b.r + utf8.UTFMax > b.w && !utf8.FullRune(b.buf[b.r:b.w]) {
		n := b.w - b.r;
		b.fill();
		if b.err != nil {
			return 0, 0, b.err
		}
		if b.w - b.r == n {
			// no bytes read
			if b.r == b.w {
				return 0, 0, io.ErrEOF
			}
			break;
		}
	}
	rune, size = int(b.buf[b.r]), 1;
	if rune >= 0x80 {
		rune, size = utf8.DecodeRune(b.buf[b.r:b.w]);
	}
	b.r += size;
	b.lastbyte = int(b.buf[b.r-1]);
	return rune, size, nil
}

// Helper function: look for byte c in array p,
// returning its index or -1.
func findByte(p []byte, c byte) int {
	for i := 0; i < len(p); i++ {
		if p[i] == c {
			return i
		}
	}
	return -1
}

// Buffered returns the number of bytes that can be read from the current buffer.
func (b *Reader) Buffered() int {
	return b.w - b.r;
}

// ReadLineSlice reads until the first occurrence of delim in the input,
// returning a slice pointing at the bytes in the buffer.
// The bytes stop being valid at the next read call.
// Fails if the line doesn't fit in the buffer.
// For internal or advanced use only; most uses should
// call ReadLineString or ReadLineBytes instead.
func (b *Reader) ReadLineSlice(delim byte) (line []byte, err os.Error) {
	if b.err != nil {
		return nil, b.err
	}

	// Look in buffer.
	if i := findByte(b.buf[b.r:b.w], delim); i >= 0 {
		line1 := b.buf[b.r:b.r+i+1];
		b.r += i+1;
		return line1, nil
	}

	// Read more into buffer, until buffer fills or we find delim.
	for {
		n := b.Buffered();
		b.fill();
		if b.err != nil {
			return nil, b.err
		}
		if b.Buffered() == n {	// no data added; end of file
			line := b.buf[b.r:b.w];
			b.r = b.w;
			return line, io.ErrEOF
		}

		// Search new part of buffer
		if i := findByte(b.buf[n:b.w], delim); i >= 0 {
			line := b.buf[0:n+i+1];
			b.r = n+i+1;
			return line, nil
		}

		// Buffer is full?
		if b.Buffered() >= len(b.buf) {
			return nil, BufferFull
		}
	}

	// BUG 6g bug100
	return nil, nil
}

// ReadLineBytes reads until the first occurrence of delim in the input,
// returning a new byte array containing the line.
// If an error happens, returns the data (without a delimiter)
// and the error.  (It can't leave the data in the buffer because
// it might have read more than the buffer size.)
func (b *Reader) ReadLineBytes(delim byte) (line []byte, err os.Error) {
	if b.err != nil {
		return nil, b.err
	}

	// Use ReadLineSlice to look for array,
	// accumulating full buffers.
	var frag []byte;
	var full [][]byte;
	nfull := 0;
	err = nil;

	for {
		var e os.Error;
		frag, e = b.ReadLineSlice(delim);
		if e == nil {	// got final fragment
			break
		}
		if e != BufferFull {	// unexpected error
			err = e;
			break
		}

		// Read bytes out of buffer.
		buf := make([]byte, b.Buffered());
		var n int;
		n, e = b.Read(buf);
		if e != nil {
			frag = buf[0:n];
			err = e;
			break
		}
		if n != len(buf) {
			frag = buf[0:n];
			err = InternalError;
			break
		}

		// Grow list if needed.
		if full == nil {
			full = make([][]byte, 16);
		} else if nfull >= len(full) {
			newfull := make([][]byte, len(full)*2);
			// BUG slice assignment
			for i := 0; i < len(full); i++ {
				newfull[i] = full[i];
			}
			full = newfull
		}

		// Save buffer
		full[nfull] = buf;
		nfull++;
	}

	// Allocate new buffer to hold the full pieces and the fragment.
	n := 0;
	for i := 0; i < nfull; i++ {
		n += len(full[i])
	}
	n += len(frag);

	// Copy full pieces and fragment in.
	buf := make([]byte, n);
	n = 0;
	for i := 0; i < nfull; i++ {
		copySlice(buf[n:n+len(full[i])], full[i]);
		n += len(full[i])
	}
	copySlice(buf[n:n+len(frag)], frag);
	return buf, err
}

// ReadLineString reads until the first occurrence of delim in the input,
// returning a new string containing the line.
// If savedelim, keep delim in the result; otherwise drop it.
func (b *Reader) ReadLineString(delim byte, savedelim bool) (line string, err os.Error) {
	bytes, e := b.ReadLineBytes(delim);
	if e != nil {
		return string(bytes), e
	}
	if !savedelim {
		bytes = bytes[0:len(bytes)-1]
	}
	return string(bytes), nil
}


// buffered output

// Writer implements buffering for an io.Writer object.
type Writer struct {
	err os.Error;
	buf []byte;
	n int;
	wr io.Writer;
}

// NewWriterSize creates a new Writer whose buffer has the specified size,
// which must be greater than zero. If the argument io.Writer is already a
// Writer with large enough size, it returns the underlying Writer.
// It returns the Writer and any error.
func NewWriterSize(wr io.Writer, size int) (*Writer, os.Error) {
	if size <= 0 {
		return nil, BadBufSize
	}
	// Is it already a Writer?
	b, ok := wr.(*Writer);
	if ok && len(b.buf) >= size {
		return b, nil
	}
	b = new(Writer);
	b.buf = make([]byte, size);
	b.wr = wr;
	return b, nil
}

// NewWriter returns a new Writer whose buffer has the default size.
func NewWriter(wr io.Writer) *Writer {
	b, err := NewWriterSize(wr, defaultBufSize);
	if err != nil {
		// cannot happen - defaultBufSize is valid size
		panic("bufio: NewWriter: ", err.String());
	}
	return b;
}

// Flush writes any buffered data to the underlying io.Writer.
func (b *Writer) Flush() os.Error {
	if b.err != nil {
		return b.err
	}
	n := 0;
	for n < b.n {
		m, e := b.wr.Write(b.buf[n:b.n]);
		n += m;
		if m == 0 && e == nil {
			e = io.ErrShortWrite
		}
		if e != nil {
			if n < b.n {
				copySlice(b.buf[0:b.n-n], b.buf[n:b.n])
			}
			b.n -= n;
			b.err = e;
			return e
		}
	}
	b.n = 0;
	return nil
}

// Available returns how many bytes are unused in the buffer.
func (b *Writer) Available() int {
	return len(b.buf) - b.n
}

// Buffered returns the number of bytes that have been written into the current buffer.
func (b *Writer) Buffered() int {
	return b.n
}

// Write writes the contents of p into the buffer.
// It returns the number of bytes written.
// If nn < len(p), also returns an error explaining
// why the write is short.
func (b *Writer) Write(p []byte) (nn int, err os.Error) {
	if b.err != nil {
		return 0, b.err
	}
	nn = 0;
	for len(p) > 0 {
		n := b.Available();
		if n <= 0 {
			if b.Flush(); b.err != nil {
				break
			}
			n = b.Available()
		}
		if b.Available() == 0 && len(p) >= len(b.buf) {
			// Large write, empty buffer.
			// Write directly from p to avoid copy.
			n, b.err = b.wr.Write(p);
			nn += n;
			p = p[n:len(p)];
			if b.err != nil {
				break;
			}
			continue;
		}
		if n > len(p) {
			n = len(p)
		}
		copySlice(b.buf[b.n:b.n+n], p[0:n]);
		b.n += n;
		nn += n;
		p = p[n:len(p)]
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
	b.buf[b.n] = c;
	b.n++;
	return nil
}

// buffered input and output

// ReadWriter stores (a pointer to) a Reader and a Writer.
// It implements io.ReadWriter.
type ReadWriter struct {
	*Reader;
	*Writer;
}

// NewReadWriter allocates a new ReadWriter holding r and w.
func NewReadWriter(r *Reader, w *Writer) *ReadWriter {
	return &ReadWriter{r, w}
}


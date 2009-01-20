// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bufio

import (
	"os";
	"io";
	"utf8";
)


// TODO:
//	- maybe define an interface
//	- BufRead: ReadRune, UnreadRune ?
//		could make ReadRune generic if we dropped UnreadRune
//	- buffered output

const (
	defaultBufSize = 4096
)

var (
	EndOfFile = os.NewError("end of file");
	PhaseError = os.NewError("phase error");
	BufferFull = os.NewError("buffer full");
	InternalError = os.NewError("bufio internal error");
	BadBufSize = os.NewError("bad bufio size");
	ShortWrite = os.NewError("short write");
)

func copySlice(dst []byte, src []byte) {
	for i := 0; i < len(dst); i++ {
		dst[i] = src[i]
	}
}


// Buffered input.

type BufRead struct {
	buf []byte;
	rd io.Read;
	r, w int;
	err *os.Error;
}

func NewBufReadSize(rd io.Read, size int) (b *BufRead, err *os.Error) {
	if size <= 0 {
		return nil, BadBufSize
	}
	b = new(BufRead);
	b.buf = make([]byte, size);
	b.rd = rd;
	return b, nil
}

func NewBufRead(rd io.Read) (b *BufRead, err *os.Error) {
	return NewBufReadSize(rd, defaultBufSize);
}

// Read a new chunk into the buffer.
func (b *BufRead) Fill() *os.Error {
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

// Read into p.
// Returns the number of bytes read into p.
// If nn < len(p), also returns an error explaining
// why the read is short.
func (b *BufRead) Read(p []byte) (nn int, err *os.Error) {
	nn = 0;
	for len(p) > 0 {
		n := len(p);
		if b.w == b.r {
			b.Fill();
			if b.err != nil {
				return nn, b.err
			}
			if b.w == b.r {
				return nn, EndOfFile
			}
		}
		if n > b.w - b.r {
			n = b.w - b.r
		}
		copySlice(p[0:n], b.buf[b.r:b.r+n]);
		p = p[n:len(p)];
		b.r += n;
		nn += n
	}
	return nn, nil
}

// Read a single byte.
// If no byte available, returns error.
func (b *BufRead) ReadByte() (c byte, err *os.Error) {
	if b.w == b.r {
		b.Fill();
		if b.err != nil {
			return 0, b.err
		}
		if b.w == b.r {
			return 0, EndOfFile
		}
	}
	c = b.buf[b.r];
	b.r++;
	return c, nil
}

// Unread the last byte.  Only guaranteed to be able to unread one byte.
func (b *BufRead) UnreadByte() *os.Error {
	if b.err != nil {
		return b.err
	}
	if b.r <= 0 {
		return PhaseError
	}
	b.r--;
	return nil
}

// Read a single Unicode character; returns the rune and its size.
func (b *BufRead) ReadRune() (rune int, size int, err *os.Error) {
	for b.r + utf8.UTFMax > b.w && !utf8.FullRune(b.buf[b.r:b.w]) {
		n := b.w - b.r;
		b.Fill();
		if b.err != nil {
			return 0, 0, b.err
		}
		if b.w - b.r == n {
			// no bytes read
			if b.r == b.w {
				return 0, 0, EndOfFile
			}
			break;
		}
	}
	rune, size = int(b.buf[b.r]), 1;
	if rune >= 0x80 {
		rune, size = utf8.DecodeRune(b.buf[b.r:b.w]);
	}
	b.r += size;
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

// Returns the number of bytes that can be read.
func (b *BufRead) Buffered() int {
	return b.w - b.r;
}

// Read until the first occurrence of delim in the input,
// returning a slice pointing at the bytes in the buffer.
// The bytes stop being valid at the next read call.
// Fails if the line doesn't fit in the buffer.
// For internal (or advanced) use only.
// Use ReadLineString or ReadLineBytes instead.

func (b *BufRead) ReadLineSlice(delim byte) (line []byte, err *os.Error) {
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
		b.Fill();
		if b.err != nil {
			return nil, b.err
		}
		if b.Buffered() == n {	// no data added; end of file
			line := b.buf[b.r:b.w];
			b.r = b.w;
			return line, EndOfFile
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

// Read until the first occurrence of delim in the input,
// returning a new byte array containing the line.
// If an error happens, returns the data (without a delimiter)
// and the error.  (Can't leave the data in the buffer because
// we might have read more than the buffer size.)
func (b *BufRead) ReadLineBytes(delim byte) (line []byte, err *os.Error) {
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
		var e *os.Error;
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

// Read until the first occurrence of delim in the input,
// returning a new string containing the line.
// If savedelim, keep delim in the result; otherwise chop it off.
func (b *BufRead) ReadLineString(delim byte, savedelim bool) (line string, err *os.Error) {
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

type BufWrite struct {
	err *os.Error;
	buf []byte;
	n int;
	wr io.Write;
}

func NewBufWriteSize(wr io.Write, size int) (b *BufWrite, err *os.Error) {
	if size <= 0 {
		return nil, BadBufSize
	}
	b = new(BufWrite);
	b.buf = make([]byte, size);
	b.wr = wr;
	return b, nil
}

func NewBufWrite(wr io.Write) (b *BufWrite, err *os.Error) {
	return NewBufWriteSize(wr, defaultBufSize);
}

// Flush the output buffer.
func (b *BufWrite) Flush() *os.Error {
	if b.err != nil {
		return b.err
	}
	n := 0;
	for n < b.n {
		m, e := b.wr.Write(b.buf[n:b.n]);
		n += m;
		if m == 0 && e == nil {
			e = ShortWrite
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

func (b *BufWrite) Available() int {
	return len(b.buf) - b.n
}

func (b *BufWrite) Buffered() int {
	return b.n
}

func (b *BufWrite) Write(p []byte) (nn int, err *os.Error) {
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

func (b *BufWrite) WriteByte(c byte) *os.Error {
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


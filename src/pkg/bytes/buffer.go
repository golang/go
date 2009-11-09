// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes

// Simple byte buffer for marshaling data.

import (
	"os";
)

// Copy from string to byte array at offset doff.  Assume there's room.
func copyString(dst []byte, doff int, str string) {
	for soff := 0; soff < len(str); soff++ {
		dst[doff] = str[soff];
		doff++;
	}
}

// Copy from bytes to byte array at offset doff.  Assume there's room.
func copyBytes(dst []byte, doff int, src []byte) {
	for soff := 0; soff < len(src); soff++ {
		dst[doff] = src[soff];
		doff++;
	}
}

// A Buffer is a variable-sized buffer of bytes
// with Read and Write methods.
// The zero value for Buffer is an empty buffer ready to use.
type Buffer struct {
	buf	[]byte;	// contents are the bytes buf[off : len(buf)]
	off	int;	// read at &buf[off], write at &buf[len(buf)]
	oneByte	[]byte;	// avoid allocation of slice on each WriteByte
}

// Bytes returns the contents of the unread portion of the buffer;
// len(b.Bytes()) == b.Len().
func (b *Buffer) Bytes() []byte	{ return b.buf[b.off : len(b.buf)] }

// String returns the contents of the unread portion of the buffer
// as a string.  If the Buffer is a nil pointer, it returns "<nil>".
func (b *Buffer) String() string {
	if b == nil {
		// Special case, useful in debugging.
		return "<nil>"
	}
	return string(b.buf[b.off : len(b.buf)]);
}

// Len returns the number of bytes of the unread portion of the buffer;
// b.Len() == len(b.Bytes()).
func (b *Buffer) Len() int	{ return len(b.buf) - b.off }

// Truncate discards all but the first n unread bytes from the buffer.
// It is an error to call b.Truncate(n) with n > b.Len().
func (b *Buffer) Truncate(n int) {
	if n == 0 {
		// Reuse buffer space.
		b.off = 0
	}
	b.buf = b.buf[0 : b.off + n];
}

// Reset resets the buffer so it has no content.
// b.Reset() is the same as b.Truncate(0).
func (b *Buffer) Reset()	{ b.Truncate(0) }

// Write appends the contents of p to the buffer.  The return
// value n is the length of p; err is always nil.
func (b *Buffer) Write(p []byte) (n int, err os.Error) {
	m := b.Len();
	n = len(p);

	if len(b.buf)+n > cap(b.buf) {
		// not enough space at end
		buf := b.buf;
		if m+n > cap(b.buf) {
			// not enough space anywhere
			buf = make([]byte, 2*cap(b.buf) + n)
		}
		copyBytes(buf, 0, b.buf[b.off : b.off + m]);
		b.buf = buf;
		b.off = 0;
	}

	b.buf = b.buf[0 : b.off + m + n];
	copyBytes(b.buf, b.off + m, p);
	return n, nil;
}

// WriteString appends the contents of s to the buffer.  The return
// value n is the length of s; err is always nil.
func (b *Buffer) WriteString(s string) (n int, err os.Error) {
	m := b.Len();
	n = len(s);

	if len(b.buf)+n > cap(b.buf) {
		// not enough space at end
		buf := b.buf;
		if m+n > cap(b.buf) {
			// not enough space anywhere
			buf = make([]byte, 2*cap(b.buf) + n)
		}
		copyBytes(buf, 0, b.buf[b.off : b.off + m]);
		b.buf = buf;
		b.off = 0;
	}

	b.buf = b.buf[0 : b.off + m + n];
	copyString(b.buf, b.off + m, s);
	return n, nil;
}

// WriteByte appends the byte c to the buffer.
// The returned error is always nil, but is included
// to match bufio.Writer's WriteByte.
func (b *Buffer) WriteByte(c byte) os.Error {
	if b.oneByte == nil {
		// Only happens once per Buffer, and then we have a slice.
		b.oneByte = make([]byte, 1)
	}
	b.oneByte[0] = c;
	b.Write(b.oneByte);
	return nil;
}

// Read reads the next len(p) bytes from the buffer or until the buffer
// is drained.  The return value n is the number of bytes read.  If the
// buffer has no data to return, err is os.EOF even if len(p) is zero;
// otherwise it is nil.
func (b *Buffer) Read(p []byte) (n int, err os.Error) {
	if b.off >= len(b.buf) {
		return 0, os.EOF
	}
	m := b.Len();
	n = len(p);

	if n > m {
		// more bytes requested than available
		n = m
	}

	copyBytes(p, 0, b.buf[b.off : b.off + n]);
	b.off += n;
	return n, err;
}

// ReadByte reads and returns the next byte from the buffer.
// If no byte is available, it returns error os.EOF.
func (b *Buffer) ReadByte() (c byte, err os.Error) {
	if b.off >= len(b.buf) {
		return 0, os.EOF
	}
	c = b.buf[b.off];
	b.off++;
	return c, nil;
}

// NewBuffer creates and initializes a new Buffer
// using buf as its initial contents.
func NewBuffer(buf []byte) *Buffer	{ return &Buffer{buf: buf} }

// NewBufferString creates and initializes a new Buffer
// using string s as its initial contents.
func NewBufferString(s string) *Buffer {
	buf := make([]byte, len(s));
	copyString(buf, 0, s);
	return &Buffer{buf: buf};
}

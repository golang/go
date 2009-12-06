// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes

// Simple byte buffer for marshaling data.

import (
	"io";
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
	if len(src) == 1 {
		dst[doff] = src[0];
		return;
	}
	copy(dst[doff:], src);
}

// A Buffer is a variable-sized buffer of bytes
// with Read and Write methods.
// The zero value for Buffer is an empty buffer ready to use.
type Buffer struct {
	buf		[]byte;		// contents are the bytes buf[off : len(buf)]
	off		int;		// read at &buf[off], write at &buf[len(buf)]
	oneByte		[1]byte;	// avoid allocation of slice on each WriteByte
	bootstrap	[64]byte;	// memory to hold first slice; helps small buffers (Printf) avoid allocation.
}

// Bytes returns the contents of the unread portion of the buffer;
// len(b.Bytes()) == b.Len().
func (b *Buffer) Bytes() []byte	{ return b.buf[b.off:] }

// String returns the contents of the unread portion of the buffer
// as a string.  If the Buffer is a nil pointer, it returns "<nil>".
func (b *Buffer) String() string {
	if b == nil {
		// Special case, useful in debugging.
		return "<nil>"
	}
	return string(b.buf[b.off:]);
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
	b.buf = b.buf[0 : b.off+n];
}

// Reset resets the buffer so it has no content.
// b.Reset() is the same as b.Truncate(0).
func (b *Buffer) Reset()	{ b.Truncate(0) }

// Resize buffer to guarantee enough space for n more bytes.
// After this call, the state of b.buf is inconsistent.
// It must be fixed up as is done in Write and WriteString.
func (b *Buffer) resize(n int) {
	var buf []byte;
	if b.buf == nil && n <= len(b.bootstrap) {
		buf = &b.bootstrap
	} else {
		buf = b.buf;
		if len(b.buf)+n > cap(b.buf) {
			// not enough space anywhere
			buf = make([]byte, 2*cap(b.buf)+n)
		}
		copy(buf, b.buf[b.off:]);
	}
	b.buf = buf;
	b.off = 0;
}

// Write appends the contents of p to the buffer.  The return
// value n is the length of p; err is always nil.
func (b *Buffer) Write(p []byte) (n int, err os.Error) {
	m := b.Len();
	n = len(p);
	if len(b.buf)+n > cap(b.buf) {
		b.resize(n)
	}
	b.buf = b.buf[0 : b.off+m+n];
	copyBytes(b.buf, b.off+m, p);
	return n, nil;
}

// WriteString appends the contents of s to the buffer.  The return
// value n is the length of s; err is always nil.
func (b *Buffer) WriteString(s string) (n int, err os.Error) {
	m := b.Len();
	n = len(s);
	if len(b.buf)+n > cap(b.buf) {
		b.resize(n)
	}
	b.buf = b.buf[0 : b.off+m+n];
	copyString(b.buf, b.off+m, s);
	return n, nil;
}

// MinRead is the minimum slice size passed to a Read call by
// Buffer.ReadFrom.  As long as the Buffer has at least MinRead bytes beyond
// what is required to hold the contents of r, ReadFrom will not grow the
// underlying buffer.
const MinRead = 512

// ReadFrom reads data from r until EOF and appends it to the buffer.
// The return value n is the number of bytes read.
// Any error except os.EOF encountered during the read
// is also returned.
func (b *Buffer) ReadFrom(r io.Reader) (n int64, err os.Error) {
	for {
		if cap(b.buf)-len(b.buf) < MinRead {
			var newBuf []byte;
			// can we get space without allocation?
			if b.off+cap(b.buf)-len(b.buf) >= MinRead {
				// reuse beginning of buffer
				newBuf = b.buf[0 : len(b.buf)-b.off]
			} else {
				// not enough space at end; put space on end
				newBuf = make([]byte, len(b.buf)-b.off, 2*(cap(b.buf)-b.off)+MinRead)
			}
			copy(newBuf, b.buf[b.off:]);
			b.buf = newBuf;
			b.off = 0;
		}
		m, e := r.Read(b.buf[len(b.buf):cap(b.buf)]);
		b.buf = b.buf[b.off : len(b.buf)+m];
		n += int64(m);
		if e == os.EOF {
			break
		}
		if e != nil {
			return n, e
		}
	}
	return n, nil;	// err is EOF, so return nil explicitly
}

// WriteTo writes data to w until the buffer is drained or an error
// occurs. The return value n is the number of bytes written.
// Any error encountered during the write is also returned.
func (b *Buffer) WriteTo(w io.Writer) (n int64, err os.Error) {
	for b.off < len(b.buf) {
		m, e := w.Write(b.buf[b.off:]);
		n += int64(m);
		b.off += m;
		if e != nil {
			return n, e
		}
	}
	return;
}

// WriteByte appends the byte c to the buffer.
// The returned error is always nil, but is included
// to match bufio.Writer's WriteByte.
func (b *Buffer) WriteByte(c byte) os.Error {
	b.oneByte[0] = c;
	b.Write(&b.oneByte);
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

	copyBytes(p, 0, b.buf[b.off:b.off+n]);
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

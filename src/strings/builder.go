// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import (
	"errors"
	"io"
	"unicode/utf8"
	"unsafe"
)

// A Builder is used to efficiently build a string using Write methods.
// It minimizes memory copying. The zero value is ready to use.
type Builder struct {
	buf []byte
}

// String returns the accumulated string.
func (b *Builder) String() string {
	return *(*string)(unsafe.Pointer(&b.buf))
}

// Len returns the number of accumulated bytes; b.Len() == len(b.String()).
func (b *Builder) Len() int { return len(b.buf) }

// Reset resets the Builder to be empty.
func (b *Builder) Reset() { b.buf = nil }

const maxInt = int(^uint(0) >> 1)

// grow copies the buffer to a new, larger buffer so that there are at least n
// bytes of capacity beyond len(b.buf).
func (b *Builder) grow(n int) {
	buf := make([]byte, len(b.buf), 2*cap(b.buf)+n)
	copy(buf, b.buf)
	b.buf = buf
}

// Grow grows b's capacity, if necessary, to guarantee space for
// another n bytes. After Grow(n), at least n bytes can be written to b
// without another allocation. If n is negative, Grow panics.
func (b *Builder) Grow(n int) {
	if n < 0 {
		panic("strings.Builder.Grow: negative count")
	}
	if cap(b.buf)-len(b.buf) < n {
		b.grow(n)
	}
}

// Write appends the contents of p to b's buffer.
// Write always returns len(p), nil.
func (b *Builder) Write(p []byte) (int, error) {
	b.buf = append(b.buf, p...)
	return len(p), nil
}

// WriteByte appends the byte c to b's buffer.
// The returned error is always nil.
func (b *Builder) WriteByte(c byte) error {
	b.buf = append(b.buf, c)
	return nil
}

// WriteRune appends the UTF-8 encoding of Unicode code point r to b's buffer.
// It returns the length of r and a nil error.
func (b *Builder) WriteRune(r rune) (int, error) {
	if r < utf8.RuneSelf {
		b.buf = append(b.buf, byte(r))
		return 1, nil
	}
	l := len(b.buf)
	if cap(b.buf)-l < utf8.UTFMax {
		b.grow(utf8.UTFMax)
	}
	n := utf8.EncodeRune(b.buf[l:l+utf8.UTFMax], r)
	b.buf = b.buf[:l+n]
	return n, nil
}

// WriteString appends the contents of s to b's buffer.
// It returns the length of s and a nil error.
func (b *Builder) WriteString(s string) (int, error) {
	b.buf = append(b.buf, s...)
	return len(s), nil
}

// minRead is the minimum slice passed to a Read call by Builder.ReadFrom.
// It is the same as bytes.MinRead.
const minRead = 512

// errNegativeRead is the panic value if the reader passed to Builder.ReadFrom
// returns a negative count.
var errNegativeRead = errors.New("strings.Builder: reader returned negative count from Read")

// ReadFrom reads data from r until EOF and appends it to b's buffer.
// The return value n is the number of bytes read.
// Any error except io.EOF encountered during the read is also returned.
func (b *Builder) ReadFrom(r io.Reader) (n int64, err error) {
	for {
		l := len(b.buf)
		if cap(b.buf)-l < minRead {
			b.grow(minRead)
		}
		m, e := r.Read(b.buf[l:cap(b.buf)])
		if m < 0 {
			panic(errNegativeRead)
		}
		b.buf = b.buf[:l+m]
		n += int64(m)
		if e == io.EOF {
			return n, nil
		}
		if e != nil {
			return n, e
		}
	}
}

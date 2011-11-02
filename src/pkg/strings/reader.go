// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import (
	"errors"
	"io"
	"utf8"
)

// A Reader implements the io.Reader, io.ByteScanner, and
// io.RuneScanner interfaces by reading from a string.
type Reader struct {
	s        string
	i        int // current reading index
	prevRune int // index of previous rune; or < 0
}

// Len returns the number of bytes of the unread portion of the
// string.
func (r *Reader) Len() int {
	return len(r.s) - r.i
}

func (r *Reader) Read(b []byte) (n int, err error) {
	if r.i >= len(r.s) {
		return 0, io.EOF
	}
	n = copy(b, r.s[r.i:])
	r.i += n
	r.prevRune = -1
	return
}

func (r *Reader) ReadByte() (b byte, err error) {
	if r.i >= len(r.s) {
		return 0, io.EOF
	}
	b = r.s[r.i]
	r.i++
	r.prevRune = -1
	return
}

// UnreadByte moves the reading position back by one byte.
// It is an error to call UnreadByte if nothing has been
// read yet.
func (r *Reader) UnreadByte() error {
	if r.i <= 0 {
		return errors.New("strings.Reader: at beginning of string")
	}
	r.i--
	r.prevRune = -1
	return nil
}

// ReadRune reads and returns the next UTF-8-encoded
// Unicode code point from the buffer.
// If no bytes are available, the error returned is os.EOF.
// If the bytes are an erroneous UTF-8 encoding, it
// consumes one byte and returns U+FFFD, 1.
func (r *Reader) ReadRune() (ch rune, size int, err error) {
	if r.i >= len(r.s) {
		return 0, 0, io.EOF
	}
	r.prevRune = r.i
	if c := r.s[r.i]; c < utf8.RuneSelf {
		r.i++
		return rune(c), 1, nil
	}
	ch, size = utf8.DecodeRuneInString(r.s[r.i:])
	r.i += size
	return
}

// UnreadRune causes the next call to ReadRune to return the same rune
// as the previous call to ReadRune.
// The last method called on r must have been ReadRune.
func (r *Reader) UnreadRune() error {
	if r.prevRune < 0 {
		return errors.New("strings.Reader: previous operation was not ReadRune")
	}
	r.i = r.prevRune
	r.prevRune = -1
	return nil
}

// NewReader returns a new Reader reading from s.
// It is similar to bytes.NewBufferString but more efficient and read-only.
func NewReader(s string) *Reader { return &Reader{s, 0, -1} }

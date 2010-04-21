// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import (
	"os"
	"utf8"
)

// A Reader satisfies calls to Read, ReadByte, and ReadRune by
// reading from a string.
type Reader string

func (r *Reader) Read(b []byte) (n int, err os.Error) {
	s := *r
	if len(s) == 0 {
		return 0, os.EOF
	}
	for n < len(s) && n < len(b) {
		b[n] = s[n]
		n++
	}
	*r = s[n:]
	return
}

func (r *Reader) ReadByte() (b byte, err os.Error) {
	s := *r
	if len(s) == 0 {
		return 0, os.EOF
	}
	b = s[0]
	*r = s[1:]
	return
}

// ReadRune reads and returns the next UTF-8-encoded
// Unicode code point from the buffer.
// If no bytes are available, the error returned is os.EOF.
// If the bytes are an erroneous UTF-8 encoding, it
// consumes one byte and returns U+FFFD, 1.
func (r *Reader) ReadRune() (rune int, size int, err os.Error) {
	s := *r
	if len(s) == 0 {
		return 0, 0, os.EOF
	}
	c := s[0]
	if c < utf8.RuneSelf {
		*r = s[1:]
		return int(c), 1, nil
	}
	rune, size = utf8.DecodeRuneInString(string(s))
	*r = s[size:]
	return
}

// NewReader returns a new Reader reading from s.
// It is similar to bytes.NewBufferString but more efficient and read-only.
func NewReader(s string) *Reader { return (*Reader)(&s) }

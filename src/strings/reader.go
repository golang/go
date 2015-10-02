// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import (
	"errors"
	"io"
	"unicode/utf8"
)

// A Reader implements the io.Reader, io.ReaderAt, io.Seeker, io.WriterTo,
// io.ByteScanner, and io.RuneScanner interfaces by reading
// from a string.
type Reader struct {
	s        string
	i        int64 // current reading index
	prevRune int   // index of previous rune; or < 0
}

// Len returns the number of bytes of the unread portion of the
// string.
func (r *Reader) Len() int {
	if r.i >= int64(len(r.s)) {
		return 0
	}
	return int(int64(len(r.s)) - r.i)
}

// Size returns the original length of the underlying string.
// Size is the number of bytes available for reading via ReadAt.
// The returned value is always the same and is not affected by calls
// to any other method.
func (r *Reader) Size() int64 { return int64(len(r.s)) }

func (r *Reader) Read(b []byte) (n int, err error) {
	if len(b) == 0 {
		return 0, nil
	}
	if r.i >= int64(len(r.s)) {
		return 0, io.EOF
	}
	r.prevRune = -1
	n = copy(b, r.s[r.i:])
	r.i += int64(n)
	return
}

func (r *Reader) ReadAt(b []byte, off int64) (n int, err error) {
	// cannot modify state - see io.ReaderAt
	if off < 0 {
		return 0, errors.New("strings.Reader.ReadAt: negative offset")
	}
	if off >= int64(len(r.s)) {
		return 0, io.EOF
	}
	n = copy(b, r.s[off:])
	if n < len(b) {
		err = io.EOF
	}
	return
}

func (r *Reader) ReadByte() (b byte, err error) {
	r.prevRune = -1
	if r.i >= int64(len(r.s)) {
		return 0, io.EOF
	}
	b = r.s[r.i]
	r.i++
	return
}

func (r *Reader) UnreadByte() error {
	r.prevRune = -1
	if r.i <= 0 {
		return errors.New("strings.Reader.UnreadByte: at beginning of string")
	}
	r.i--
	return nil
}

func (r *Reader) ReadRune() (ch rune, size int, err error) {
	if r.i >= int64(len(r.s)) {
		r.prevRune = -1
		return 0, 0, io.EOF
	}
	r.prevRune = int(r.i)
	if c := r.s[r.i]; c < utf8.RuneSelf {
		r.i++
		return rune(c), 1, nil
	}
	ch, size = utf8.DecodeRuneInString(r.s[r.i:])
	r.i += int64(size)
	return
}

func (r *Reader) UnreadRune() error {
	if r.prevRune < 0 {
		return errors.New("strings.Reader.UnreadRune: previous operation was not ReadRune")
	}
	r.i = int64(r.prevRune)
	r.prevRune = -1
	return nil
}

// Seek implements the io.Seeker interface.
func (r *Reader) Seek(offset int64, whence int) (int64, error) {
	r.prevRune = -1
	var abs int64
	switch whence {
	case 0:
		abs = offset
	case 1:
		abs = int64(r.i) + offset
	case 2:
		abs = int64(len(r.s)) + offset
	default:
		return 0, errors.New("strings.Reader.Seek: invalid whence")
	}
	if abs < 0 {
		return 0, errors.New("strings.Reader.Seek: negative position")
	}
	r.i = abs
	return abs, nil
}

// WriteTo implements the io.WriterTo interface.
func (r *Reader) WriteTo(w io.Writer) (n int64, err error) {
	r.prevRune = -1
	if r.i >= int64(len(r.s)) {
		return 0, nil
	}
	s := r.s[r.i:]
	m, err := io.WriteString(w, s)
	if m > len(s) {
		panic("strings.Reader.WriteTo: invalid WriteString count")
	}
	r.i += int64(m)
	n = int64(m)
	if m != len(s) && err == nil {
		err = io.ErrShortWrite
	}
	return
}

// NewReader returns a new Reader reading from s.
// It is similar to bytes.NewBufferString but more efficient and read-only.
func NewReader(s string) *Reader { return &Reader{s, 0, -1} }

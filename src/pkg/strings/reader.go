// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import "os"

// A Reader satisfies calls to Read and ReadByte by
// reading from a string.
type Reader string

func (r *Reader) Read(b []byte) (n int, err os.Error) {
	s := *r;
	if len(s) == 0 {
		return 0, os.EOF
	}
	for n < len(s) && n < len(b) {
		b[n] = s[n];
		n++;
	}
	*r = s[n:];
	return;
}

func (r *Reader) ReadByte() (b byte, err os.Error) {
	s := *r;
	if len(s) == 0 {
		return 0, os.EOF
	}
	b = s[0];
	*r = s[1:];
	return;
}

// NewReader returns a new Reader reading from s.
// It is similar to bytes.NewBufferString but more efficient and read-only.
func NewReader(s string) *Reader	{ return (*Reader)(&s) }

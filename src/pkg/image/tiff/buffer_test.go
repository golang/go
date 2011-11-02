// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"io"
	"strings"
	"testing"
)

var readAtTests = []struct {
	n   int
	off int64
	s   string
	err error
}{
	{2, 0, "ab", nil},
	{6, 0, "abcdef", nil},
	{3, 3, "def", nil},
	{3, 5, "f", io.EOF},
	{3, 6, "", io.EOF},
}

func TestReadAt(t *testing.T) {
	r := newReaderAt(strings.NewReader("abcdef"))
	b := make([]byte, 10)
	for _, test := range readAtTests {
		n, err := r.ReadAt(b[:test.n], test.off)
		s := string(b[:n])
		if s != test.s || err != test.err {
			t.Errorf("buffer.ReadAt(<%v bytes>, %v): got %v, %q; want %v, %q", test.n, test.off, err, s, test.err, test.s)
		}
	}
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package openpgp

import (
	"bytes"
	"testing"
)

type recordingHash struct {
	buf *bytes.Buffer
}

func (r recordingHash) Write(b []byte) (n int, err error) {
	return r.buf.Write(b)
}

func (r recordingHash) Sum(in []byte) []byte {
	return append(in, r.buf.Bytes()...)
}

func (r recordingHash) Reset() {
	panic("shouldn't be called")
}

func (r recordingHash) Size() int {
	panic("shouldn't be called")
}

func (r recordingHash) BlockSize() int {
	panic("shouldn't be called")
}

func testCanonicalText(t *testing.T, input, expected string) {
	r := recordingHash{bytes.NewBuffer(nil)}
	c := NewCanonicalTextHash(r)
	c.Write([]byte(input))
	result := c.Sum(nil)
	if expected != string(result) {
		t.Errorf("input: %x got: %x want: %x", input, result, expected)
	}
}

func TestCanonicalText(t *testing.T) {
	testCanonicalText(t, "foo\n", "foo\r\n")
	testCanonicalText(t, "foo", "foo")
	testCanonicalText(t, "foo\r\n", "foo\r\n")
	testCanonicalText(t, "foo\r\nbar", "foo\r\nbar")
	testCanonicalText(t, "foo\r\nbar\n\n", "foo\r\nbar\r\n\r\n")
}

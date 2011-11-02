// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io_test

import (
	. "io"
	"bytes"
	"crypto/sha1"
	"fmt"
	"strings"
	"testing"
)

func TestMultiReader(t *testing.T) {
	var mr Reader
	var buf []byte
	nread := 0
	withFooBar := func(tests func()) {
		r1 := strings.NewReader("foo ")
		r2 := strings.NewReader("")
		r3 := strings.NewReader("bar")
		mr = MultiReader(r1, r2, r3)
		buf = make([]byte, 20)
		tests()
	}
	expectRead := func(size int, expected string, eerr error) {
		nread++
		n, gerr := mr.Read(buf[0:size])
		if n != len(expected) {
			t.Errorf("#%d, expected %d bytes; got %d",
				nread, len(expected), n)
		}
		got := string(buf[0:n])
		if got != expected {
			t.Errorf("#%d, expected %q; got %q",
				nread, expected, got)
		}
		if gerr != eerr {
			t.Errorf("#%d, expected error %v; got %v",
				nread, eerr, gerr)
		}
		buf = buf[n:]
	}
	withFooBar(func() {
		expectRead(2, "fo", nil)
		expectRead(5, "o ", nil)
		expectRead(5, "bar", nil)
		expectRead(5, "", EOF)
	})
	withFooBar(func() {
		expectRead(4, "foo ", nil)
		expectRead(1, "b", nil)
		expectRead(3, "ar", nil)
		expectRead(1, "", EOF)
	})
	withFooBar(func() {
		expectRead(5, "foo ", nil)
	})
}

func TestMultiWriter(t *testing.T) {
	sha1 := sha1.New()
	sink := new(bytes.Buffer)
	mw := MultiWriter(sha1, sink)

	sourceString := "My input text."
	source := strings.NewReader(sourceString)
	written, err := Copy(mw, source)

	if written != int64(len(sourceString)) {
		t.Errorf("short write of %d, not %d", written, len(sourceString))
	}

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	sha1hex := fmt.Sprintf("%x", sha1.Sum())
	if sha1hex != "01cb303fa8c30a64123067c5aa6284ba7ec2d31b" {
		t.Error("incorrect sha1 value")
	}

	if sink.String() != sourceString {
		t.Errorf("expected %q; got %q", sourceString, sink.String())
	}
}

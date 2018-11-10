// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bytes"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

func TestReset(t *testing.T) {
	ss := []string{
		"lorem ipsum izzle fo rizzle",
		"the quick brown fox jumped over",
	}

	deflated := make([]bytes.Buffer, 2)
	for i, s := range ss {
		w, _ := NewWriter(&deflated[i], 1)
		w.Write([]byte(s))
		w.Close()
	}

	inflated := make([]bytes.Buffer, 2)

	f := NewReader(&deflated[0])
	io.Copy(&inflated[0], f)
	f.(Resetter).Reset(&deflated[1], nil)
	io.Copy(&inflated[1], f)
	f.Close()

	for i, s := range ss {
		if s != inflated[i].String() {
			t.Errorf("inflated[%d]:\ngot  %q\nwant %q", i, inflated[i], s)
		}
	}
}

func TestReaderTruncated(t *testing.T) {
	vectors := []struct{ input, output string }{
		{"\x00", ""},
		{"\x00\f", ""},
		{"\x00\f\x00", ""},
		{"\x00\f\x00\xf3\xff", ""},
		{"\x00\f\x00\xf3\xffhello", "hello"},
		{"\x00\f\x00\xf3\xffhello, world", "hello, world"},
		{"\x02", ""},
		{"\xf2H\xcd", "He"},
		{"\xf2H͙0a\u0084\t", "Hel\x90\x90\x90\x90\x90"},
		{"\xf2H͙0a\u0084\t\x00", "Hel\x90\x90\x90\x90\x90"},
	}

	for i, v := range vectors {
		r := strings.NewReader(v.input)
		zr := NewReader(r)
		b, err := ioutil.ReadAll(zr)
		if err != io.ErrUnexpectedEOF {
			t.Errorf("test %d, error mismatch: got %v, want io.ErrUnexpectedEOF", i, err)
		}
		if string(b) != v.output {
			t.Errorf("test %d, output mismatch: got %q, want %q", i, b, v.output)
		}
	}
}

func TestResetDict(t *testing.T) {
	dict := []byte("the lorem fox")
	ss := []string{
		"lorem ipsum izzle fo rizzle",
		"the quick brown fox jumped over",
	}

	deflated := make([]bytes.Buffer, len(ss))
	for i, s := range ss {
		w, _ := NewWriterDict(&deflated[i], DefaultCompression, dict)
		w.Write([]byte(s))
		w.Close()
	}

	inflated := make([]bytes.Buffer, len(ss))

	f := NewReader(nil)
	for i := range inflated {
		f.(Resetter).Reset(&deflated[i], dict)
		io.Copy(&inflated[i], f)
	}
	f.Close()

	for i, s := range ss {
		if s != inflated[i].String() {
			t.Errorf("inflated[%d]:\ngot  %q\nwant %q", i, inflated[i], s)
		}
	}
}

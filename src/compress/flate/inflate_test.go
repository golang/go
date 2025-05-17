// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bufio"
	"bytes"
	"io"
	"os"
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
		b, err := io.ReadAll(zr)
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

func TestReaderReusesReaderBuffer(t *testing.T) {
	encodedReader := bytes.NewReader([]byte{})
	encodedNotByteReader := struct{ io.Reader }{encodedReader}

	t.Run("BufferIsReused", func(t *testing.T) {
		f := NewReader(encodedNotByteReader).(*decompressor)
		bufioR, ok := f.r.(*bufio.Reader)
		if !ok {
			t.Fatalf("bufio.Reader should be created")
		}
		f.Reset(encodedNotByteReader, nil)
		if bufioR != f.r {
			t.Fatalf("bufio.Reader was not reused")
		}
	})
	t.Run("BufferIsNotReusedWhenGotByteReader", func(t *testing.T) {
		f := NewReader(encodedNotByteReader).(*decompressor)
		if _, ok := f.r.(*bufio.Reader); !ok {
			t.Fatalf("bufio.Reader should be created")
		}
		f.Reset(encodedReader, nil)
		if f.r != encodedReader {
			t.Fatalf("provided io.ByteReader should be used directly")
		}
	})
	t.Run("BufferIsCreatedAfterByteReader", func(t *testing.T) {
		for i, r := range []io.Reader{encodedReader, bufio.NewReader(encodedReader)} {
			f := NewReader(r).(*decompressor)
			if f.r != r {
				t.Fatalf("provided io.ByteReader should be used directly, i=%d", i)
			}
			f.Reset(encodedNotByteReader, nil)
			if _, ok := f.r.(*bufio.Reader); !ok {
				t.Fatalf("bufio.Reader should be created, i=%d", i)
			}
		}
	})
}

func TestReaderPartialBlock(t *testing.T) {
	data, err := os.ReadFile("testdata/partial-block")
	if err != nil {
		t.Error(err)
	}

	r := NewReader(bytes.NewReader(data))
	rb := make([]byte, 32)
	n, err := r.Read(rb)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}

	expected := "hello, world"
	actual := string(rb[:n])
	if expected != actual {
		t.Fatalf("expected: %v, got: %v", expected, actual)
	}
}

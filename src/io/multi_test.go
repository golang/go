// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io_test

import (
	"bytes"
	"crypto/sha1"
	"errors"
	"fmt"
	. "io"
	"io/ioutil"
	"runtime"
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
	sink := new(bytes.Buffer)
	// Hide bytes.Buffer's WriteString method:
	testMultiWriter(t, struct {
		Writer
		fmt.Stringer
	}{sink, sink})
}

func TestMultiWriter_String(t *testing.T) {
	testMultiWriter(t, new(bytes.Buffer))
}

// test that a multiWriter.WriteString calls results in at most 1 allocation,
// even if multiple targets don't support WriteString.
func TestMultiWriter_WriteStringSingleAlloc(t *testing.T) {
	var sink1, sink2 bytes.Buffer
	type simpleWriter struct { // hide bytes.Buffer's WriteString
		Writer
	}
	mw := MultiWriter(simpleWriter{&sink1}, simpleWriter{&sink2})
	allocs := int(testing.AllocsPerRun(1000, func() {
		WriteString(mw, "foo")
	}))
	if allocs != 1 {
		t.Errorf("num allocations = %d; want 1", allocs)
	}
}

type writeStringChecker struct{ called bool }

func (c *writeStringChecker) WriteString(s string) (n int, err error) {
	c.called = true
	return len(s), nil
}

func (c *writeStringChecker) Write(p []byte) (n int, err error) {
	return len(p), nil
}

func TestMultiWriter_StringCheckCall(t *testing.T) {
	var c writeStringChecker
	mw := MultiWriter(&c)
	WriteString(mw, "foo")
	if !c.called {
		t.Error("did not see WriteString call to writeStringChecker")
	}
}

func testMultiWriter(t *testing.T, sink interface {
	Writer
	fmt.Stringer
}) {
	sha1 := sha1.New()
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

	sha1hex := fmt.Sprintf("%x", sha1.Sum(nil))
	if sha1hex != "01cb303fa8c30a64123067c5aa6284ba7ec2d31b" {
		t.Error("incorrect sha1 value")
	}

	if sink.String() != sourceString {
		t.Errorf("expected %q; got %q", sourceString, sink.String())
	}
}

// Test that MultiReader copies the input slice and is insulated from future modification.
func TestMultiReaderCopy(t *testing.T) {
	slice := []Reader{strings.NewReader("hello world")}
	r := MultiReader(slice...)
	slice[0] = nil
	data, err := ioutil.ReadAll(r)
	if err != nil || string(data) != "hello world" {
		t.Errorf("ReadAll() = %q, %v, want %q, nil", data, err, "hello world")
	}
}

// Test that MultiWriter copies the input slice and is insulated from future modification.
func TestMultiWriterCopy(t *testing.T) {
	var buf bytes.Buffer
	slice := []Writer{&buf}
	w := MultiWriter(slice...)
	slice[0] = nil
	n, err := w.Write([]byte("hello world"))
	if err != nil || n != 11 {
		t.Errorf("Write(`hello world`) = %d, %v, want 11, nil", n, err)
	}
	if buf.String() != "hello world" {
		t.Errorf("buf.String() = %q, want %q", buf.String(), "hello world")
	}
}

// readerFunc is an io.Reader implemented by the underlying func.
type readerFunc func(p []byte) (int, error)

func (f readerFunc) Read(p []byte) (int, error) {
	return f(p)
}

// Test that MultiReader properly flattens chained multiReaders when Read is called
func TestMultiReaderFlatten(t *testing.T) {
	pc := make([]uintptr, 1000) // 1000 should fit the full stack
	var myDepth = runtime.Callers(0, pc)
	var readDepth int // will contain the depth from which fakeReader.Read was called
	var r Reader = MultiReader(readerFunc(func(p []byte) (int, error) {
		readDepth = runtime.Callers(1, pc)
		return 0, errors.New("irrelevant")
	}))

	// chain a bunch of multiReaders
	for i := 0; i < 100; i++ {
		r = MultiReader(r)
	}

	r.Read(nil) // don't care about errors, just want to check the call-depth for Read

	if readDepth != myDepth+2 { // 2 should be multiReader.Read and fakeReader.Read
		t.Errorf("multiReader did not flatten chained multiReaders: expected readDepth %d, got %d",
			myDepth+2, readDepth)
	}
}

// byteAndEOFReader is a Reader which reads one byte (the underlying
// byte) and io.EOF at once in its Read call.
type byteAndEOFReader byte

func (b byteAndEOFReader) Read(p []byte) (n int, err error) {
	if len(p) == 0 {
		// Read(0 bytes) is useless. We expect no such useless
		// calls in this test.
		panic("unexpected call")
	}
	p[0] = byte(b)
	return 1, EOF
}

// In Go 1.7, this yielded bytes forever.
func TestMultiReaderSingleByteWithEOF(t *testing.T) {
	got, err := ioutil.ReadAll(LimitReader(MultiReader(byteAndEOFReader('a'), byteAndEOFReader('b')), 10))
	if err != nil {
		t.Fatal(err)
	}
	const want = "ab"
	if string(got) != want {
		t.Errorf("got %q; want %q", got, want)
	}
}

// Test that a reader returning (n, EOF) at the end of an MultiReader
// chain continues to return EOF on its final read, rather than
// yielding a (0, EOF).
func TestMultiReaderFinalEOF(t *testing.T) {
	r := MultiReader(bytes.NewReader(nil), byteAndEOFReader('a'))
	buf := make([]byte, 2)
	n, err := r.Read(buf)
	if n != 1 || err != EOF {
		t.Errorf("got %v, %v; want 1, EOF", n, err)
	}
}

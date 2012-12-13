// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io_test

import (
	"bytes"
	. "io"
	"strings"
	"testing"
)

// An version of bytes.Buffer without ReadFrom and WriteTo
type Buffer struct {
	bytes.Buffer
	ReaderFrom // conflicts with and hides bytes.Buffer's ReaderFrom.
	WriterTo   // conflicts with and hides bytes.Buffer's WriterTo.
}

// Simple tests, primarily to verify the ReadFrom and WriteTo callouts inside Copy and CopyN.

func TestCopy(t *testing.T) {
	rb := new(Buffer)
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	Copy(wb, rb)
	if wb.String() != "hello, world." {
		t.Errorf("Copy did not work properly")
	}
}

func TestCopyReadFrom(t *testing.T) {
	rb := new(Buffer)
	wb := new(bytes.Buffer) // implements ReadFrom.
	rb.WriteString("hello, world.")
	Copy(wb, rb)
	if wb.String() != "hello, world." {
		t.Errorf("Copy did not work properly")
	}
}

func TestCopyWriteTo(t *testing.T) {
	rb := new(bytes.Buffer) // implements WriteTo.
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	Copy(wb, rb)
	if wb.String() != "hello, world." {
		t.Errorf("Copy did not work properly")
	}
}

func TestCopyN(t *testing.T) {
	rb := new(Buffer)
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	CopyN(wb, rb, 5)
	if wb.String() != "hello" {
		t.Errorf("CopyN did not work properly")
	}
}

func TestCopyNReadFrom(t *testing.T) {
	rb := new(Buffer)
	wb := new(bytes.Buffer) // implements ReadFrom.
	rb.WriteString("hello")
	CopyN(wb, rb, 5)
	if wb.String() != "hello" {
		t.Errorf("CopyN did not work properly")
	}
}

func TestCopyNWriteTo(t *testing.T) {
	rb := new(bytes.Buffer) // implements WriteTo.
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	CopyN(wb, rb, 5)
	if wb.String() != "hello" {
		t.Errorf("CopyN did not work properly")
	}
}

type noReadFrom struct {
	w Writer
}

func (w *noReadFrom) Write(p []byte) (n int, err error) {
	return w.w.Write(p)
}

func TestCopyNEOF(t *testing.T) {
	// Test that EOF behavior is the same regardless of whether
	// argument to CopyN has ReadFrom.

	b := new(bytes.Buffer)

	n, err := CopyN(&noReadFrom{b}, strings.NewReader("foo"), 3)
	if n != 3 || err != nil {
		t.Errorf("CopyN(noReadFrom, foo, 3) = %d, %v; want 3, nil", n, err)
	}

	n, err = CopyN(&noReadFrom{b}, strings.NewReader("foo"), 4)
	if n != 3 || err != EOF {
		t.Errorf("CopyN(noReadFrom, foo, 4) = %d, %v; want 3, EOF", n, err)
	}

	n, err = CopyN(b, strings.NewReader("foo"), 3) // b has read from
	if n != 3 || err != nil {
		t.Errorf("CopyN(bytes.Buffer, foo, 3) = %d, %v; want 3, nil", n, err)
	}

	n, err = CopyN(b, strings.NewReader("foo"), 4) // b has read from
	if n != 3 || err != EOF {
		t.Errorf("CopyN(bytes.Buffer, foo, 4) = %d, %v; want 3, EOF", n, err)
	}
}

func TestReadAtLeast(t *testing.T) {
	var rb bytes.Buffer
	testReadAtLeast(t, &rb)
}

// A version of bytes.Buffer that returns n > 0, EOF on Read
// when the input is exhausted.
type dataAndEOFBuffer struct {
	bytes.Buffer
}

func (r *dataAndEOFBuffer) Read(p []byte) (n int, err error) {
	n, err = r.Buffer.Read(p)
	if n > 0 && r.Buffer.Len() == 0 && err == nil {
		err = EOF
	}
	return
}

func TestReadAtLeastWithDataAndEOF(t *testing.T) {
	var rb dataAndEOFBuffer
	testReadAtLeast(t, &rb)
}

func testReadAtLeast(t *testing.T, rb ReadWriter) {
	rb.Write([]byte("0123"))
	buf := make([]byte, 2)
	n, err := ReadAtLeast(rb, buf, 2)
	if err != nil {
		t.Error(err)
	}
	n, err = ReadAtLeast(rb, buf, 4)
	if err != ErrShortBuffer {
		t.Errorf("expected ErrShortBuffer got %v", err)
	}
	if n != 0 {
		t.Errorf("expected to have read 0 bytes, got %v", n)
	}
	n, err = ReadAtLeast(rb, buf, 1)
	if err != nil {
		t.Error(err)
	}
	if n != 2 {
		t.Errorf("expected to have read 2 bytes, got %v", n)
	}
	n, err = ReadAtLeast(rb, buf, 2)
	if err != EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected to have read 0 bytes, got %v", n)
	}
	rb.Write([]byte("4"))
	n, err = ReadAtLeast(rb, buf, 2)
	if err != ErrUnexpectedEOF {
		t.Errorf("expected ErrUnexpectedEOF, got %v", err)
	}
	if n != 1 {
		t.Errorf("expected to have read 1 bytes, got %v", n)
	}
}

func TestTeeReader(t *testing.T) {
	src := []byte("hello, world")
	dst := make([]byte, len(src))
	rb := bytes.NewBuffer(src)
	wb := new(bytes.Buffer)
	r := TeeReader(rb, wb)
	if n, err := ReadFull(r, dst); err != nil || n != len(src) {
		t.Fatalf("ReadFull(r, dst) = %d, %v; want %d, nil", n, err, len(src))
	}
	if !bytes.Equal(dst, src) {
		t.Errorf("bytes read = %q want %q", dst, src)
	}
	if !bytes.Equal(wb.Bytes(), src) {
		t.Errorf("bytes written = %q want %q", wb.Bytes(), src)
	}
	if n, err := r.Read(dst); n != 0 || err != EOF {
		t.Errorf("r.Read at EOF = %d, %v want 0, EOF", n, err)
	}
	rb = bytes.NewBuffer(src)
	pr, pw := Pipe()
	pr.Close()
	r = TeeReader(rb, pw)
	if n, err := ReadFull(r, dst); n != 0 || err != ErrClosedPipe {
		t.Errorf("closed tee: ReadFull(r, dst) = %d, %v; want 0, EPIPE", n, err)
	}
}

func TestSectionReader_ReadAt(tst *testing.T) {
	dat := "a long sample data, 1234567890"
	tests := []struct {
		data   string
		off    int
		n      int
		bufLen int
		at     int
		exp    string
		err    error
	}{
		{data: "", off: 0, n: 10, bufLen: 2, at: 0, exp: "", err: EOF},
		{data: dat, off: 0, n: len(dat), bufLen: 0, at: 0, exp: "", err: nil},
		{data: dat, off: len(dat), n: 1, bufLen: 1, at: 0, exp: "", err: EOF},
		{data: dat, off: 0, n: len(dat) + 2, bufLen: len(dat), at: 0, exp: dat, err: nil},
		{data: dat, off: 0, n: len(dat), bufLen: len(dat) / 2, at: 0, exp: dat[:len(dat)/2], err: nil},
		{data: dat, off: 0, n: len(dat), bufLen: len(dat), at: 0, exp: dat, err: nil},
		{data: dat, off: 0, n: len(dat), bufLen: len(dat) / 2, at: 2, exp: dat[2 : 2+len(dat)/2], err: nil},
		{data: dat, off: 3, n: len(dat), bufLen: len(dat) / 2, at: 2, exp: dat[5 : 5+len(dat)/2], err: nil},
		{data: dat, off: 3, n: len(dat) / 2, bufLen: len(dat)/2 - 2, at: 2, exp: dat[5 : 5+len(dat)/2-2], err: nil},
		{data: dat, off: 3, n: len(dat) / 2, bufLen: len(dat)/2 + 2, at: 2, exp: dat[5 : 5+len(dat)/2-2], err: EOF},
	}
	for i, t := range tests {
		r := strings.NewReader(t.data)
		s := NewSectionReader(r, int64(t.off), int64(t.n))
		buf := make([]byte, t.bufLen)
		if n, err := s.ReadAt(buf, int64(t.at)); n != len(t.exp) || string(buf[:n]) != t.exp || err != t.err {
			tst.Fatalf("%d: ReadAt(%d) = %q, %v; expected %q, %v", i, t.at, buf[:n], err, t.exp, t.err)
		}
	}
}

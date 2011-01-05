// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io_test

import (
	"bytes"
	. "io"
	"os"
	"strings"
	"testing"
)

// An version of bytes.Buffer without ReadFrom and WriteTo
type Buffer struct {
	bytes.Buffer
	ReaderFrom // conflicts with and hides bytes.Buffer's ReaderFrom.
	WriterTo   // conflicts with and hides bytes.Buffer's WriterTo.
}

// Simple tests, primarily to verify the ReadFrom and WriteTo callouts inside Copy and Copyn.

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

func TestCopyn(t *testing.T) {
	rb := new(Buffer)
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	Copyn(wb, rb, 5)
	if wb.String() != "hello" {
		t.Errorf("Copyn did not work properly")
	}
}

func TestCopynReadFrom(t *testing.T) {
	rb := new(Buffer)
	wb := new(bytes.Buffer) // implements ReadFrom.
	rb.WriteString("hello")
	Copyn(wb, rb, 5)
	if wb.String() != "hello" {
		t.Errorf("Copyn did not work properly")
	}
}

func TestCopynWriteTo(t *testing.T) {
	rb := new(bytes.Buffer) // implements WriteTo.
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	Copyn(wb, rb, 5)
	if wb.String() != "hello" {
		t.Errorf("Copyn did not work properly")
	}
}

type noReadFrom struct {
	w Writer
}

func (w *noReadFrom) Write(p []byte) (n int, err os.Error) {
	return w.w.Write(p)
}

func TestCopynEOF(t *testing.T) {
	// Test that EOF behavior is the same regardless of whether
	// argument to Copyn has ReadFrom.

	b := new(bytes.Buffer)

	n, err := Copyn(&noReadFrom{b}, strings.NewReader("foo"), 3)
	if n != 3 || err != nil {
		t.Errorf("Copyn(noReadFrom, foo, 3) = %d, %v; want 3, nil", n, err)
	}

	n, err = Copyn(&noReadFrom{b}, strings.NewReader("foo"), 4)
	if n != 3 || err != os.EOF {
		t.Errorf("Copyn(noReadFrom, foo, 4) = %d, %v; want 3, EOF", n, err)
	}

	n, err = Copyn(b, strings.NewReader("foo"), 3) // b has read from
	if n != 3 || err != nil {
		t.Errorf("Copyn(bytes.Buffer, foo, 3) = %d, %v; want 3, nil", n, err)
	}

	n, err = Copyn(b, strings.NewReader("foo"), 4) // b has read from
	if n != 3 || err != os.EOF {
		t.Errorf("Copyn(bytes.Buffer, foo, 4) = %d, %v; want 3, EOF", n, err)
	}
}

func TestReadAtLeast(t *testing.T) {
	var rb bytes.Buffer
	rb.Write([]byte("0123"))
	buf := make([]byte, 2)
	n, err := ReadAtLeast(&rb, buf, 2)
	if err != nil {
		t.Error(err)
	}
	n, err = ReadAtLeast(&rb, buf, 4)
	if err != ErrShortBuffer {
		t.Errorf("expected ErrShortBuffer got %v", err)
	}
	if n != 0 {
		t.Errorf("expected to have read 0 bytes, got %v", n)
	}
	n, err = ReadAtLeast(&rb, buf, 1)
	if err != nil {
		t.Error(err)
	}
	if n != 2 {
		t.Errorf("expected to have read 2 bytes, got %v", n)
	}
	n, err = ReadAtLeast(&rb, buf, 2)
	if err != os.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected to have read 0 bytes, got %v", n)
	}
	rb.Write([]byte("4"))
	n, err = ReadAtLeast(&rb, buf, 2)
	if err != ErrUnexpectedEOF {
		t.Errorf("expected ErrUnexpectedEOF, got %v", err)
	}
	if n != 1 {
		t.Errorf("expected to have read 1 bytes, got %v", n)
	}
}

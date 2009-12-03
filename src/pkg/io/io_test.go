// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io_test

import (
	"bytes";
	. "io";
	"testing";
)

// An version of bytes.Buffer without ReadFrom and WriteTo
type Buffer struct {
	bytes.Buffer;
	ReaderFrom;	// conflicts with and hides bytes.Buffer's ReaderFrom.
	WriterTo;	// conflicts with and hides bytes.Buffer's WriterTo.
}

// Simple tests, primarily to verify the ReadFrom and WriteTo callouts inside Copy and Copyn.

func TestCopy(t *testing.T) {
	rb := new(Buffer);
	wb := new(Buffer);
	rb.WriteString("hello, world.");
	Copy(wb, rb);
	if wb.String() != "hello, world." {
		t.Errorf("Copy did not work properly")
	}
}

func TestCopyReadFrom(t *testing.T) {
	rb := new(Buffer);
	wb := new(bytes.Buffer);	// implements ReadFrom.
	rb.WriteString("hello, world.");
	Copy(wb, rb);
	if wb.String() != "hello, world." {
		t.Errorf("Copy did not work properly")
	}
}

func TestCopyWriteTo(t *testing.T) {
	rb := new(bytes.Buffer);	// implements WriteTo.
	wb := new(Buffer);
	rb.WriteString("hello, world.");
	Copy(wb, rb);
	if wb.String() != "hello, world." {
		t.Errorf("Copy did not work properly")
	}
}

func TestCopyn(t *testing.T) {
	rb := new(Buffer);
	wb := new(Buffer);
	rb.WriteString("hello, world.");
	Copyn(wb, rb, 5);
	if wb.String() != "hello" {
		t.Errorf("Copyn did not work properly")
	}
}

func TestCopynReadFrom(t *testing.T) {
	rb := new(Buffer);
	wb := new(bytes.Buffer);	// implements ReadFrom.
	rb.WriteString("hello");
	Copyn(wb, rb, 5);
	if wb.String() != "hello" {
		t.Errorf("Copyn did not work properly")
	}
}

func TestCopynWriteTo(t *testing.T) {
	rb := new(bytes.Buffer);	// implements WriteTo.
	wb := new(Buffer);
	rb.WriteString("hello, world.");
	Copyn(wb, rb, 5);
	if wb.String() != "hello" {
		t.Errorf("Copyn did not work properly")
	}
}

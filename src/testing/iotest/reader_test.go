// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iotest

import (
	"bytes"
	"io"
	"testing"
)

func TestOneByteReader_nonEmptyReader(t *testing.T) {
	msg := "Hello, World!"
	buf := new(bytes.Buffer)
	buf.WriteString(msg)

	obr := OneByteReader(buf)
	var b []byte
	n, err := obr.Read(b)
	if err != nil || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}

	b = make([]byte, 3)
	// Read from obr until EOF.
	got := new(bytes.Buffer)
	for i := 0; ; i++ {
		n, err = obr.Read(b)
		if err != nil {
			break
		}
		if g, w := n, 1; g != w {
			t.Errorf("Iteration #%d read %d bytes, want %d", i, g, w)
		}
		got.Write(b[:n])
	}
	if g, w := err, io.EOF; g != w {
		t.Errorf("Unexpected error after reading all bytes\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := got.String(), "Hello, World!"; g != w {
		t.Errorf("Read mismatch\n\tGot:  %q\n\tWant: %q", g, w)
	}
}

func TestOneByteReader_emptyReader(t *testing.T) {
	r := new(bytes.Buffer)

	obr := OneByteReader(r)
	var b []byte
	if n, err := obr.Read(b); err != nil || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}

	b = make([]byte, 5)
	n, err := obr.Read(b)
	if g, w := err, io.EOF; g != w {
		t.Errorf("Error mismatch\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := n, 0; g != w {
		t.Errorf("Unexpectedly read %d bytes, wanted %d", g, w)
	}
}

func TestHalfReader_nonEmptyReader(t *testing.T) {
	msg := "Hello, World!"
	buf := new(bytes.Buffer)
	buf.WriteString(msg)
	// empty read buffer
	hr := HalfReader(buf)
	var b []byte
	n, err := hr.Read(b)
	if err != nil || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}
	// non empty read buffer
	b = make([]byte, 2)
	got := new(bytes.Buffer)
	for i := 0; ; i++ {
		n, err = hr.Read(b)
		if err != nil {
			break
		}
		if g, w := n, 1; g != w {
			t.Errorf("Iteration #%d read %d bytes, want %d", i, g, w)
		}
		got.Write(b[:n])
	}
	if g, w := err, io.EOF; g != w {
		t.Errorf("Unexpected error after reading all bytes\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := got.String(), "Hello, World!"; g != w {
		t.Errorf("Read mismatch\n\tGot:  %q\n\tWant: %q", g, w)
	}
}

func TestHalfReader_emptyReader(t *testing.T) {
	r := new(bytes.Buffer)

	hr := HalfReader(r)
	var b []byte
	if n, err := hr.Read(b); err != nil || n != 0 {
		t.Errorf("Empty buffer read returned n=%d err=%v", n, err)
	}

	b = make([]byte, 5)
	n, err := hr.Read(b)
	if g, w := err, io.EOF; g != w {
		t.Errorf("Error mismatch\n\tGot:  %v\n\tWant: %v", g, w)
	}
	if g, w := n, 0; g != w {
		t.Errorf("Unexpectedly read %d bytes, wanted %d", g, w)
	}
}

func TestTimeoutReader(t *testing.T) {
	data := []byte("hello, world")
	r := TimeoutReader(bytes.NewReader(data))
	p := make([]byte, 2)

	n, err := r.Read(p)
	if err != nil {
		t.Error(err)
	}
	if n != 2 {
		t.Errorf("read %d, but expected to have read %d bytes", n, 2)
	}

	n, err = r.Read(p)
	if err != ErrTimeout {
		t.Errorf("got %v, but second call to Read should return %v", err, ErrTimeout)
	}
	if n != 0 {
		t.Errorf("read %d, but expected to have read %d bytes", n, 0)
	}

	n, err = r.Read(p)
	if err != nil {
		t.Errorf("got %v, but subsequent call to Read should succeed.", err)
	}
	if n != 2 {
		t.Errorf("read %d, but expected to have read %d bytes", n, 2)
	}
}

var dataErrReaderTests = []struct {
	data []byte
	p    []byte
	n    int
}{
	{[]byte("hello"), []byte("o"), 1},
	{[]byte("abcdef"), []byte("ef"), 2},
}

func TestDataErrReader(t *testing.T) {
	for _, tt := range dataErrReaderTests {
		var n int
		var err error
		p := make([]byte, 2)
		r := DataErrReader(bytes.NewReader(tt.data))
		for {
			n, err = r.Read(p)
			if err == io.EOF {
				break
			}
		}
		if n != tt.n {
			t.Errorf("Last call to Read should have read %d bytes instead of %d", n, tt.n)
		}
		if !bytes.Equal(p[:n], tt.p) {
			t.Errorf("got %q, expected %q ", p[:n], tt.p)
		}
	}
}

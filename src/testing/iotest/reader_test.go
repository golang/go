// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iotest

import (
	"bytes"
	"io"
	"reflect"
	"testing"
)

var oneByteReaderTests = []struct {
	data []byte
	p    []byte
	n    int
}{
	{[]byte(""), []byte(""), 0},
	{[]byte("h"), []byte("h"), 1},
	{[]byte("hello, world"), []byte("h"), 1},
}

func TestOneByteReader(t *testing.T) {
	for _, tt := range oneByteReaderTests {
		r := OneByteReader(bytes.NewReader(tt.data))
		p := make([]byte, len(tt.data))
		n, err := r.Read(p)
		if err != nil {
			t.Error(err)
		}
		if n != tt.n {
			t.Errorf("expected to have read %d bytes, but read %d", tt.n, n)
		}
		got := p[:n]
		want := tt.p
		if reflect.DeepEqual(got, want) != true {
			t.Errorf("expected %v, got %v", want, got)
		}
	}
}

var halfReaderTests = []struct {
	data []byte
	p    []byte
	n    int
}{
	{[]byte("h"), []byte("h"), 1},
	{[]byte("hello, world"), []byte("hello,"), 6},
}

func TestHalfReader(t *testing.T) {
	for _, tt := range halfReaderTests {
		r := HalfReader(bytes.NewReader(tt.data))
		p := make([]byte, len(tt.data))
		n, err := r.Read(p)
		if err != nil {
			t.Error(err)
		}
		if n != tt.n {
			t.Errorf("expected to have read %d bytes, but read %d", tt.n, n)
		}
		got := p[:n]
		want := tt.p
		if reflect.DeepEqual(got, want) != true {
			t.Errorf("expected %v, got %v", want, got)
		}
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
		t.Errorf("expected to have read %d bytes, but read %d", 2, n)
	}

	n, err = r.Read(p)
	if err != ErrTimeout {
		t.Errorf("Second call to Read should return %v, got %v", ErrTimeout, err)
	}
	if n != 0 {
		t.Errorf("expected to have read %d bytes, but read %d", 0, n)
	}

	n, err = r.Read(p)
	if err != nil {
		t.Errorf("Subsequent call to read succeed. Got %v", err)
	}
	if n != 2 {
		t.Errorf("expected to have read %d bytes, but read %d", 2, n)
	}
}

var dataErrReaderTests = []struct {
	data []byte
	p    []byte
	n    int
}{
	{[]byte("hello"), []byte("o"), 1},
	{[]byte("hello,"), []byte("o,"), 2},
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
		if reflect.DeepEqual(p[:n], tt.p) != true {
			t.Errorf("Wanted %v got %v", tt.p, p[:n])
		}
	}
}

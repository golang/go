// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iotest

import (
	"bytes"
	"io"
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
			t.Errorf("read %d, but expected to have read %d bytes", n, tt.n)
		}
		got := p[:n]
		want := tt.p
		if !bytes.Equal(got, want) {
			t.Errorf("got %q, expected %q", got, want)
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
			t.Errorf("read %d, but expected to have read %d bytes", n, tt.n)
		}
		got := p[:n]
		want := tt.p
		if !bytes.Equal(got, want) {
			t.Errorf("got %q, expected %q", got, want)
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

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gzip0

import (
	"bytes"
	"compress/gzip"
	"io/ioutil"
	"testing"
)

func TestWriter(t *testing.T) {
	testWriter(t, nil)
	testWriter(t, []byte("hello world"))
	testWriter(t, make([]byte, 100000))
	testWriter(t, make([]byte, 65536))
	testWriter(t, make([]byte, 65535))
}

func testWriter(t *testing.T, data []byte) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	n, err := w.Write(data)
	if n != len(data) || err != nil {
		t.Fatalf("Write: %d, %v", n, err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	r, err := gzip.NewReader(&buf)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	data1, err := ioutil.ReadAll(r)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if !bytes.Equal(data, data1) {
		t.Fatalf("not equal: %q != %q", data, data1)
	}
}

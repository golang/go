// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bytes"
	"io/ioutil"
	"rand"
	"testing"
)

// TODO(adg): a more sophisticated test suite

const testString = "Rabbits, guinea pigs, gophers, marsupial rats, and quolls."

func TestWriter(t *testing.T) {
	largeData := make([]byte, 1<<17)
	for i := range largeData {
		largeData[i] = byte(rand.Int())
	}

	// write a zip file
	buf := new(bytes.Buffer)
	w := NewWriter(buf)
	testCreate(t, w, "foo", []byte(testString), Store)
	testCreate(t, w, "bar", largeData, Deflate)
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// read it back
	r, err := NewReader(sliceReaderAt(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	testReadFile(t, r.File[0], []byte(testString))
	testReadFile(t, r.File[1], largeData)
}

func testCreate(t *testing.T, w *Writer, name string, data []byte, method uint16) {
	header := &FileHeader{
		Name:   name,
		Method: method,
	}
	f, err := w.CreateHeader(header)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.Write(data)
	if err != nil {
		t.Fatal(err)
	}
}

func testReadFile(t *testing.T, f *File, data []byte) {
	rc, err := f.Open()
	if err != nil {
		t.Fatal("opening:", err)
	}
	b, err := ioutil.ReadAll(rc)
	if err != nil {
		t.Fatal("reading:", err)
	}
	err = rc.Close()
	if err != nil {
		t.Fatal("closing:", err)
	}
	if !bytes.Equal(b, data) {
		t.Errorf("File contents %q, want %q", b, data)
	}
}

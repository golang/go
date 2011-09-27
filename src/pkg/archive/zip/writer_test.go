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

type WriteTest struct {
	Name   string
	Data   []byte
	Method uint16
	Mode   uint32
}

var writeTests = []WriteTest{
	WriteTest{
		Name:   "foo",
		Data:   []byte("Rabbits, guinea pigs, gophers, marsupial rats, and quolls."),
		Method: Store,
	},
	WriteTest{
		Name:   "bar",
		Data:   nil, // large data set in the test
		Method: Deflate,
		Mode:   0x81ed,
	},
}

func TestWriter(t *testing.T) {
	largeData := make([]byte, 1<<17)
	for i := range largeData {
		largeData[i] = byte(rand.Int())
	}
	writeTests[1].Data = largeData
	defer func() {
		writeTests[1].Data = nil
	}()

	// write a zip file
	buf := new(bytes.Buffer)
	w := NewWriter(buf)

	for _, wt := range writeTests {
		testCreate(t, w, &wt)
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// read it back
	r, err := NewReader(sliceReaderAt(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	for i, wt := range writeTests {
		testReadFile(t, r.File[i], &wt)
	}
}

func testCreate(t *testing.T, w *Writer, wt *WriteTest) {
	header := &FileHeader{
		Name:   wt.Name,
		Method: wt.Method,
	}
	if wt.Mode != 0 {
		header.SetMode(wt.Mode)
	}
	f, err := w.CreateHeader(header)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.Write(wt.Data)
	if err != nil {
		t.Fatal(err)
	}
}

func testReadFile(t *testing.T, f *File, wt *WriteTest) {
	if f.Name != wt.Name {
		t.Fatalf("File name: got %q, want %q", f.Name, wt.Name)
	}
	testFileMode(t, f, wt.Mode)
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
	if !bytes.Equal(b, wt.Data) {
		t.Errorf("File contents %q, want %q", b, wt.Data)
	}
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bytes"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"testing"
	"time"
)

// TODO(adg): a more sophisticated test suite

type WriteTest struct {
	Name   string
	Data   []byte
	Method uint16
	Mode   os.FileMode
	Mtime  string
}

var writeTests = []WriteTest{
	{
		Name:   "foo",
		Data:   []byte("Rabbits, guinea pigs, gophers, marsupial rats, and quolls."),
		Method: Store,
		Mode:   0666,
		Mtime:  "02-01-08 00:01:02",
	},
	{
		Name:   "bar",
		Data:   nil, // large data set in the test
		Method: Deflate,
		Mode:   0644,
		Mtime:  "03-02-08 01:02:03",
	},
	{
		Name:   "setuid",
		Data:   []byte("setuid file"),
		Method: Deflate,
		Mode:   0755 | os.ModeSetuid,
		Mtime:  "04-03-08 02:03:04",
	},
	{
		Name:   "setgid",
		Data:   []byte("setgid file"),
		Method: Deflate,
		Mode:   0755 | os.ModeSetgid,
		Mtime:  "05-04-08 03:04:04",
	},
	{
		Name:   "symlink",
		Data:   []byte("../link/target"),
		Method: Deflate,
		Mode:   0755 | os.ModeSymlink,
		Mtime:  "03-02-08 11:22:33",
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
	r, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	for i, wt := range writeTests {
		testReadFile(t, r.File[i], &wt)
	}
}

func TestWriterOffset(t *testing.T) {
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
	existingData := []byte{1, 2, 3, 1, 2, 3, 1, 2, 3}
	n, _ := buf.Write(existingData)
	w := NewWriter(buf)
	w.SetOffset(int64(n))

	for _, wt := range writeTests {
		testCreate(t, w, &wt)
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// read it back
	r, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	for i, wt := range writeTests {
		testReadFile(t, r.File[i], &wt)
	}
}

func TestWriterFlush(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(struct{ io.Writer }{&buf})
	_, err := w.Create("foo")
	if err != nil {
		t.Fatal(err)
	}
	if buf.Len() > 0 {
		t.Fatalf("Unexpected %d bytes already in buffer", buf.Len())
	}
	if err := w.Flush(); err != nil {
		t.Fatal(err)
	}
	if buf.Len() == 0 {
		t.Fatal("No bytes written after Flush")
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
	mtime, err := time.Parse("01-02-06 15:04:05", wt.Mtime)
	if err != nil {
		t.Fatal("time.Parse:", err)
	}
	header.SetModTime(mtime)
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
	testFileMode(t, wt.Name, f, wt.Mode)
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

	mtime, err := time.Parse("01-02-06 15:04:05", wt.Mtime)
	if err != nil {
		t.Fatal("time.Parse:", err)
	}

	diff := mtime.Sub(f.ModTime())
	if diff < 0 {
		diff = -diff
	}

	// allow several time span
	if diff > 5*time.Second {
		t.Errorf("File modtime %v, want %v", mtime, f.ModTime())
	}
}

func BenchmarkCompressedZipGarbage(b *testing.B) {
	b.ReportAllocs()
	var buf bytes.Buffer
	bigBuf := bytes.Repeat([]byte("a"), 1<<20)
	for i := 0; i <= b.N; i++ {
		buf.Reset()
		zw := NewWriter(&buf)
		for j := 0; j < 3; j++ {
			w, _ := zw.CreateHeader(&FileHeader{
				Name:   "foo",
				Method: Deflate,
			})
			w.Write(bigBuf)
		}
		zw.Close()
		if i == 0 {
			// Reset the timer after the first time through.
			// This effectively discards the very large initial flate setup cost,
			// as well as the initialization of bigBuf.
			b.ResetTimer()
		}
	}
}

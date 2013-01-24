// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that involve both reading and writing.

package zip

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"testing"
	"time"
)

func TestOver65kFiles(t *testing.T) {
	if testing.Short() {
		t.Skip("slow test; skipping")
	}
	buf := new(bytes.Buffer)
	w := NewWriter(buf)
	const nFiles = (1 << 16) + 42
	for i := 0; i < nFiles; i++ {
		_, err := w.Create(fmt.Sprintf("%d.dat", i))
		if err != nil {
			t.Fatalf("creating file %d: %v", i, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Writer.Close: %v", err)
	}
	s := buf.String()
	zr, err := NewReader(strings.NewReader(s), int64(len(s)))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if got := len(zr.File); got != nFiles {
		t.Fatalf("File contains %d files, want %d", got, nFiles)
	}
	for i := 0; i < nFiles; i++ {
		want := fmt.Sprintf("%d.dat", i)
		if zr.File[i].Name != want {
			t.Fatalf("File(%d) = %q, want %q", i, zr.File[i].Name, want)
		}
	}
}

func TestModTime(t *testing.T) {
	var testTime = time.Date(2009, time.November, 10, 23, 45, 58, 0, time.UTC)
	fh := new(FileHeader)
	fh.SetModTime(testTime)
	outTime := fh.ModTime()
	if !outTime.Equal(testTime) {
		t.Errorf("times don't match: got %s, want %s", outTime, testTime)
	}
}

func testHeaderRoundTrip(fh *FileHeader, wantUncompressedSize uint32, wantUncompressedSize64 uint64, t *testing.T) {
	fi := fh.FileInfo()
	fh2, err := FileInfoHeader(fi)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := fh2.Name, fh.Name; got != want {
		t.Errorf("Name: got %s, want %s\n", got, want)
	}
	if got, want := fh2.UncompressedSize, wantUncompressedSize; got != want {
		t.Errorf("UncompressedSize: got %d, want %d\n", got, want)
	}
	if got, want := fh2.UncompressedSize64, wantUncompressedSize64; got != want {
		t.Errorf("UncompressedSize64: got %d, want %d\n", got, want)
	}
	if got, want := fh2.ModifiedTime, fh.ModifiedTime; got != want {
		t.Errorf("ModifiedTime: got %d, want %d\n", got, want)
	}
	if got, want := fh2.ModifiedDate, fh.ModifiedDate; got != want {
		t.Errorf("ModifiedDate: got %d, want %d\n", got, want)
	}

	if sysfh, ok := fi.Sys().(*FileHeader); !ok && sysfh != fh {
		t.Errorf("Sys didn't return original *FileHeader")
	}
}

func TestFileHeaderRoundTrip(t *testing.T) {
	fh := &FileHeader{
		Name:             "foo.txt",
		UncompressedSize: 987654321,
		ModifiedTime:     1234,
		ModifiedDate:     5678,
	}
	testHeaderRoundTrip(fh, fh.UncompressedSize, uint64(fh.UncompressedSize), t)
}

func TestFileHeaderRoundTrip64(t *testing.T) {
	fh := &FileHeader{
		Name:               "foo.txt",
		UncompressedSize64: 9876543210,
		ModifiedTime:       1234,
		ModifiedDate:       5678,
	}
	testHeaderRoundTrip(fh, uint32max, fh.UncompressedSize64, t)
}

func TestZip64(t *testing.T) {
	if testing.Short() {
		t.Skip("slow test; skipping")
	}
	// write 2^32 bytes plus "END\n" to a zip file
	buf := new(bytes.Buffer)
	w := NewWriter(buf)
	f, err := w.Create("huge.txt")
	if err != nil {
		t.Fatal(err)
	}
	chunk := make([]byte, 1024)
	for i := range chunk {
		chunk[i] = '.'
	}
	chunk[len(chunk)-1] = '\n'
	end := []byte("END\n")
	for i := 0; i < (1<<32)/1024; i++ {
		_, err := f.Write(chunk)
		if err != nil {
			t.Fatal("write chunk:", err)
		}
	}
	_, err = f.Write(end)
	if err != nil {
		t.Fatal("write end:", err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// read back zip file and check that we get to the end of it
	r, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal("reader:", err)
	}
	f0 := r.File[0]
	rc, err := f0.Open()
	if err != nil {
		t.Fatal("opening:", err)
	}
	for i := 0; i < (1<<32)/1024; i++ {
		_, err := io.ReadFull(rc, chunk)
		if err != nil {
			t.Fatal("read:", err)
		}
	}
	gotEnd, err := ioutil.ReadAll(rc)
	if err != nil {
		t.Fatal("read end:", err)
	}
	if !bytes.Equal(gotEnd, end) {
		t.Errorf("End of zip64 archive %q, want %q", gotEnd, end)
	}
	err = rc.Close()
	if err != nil {
		t.Fatal("closing:", err)
	}
	if got, want := f0.UncompressedSize, uint32(uint32max); got != want {
		t.Errorf("UncompressedSize %d, want %d", got, want)
	}

	if got, want := f0.UncompressedSize64, (1<<32)+uint64(len(end)); got != want {
		t.Errorf("UncompressedSize64 %d, want %d", got, want)
	}
}

func testInvalidHeader(h *FileHeader, t *testing.T) {
	var buf bytes.Buffer
	z := NewWriter(&buf)

	f, err := z.CreateHeader(h)
	if err != nil {
		t.Fatalf("error creating header: %v", err)
	}
	if _, err := f.Write([]byte("hi")); err != nil {
		t.Fatalf("error writing content: %v", err)
	}
	if err := z.Close(); err != nil {
		t.Fatalf("error closing zip writer: %v", err)
	}

	b := buf.Bytes()
	if _, err = NewReader(bytes.NewReader(b), int64(len(b))); err != ErrFormat {
		t.Fatalf("got %v, expected ErrFormat", err)
	}
}

func testValidHeader(h *FileHeader, t *testing.T) {
	var buf bytes.Buffer
	z := NewWriter(&buf)

	f, err := z.CreateHeader(h)
	if err != nil {
		t.Fatalf("error creating header: %v", err)
	}
	if _, err := f.Write([]byte("hi")); err != nil {
		t.Fatalf("error writing content: %v", err)
	}
	if err := z.Close(); err != nil {
		t.Fatalf("error closing zip writer: %v", err)
	}

	b := buf.Bytes()
	if _, err = NewReader(bytes.NewReader(b), int64(len(b))); err != nil {
		t.Fatalf("got %v, expected nil", err)
	}
}

// Issue 4302.
func TestHeaderInvalidTagAndSize(t *testing.T) {
	const timeFormat = "20060102T150405.000.txt"

	ts := time.Now()
	filename := ts.Format(timeFormat)

	h := FileHeader{
		Name:   filename,
		Method: Deflate,
		Extra:  []byte(ts.Format(time.RFC3339Nano)), // missing tag and len
	}
	h.SetModTime(ts)

	testInvalidHeader(&h, t)
}

func TestHeaderTooShort(t *testing.T) {
	h := FileHeader{
		Name:   "foo.txt",
		Method: Deflate,
		Extra:  []byte{zip64ExtraId}, // missing size
	}
	testInvalidHeader(&h, t)
}

// Issue 4393. It is valid to have an extra data header
// which contains no body.
func TestZeroLengthHeader(t *testing.T) {
	h := FileHeader{
		Name:   "extadata.txt",
		Method: Deflate,
		Extra: []byte{
			85, 84, 5, 0, 3, 154, 144, 195, 77, // tag 21589 size 5
			85, 120, 0, 0, // tag 30805 size 0
		},
	}
	testValidHeader(&h, t)
}

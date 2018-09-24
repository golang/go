// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"
)

// TODO(adg): a more sophisticated test suite

type WriteTest struct {
	Name   string
	Data   []byte
	Method uint16
	Mode   os.FileMode
}

var writeTests = []WriteTest{
	{
		Name:   "foo",
		Data:   []byte("Rabbits, guinea pigs, gophers, marsupial rats, and quolls."),
		Method: Store,
		Mode:   0666,
	},
	{
		Name:   "bar",
		Data:   nil, // large data set in the test
		Method: Deflate,
		Mode:   0644,
	},
	{
		Name:   "setuid",
		Data:   []byte("setuid file"),
		Method: Deflate,
		Mode:   0755 | os.ModeSetuid,
	},
	{
		Name:   "setgid",
		Data:   []byte("setgid file"),
		Method: Deflate,
		Mode:   0755 | os.ModeSetgid,
	},
	{
		Name:   "symlink",
		Data:   []byte("../link/target"),
		Method: Deflate,
		Mode:   0755 | os.ModeSymlink,
	},
}

func TestWriter(t *testing.T) {
	largeData := make([]byte, 1<<17)
	if _, err := rand.Read(largeData); err != nil {
		t.Fatal("rand.Read failed:", err)
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

// TestWriterComment is test for EOCD comment read/write.
func TestWriterComment(t *testing.T) {
	var tests = []struct {
		comment string
		ok      bool
	}{
		{"hi, hello", true},
		{"hi, こんにちわ", true},
		{strings.Repeat("a", uint16max), true},
		{strings.Repeat("a", uint16max+1), false},
	}

	for _, test := range tests {
		// write a zip file
		buf := new(bytes.Buffer)
		w := NewWriter(buf)
		if err := w.SetComment(test.comment); err != nil {
			if test.ok {
				t.Fatalf("SetComment: unexpected error %v", err)
			}
			continue
		} else {
			if !test.ok {
				t.Fatalf("SetComment: unexpected success, want error")
			}
		}

		if err := w.Close(); test.ok == (err != nil) {
			t.Fatal(err)
		}

		if w.closed != test.ok {
			t.Fatalf("Writer.closed: got %v, want %v", w.closed, test.ok)
		}

		// skip read test in failure cases
		if !test.ok {
			continue
		}

		// read it back
		r, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
		if err != nil {
			t.Fatal(err)
		}
		if r.Comment != test.comment {
			t.Fatalf("Reader.Comment: got %v, want %v", r.Comment, test.comment)
		}
	}
}

func TestWriterUTF8(t *testing.T) {
	var utf8Tests = []struct {
		name    string
		comment string
		nonUTF8 bool
		flags   uint16
	}{
		{
			name:    "hi, hello",
			comment: "in the world",
			flags:   0x8,
		},
		{
			name:    "hi, こんにちわ",
			comment: "in the world",
			flags:   0x808,
		},
		{
			name:    "hi, こんにちわ",
			comment: "in the world",
			nonUTF8: true,
			flags:   0x8,
		},
		{
			name:    "hi, hello",
			comment: "in the 世界",
			flags:   0x808,
		},
		{
			name:    "hi, こんにちわ",
			comment: "in the 世界",
			flags:   0x808,
		},
		{
			name:    "the replacement rune is �",
			comment: "the replacement rune is �",
			flags:   0x808,
		},
		{
			// Name is Japanese encoded in Shift JIS.
			name:    "\x93\xfa\x96{\x8c\xea.txt",
			comment: "in the 世界",
			flags:   0x008, // UTF-8 must not be set
		},
	}

	// write a zip file
	buf := new(bytes.Buffer)
	w := NewWriter(buf)

	for _, test := range utf8Tests {
		h := &FileHeader{
			Name:    test.name,
			Comment: test.comment,
			NonUTF8: test.nonUTF8,
			Method:  Deflate,
		}
		w, err := w.CreateHeader(h)
		if err != nil {
			t.Fatal(err)
		}
		w.Write([]byte{})
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// read it back
	r, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	for i, test := range utf8Tests {
		flags := r.File[i].Flags
		if flags != test.flags {
			t.Errorf("CreateHeader(name=%q comment=%q nonUTF8=%v): flags=%#x, want %#x", test.name, test.comment, test.nonUTF8, flags, test.flags)
		}
	}
}

func TestWriterTime(t *testing.T) {
	var buf bytes.Buffer
	h := &FileHeader{
		Name:     "test.txt",
		Modified: time.Date(2017, 10, 31, 21, 11, 57, 0, timeZone(-7*time.Hour)),
	}
	w := NewWriter(&buf)
	if _, err := w.CreateHeader(h); err != nil {
		t.Fatalf("unexpected CreateHeader error: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("unexpected Close error: %v", err)
	}

	want, err := ioutil.ReadFile("testdata/time-go.zip")
	if err != nil {
		t.Fatalf("unexpected ReadFile error: %v", err)
	}
	if got := buf.Bytes(); !bytes.Equal(got, want) {
		fmt.Printf("%x\n%x\n", got, want)
		t.Error("contents of time-go.zip differ")
	}
}

func TestWriterOffset(t *testing.T) {
	largeData := make([]byte, 1<<17)
	if _, err := rand.Read(largeData); err != nil {
		t.Fatal("rand.Read failed:", err)
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

func TestWriterDir(t *testing.T) {
	w := NewWriter(ioutil.Discard)
	dw, err := w.Create("dir/")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := dw.Write(nil); err != nil {
		t.Errorf("Write(nil) to directory: got %v, want nil", err)
	}
	if _, err := dw.Write([]byte("hello")); err == nil {
		t.Error(`Write("hello") to directory: got nil error, want non-nil`)
	}
}

func TestWriterDirAttributes(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	if _, err := w.CreateHeader(&FileHeader{
		Name:               "dir/",
		Method:             Deflate,
		CompressedSize64:   1234,
		UncompressedSize64: 5678,
	}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()

	var sig [4]byte
	binary.LittleEndian.PutUint32(sig[:], uint32(fileHeaderSignature))

	idx := bytes.Index(b, sig[:])
	if idx == -1 {
		t.Fatal("file header not found")
	}
	b = b[idx:]

	if !bytes.Equal(b[6:10], []byte{0, 0, 0, 0}) { // FileHeader.Flags: 0, FileHeader.Method: 0
		t.Errorf("unexpected method and flags: %v", b[6:10])
	}

	if !bytes.Equal(b[14:26], make([]byte, 12)) { // FileHeader.{CRC32,CompressSize,UncompressedSize} all zero.
		t.Errorf("unexpected crc, compress and uncompressed size to be 0 was: %v", b[14:26])
	}

	binary.LittleEndian.PutUint32(sig[:], uint32(dataDescriptorSignature))
	if bytes.Index(b, sig[:]) != -1 {
		t.Error("there should be no data descriptor")
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

func BenchmarkCompressedZipGarbage(b *testing.B) {
	bigBuf := bytes.Repeat([]byte("a"), 1<<20)

	runOnce := func(buf *bytes.Buffer) {
		buf.Reset()
		zw := NewWriter(buf)
		for j := 0; j < 3; j++ {
			w, _ := zw.CreateHeader(&FileHeader{
				Name:   "foo",
				Method: Deflate,
			})
			w.Write(bigBuf)
		}
		zw.Close()
	}

	b.ReportAllocs()
	// Run once and then reset the timer.
	// This effectively discards the very large initial flate setup cost,
	// as well as the initialization of bigBuf.
	runOnce(&bytes.Buffer{})
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		var buf bytes.Buffer
		for pb.Next() {
			runOnce(&buf)
		}
	})
}

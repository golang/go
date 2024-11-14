// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"io/fs"
	"math/rand"
	"os"
	"strings"
	"testing"
	"testing/fstest"
	"time"
)

// TODO(adg): a more sophisticated test suite

type WriteTest struct {
	Name   string
	Data   []byte
	Method uint16
	Mode   fs.FileMode
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
		Mode:   0755 | fs.ModeSetuid,
	},
	{
		Name:   "setgid",
		Data:   []byte("setgid file"),
		Method: Deflate,
		Mode:   0755 | fs.ModeSetgid,
	},
	{
		Name:   "symlink",
		Data:   []byte("../link/target"),
		Method: Deflate,
		Mode:   0755 | fs.ModeSymlink,
	},
	{
		Name:   "device",
		Data:   []byte("device file"),
		Method: Deflate,
		Mode:   0755 | fs.ModeDevice,
	},
	{
		Name:   "chardevice",
		Data:   []byte("char device file"),
		Method: Deflate,
		Mode:   0755 | fs.ModeDevice | fs.ModeCharDevice,
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

	want, err := os.ReadFile("testdata/time-go.zip")
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
	w := NewWriter(io.Discard)
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
	if bytes.Contains(b, sig[:]) {
		t.Error("there should be no data descriptor")
	}
}

func TestWriterCopy(t *testing.T) {
	// make a zip file
	buf := new(bytes.Buffer)
	w := NewWriter(buf)
	for _, wt := range writeTests {
		testCreate(t, w, &wt)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// read it back
	src, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	for i, wt := range writeTests {
		testReadFile(t, src.File[i], &wt)
	}

	// make a new zip file copying the old compressed data.
	buf2 := new(bytes.Buffer)
	dst := NewWriter(buf2)
	for _, f := range src.File {
		if err := dst.Copy(f); err != nil {
			t.Fatal(err)
		}
	}
	if err := dst.Close(); err != nil {
		t.Fatal(err)
	}

	// read the new one back
	r, err := NewReader(bytes.NewReader(buf2.Bytes()), int64(buf2.Len()))
	if err != nil {
		t.Fatal(err)
	}
	for i, wt := range writeTests {
		testReadFile(t, r.File[i], &wt)
	}
}

func TestWriterCreateRaw(t *testing.T) {
	files := []struct {
		name             string
		content          []byte
		method           uint16
		flags            uint16
		crc32            uint32
		uncompressedSize uint64
		compressedSize   uint64
	}{
		{
			name:    "small store w desc",
			content: []byte("gophers"),
			method:  Store,
			flags:   0x8,
		},
		{
			name:    "small deflate wo desc",
			content: bytes.Repeat([]byte("abcdefg"), 2048),
			method:  Deflate,
		},
	}

	// write a zip file
	archive := new(bytes.Buffer)
	w := NewWriter(archive)

	for i := range files {
		f := &files[i]
		f.crc32 = crc32.ChecksumIEEE(f.content)
		size := uint64(len(f.content))
		f.uncompressedSize = size
		f.compressedSize = size

		var compressedContent []byte
		if f.method == Deflate {
			var buf bytes.Buffer
			w, err := flate.NewWriter(&buf, flate.BestSpeed)
			if err != nil {
				t.Fatalf("flate.NewWriter err = %v", err)
			}
			_, err = w.Write(f.content)
			if err != nil {
				t.Fatalf("flate Write err = %v", err)
			}
			err = w.Close()
			if err != nil {
				t.Fatalf("flate Writer.Close err = %v", err)
			}
			compressedContent = buf.Bytes()
			f.compressedSize = uint64(len(compressedContent))
		}

		h := &FileHeader{
			Name:               f.name,
			Method:             f.method,
			Flags:              f.flags,
			CRC32:              f.crc32,
			CompressedSize64:   f.compressedSize,
			UncompressedSize64: f.uncompressedSize,
		}
		w, err := w.CreateRaw(h)
		if err != nil {
			t.Fatal(err)
		}
		if compressedContent != nil {
			_, err = w.Write(compressedContent)
		} else {
			_, err = w.Write(f.content)
		}
		if err != nil {
			t.Fatalf("%s Write got %v; want nil", f.name, err)
		}
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// read it back
	r, err := NewReader(bytes.NewReader(archive.Bytes()), int64(archive.Len()))
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range files {
		got := r.File[i]
		if got.Name != want.name {
			t.Errorf("got Name %s; want %s", got.Name, want.name)
		}
		if got.Method != want.method {
			t.Errorf("%s: got Method %#x; want %#x", want.name, got.Method, want.method)
		}
		if got.Flags != want.flags {
			t.Errorf("%s: got Flags %#x; want %#x", want.name, got.Flags, want.flags)
		}
		if got.CRC32 != want.crc32 {
			t.Errorf("%s: got CRC32 %#x; want %#x", want.name, got.CRC32, want.crc32)
		}
		if got.CompressedSize64 != want.compressedSize {
			t.Errorf("%s: got CompressedSize64 %d; want %d", want.name, got.CompressedSize64, want.compressedSize)
		}
		if got.UncompressedSize64 != want.uncompressedSize {
			t.Errorf("%s: got UncompressedSize64 %d; want %d", want.name, got.UncompressedSize64, want.uncompressedSize)
		}

		r, err := got.Open()
		if err != nil {
			t.Errorf("%s: Open err = %v", got.Name, err)
			continue
		}

		buf, err := io.ReadAll(r)
		if err != nil {
			t.Errorf("%s: ReadAll err = %v", got.Name, err)
			continue
		}

		if !bytes.Equal(buf, want.content) {
			t.Errorf("%v: ReadAll returned unexpected bytes", got.Name)
		}
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
		t.Fatalf("opening %s: %v", f.Name, err)
	}
	b, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("reading %s: %v", f.Name, err)
	}
	err = rc.Close()
	if err != nil {
		t.Fatalf("closing %s: %v", f.Name, err)
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

	b.RunParallel(func { pb ->
		var buf bytes.Buffer
		for pb.Next() {
			runOnce(&buf)
		}
	})
}

func writeTestsToFS(tests []WriteTest) fs.FS {
	fsys := fstest.MapFS{}
	for _, wt := range tests {
		fsys[wt.Name] = &fstest.MapFile{
			Data: wt.Data,
			Mode: wt.Mode,
		}
	}
	return fsys
}

func TestWriterAddFS(t *testing.T) {
	buf := new(bytes.Buffer)
	w := NewWriter(buf)
	tests := []WriteTest{
		{
			Name: "file.go",
			Data: []byte("hello"),
			Mode: 0644,
		},
		{
			Name: "subfolder/another.go",
			Data: []byte("world"),
			Mode: 0644,
		},
	}
	err := w.AddFS(writeTestsToFS(tests))
	if err != nil {
		t.Fatal(err)
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// read it back
	r, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	for i, wt := range tests {
		testReadFile(t, r.File[i], &wt)
	}
}

func TestIssue61875(t *testing.T) {
	buf := new(bytes.Buffer)
	w := NewWriter(buf)
	tests := []WriteTest{
		{
			Name:   "symlink",
			Data:   []byte("../link/target"),
			Method: Deflate,
			Mode:   0755 | fs.ModeSymlink,
		},
		{
			Name:   "device",
			Data:   []byte(""),
			Method: Deflate,
			Mode:   0755 | fs.ModeDevice,
		},
	}
	err := w.AddFS(writeTestsToFS(tests))
	if err == nil {
		t.Errorf("expected error, got nil")
	}
}

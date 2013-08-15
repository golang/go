// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that involve both reading and writing.

package zip

import (
	"bytes"
	"fmt"
	"hash"
	"io"
	"io/ioutil"
	"sort"
	"strings"
	"testing"
	"time"
)

func TestOver65kFiles(t *testing.T) {
	buf := new(bytes.Buffer)
	w := NewWriter(buf)
	const nFiles = (1 << 16) + 42
	for i := 0; i < nFiles; i++ {
		_, err := w.CreateHeader(&FileHeader{
			Name:   fmt.Sprintf("%d.dat", i),
			Method: Store, // avoid Issue 6136 and Issue 6138
		})
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

type repeatedByte struct {
	off int64
	b   byte
	n   int64
}

// rleBuffer is a run-length-encoded byte buffer.
// It's an io.Writer (like a bytes.Buffer) and also an io.ReaderAt,
// allowing random-access reads.
type rleBuffer struct {
	buf []repeatedByte
}

func (r *rleBuffer) Size() int64 {
	if len(r.buf) == 0 {
		return 0
	}
	last := &r.buf[len(r.buf)-1]
	return last.off + last.n
}

func (r *rleBuffer) Write(p []byte) (n int, err error) {
	var rp *repeatedByte
	if len(r.buf) > 0 {
		rp = &r.buf[len(r.buf)-1]
		// Fast path, if p is entirely the same byte repeated.
		if lastByte := rp.b; len(p) > 0 && p[0] == lastByte {
			all := true
			for _, b := range p {
				if b != lastByte {
					all = false
					break
				}
			}
			if all {
				rp.n += int64(len(p))
				return len(p), nil
			}
		}
	}

	for _, b := range p {
		if rp == nil || rp.b != b {
			r.buf = append(r.buf, repeatedByte{r.Size(), b, 1})
			rp = &r.buf[len(r.buf)-1]
		} else {
			rp.n++
		}
	}
	return len(p), nil
}

func (r *rleBuffer) ReadAt(p []byte, off int64) (n int, err error) {
	if len(p) == 0 {
		return
	}
	skipParts := sort.Search(len(r.buf), func(i int) bool {
		part := &r.buf[i]
		return part.off+part.n > off
	})
	parts := r.buf[skipParts:]
	if len(parts) > 0 {
		skipBytes := off - parts[0].off
		for len(parts) > 0 {
			part := parts[0]
			for i := skipBytes; i < part.n; i++ {
				if n == len(p) {
					return
				}
				p[n] = part.b
				n++
			}
			parts = parts[1:]
			skipBytes = 0
		}
	}
	if n != len(p) {
		err = io.ErrUnexpectedEOF
	}
	return
}

// Just testing the rleBuffer used in the Zip64 test above. Not used by the zip code.
func TestRLEBuffer(t *testing.T) {
	b := new(rleBuffer)
	var all []byte
	writes := []string{"abcdeee", "eeeeeee", "eeeefghaaiii"}
	for _, w := range writes {
		b.Write([]byte(w))
		all = append(all, w...)
	}
	if len(b.buf) != 10 {
		t.Fatalf("len(b.buf) = %d; want 10", len(b.buf))
	}

	for i := 0; i < len(all); i++ {
		for j := 0; j < len(all)-i; j++ {
			buf := make([]byte, j)
			n, err := b.ReadAt(buf, int64(i))
			if err != nil || n != len(buf) {
				t.Errorf("ReadAt(%d, %d) = %d, %v; want %d, nil", i, j, n, err, len(buf))
			}
			if !bytes.Equal(buf, all[i:i+j]) {
				t.Errorf("ReadAt(%d, %d) = %q; want %q", i, j, buf, all[i:i+j])
			}
		}
	}
}

// fakeHash32 is a dummy Hash32 that always returns 0.
type fakeHash32 struct {
	hash.Hash32
}

func (fakeHash32) Write(p []byte) (int, error) { return len(p), nil }
func (fakeHash32) Sum32() uint32               { return 0 }

func TestZip64(t *testing.T) {
	if testing.Short() {
		t.Skip("slow test; skipping")
	}
	const size = 1 << 32 // before the "END\n" part
	testZip64(t, size)
}

func testZip64(t testing.TB, size int64) {
	const chunkSize = 1024
	chunks := int(size / chunkSize)
	// write 2^32 bytes plus "END\n" to a zip file
	buf := new(rleBuffer)
	w := NewWriter(buf)
	f, err := w.CreateHeader(&FileHeader{
		Name:   "huge.txt",
		Method: Store,
	})
	if err != nil {
		t.Fatal(err)
	}
	f.(*fileWriter).crc32 = fakeHash32{}
	chunk := make([]byte, chunkSize)
	for i := range chunk {
		chunk[i] = '.'
	}
	for i := 0; i < chunks; i++ {
		_, err := f.Write(chunk)
		if err != nil {
			t.Fatal("write chunk:", err)
		}
	}
	end := []byte("END\n")
	_, err = f.Write(end)
	if err != nil {
		t.Fatal("write end:", err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// read back zip file and check that we get to the end of it
	r, err := NewReader(buf, int64(buf.Size()))
	if err != nil {
		t.Fatal("reader:", err)
	}
	f0 := r.File[0]
	rc, err := f0.Open()
	if err != nil {
		t.Fatal("opening:", err)
	}
	rc.(*checksumReader).hash = fakeHash32{}
	for i := 0; i < chunks; i++ {
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
	if size == 1<<32 {
		if got, want := f0.UncompressedSize, uint32(uint32max); got != want {
			t.Errorf("UncompressedSize %d, want %d", got, want)
		}
	}

	if got, want := f0.UncompressedSize64, uint64(size)+uint64(len(end)); got != want {
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

// Just benchmarking how fast the Zip64 test above is. Not related to
// our zip performance, since the test above disabled CRC32 and flate.
func BenchmarkZip64Test(b *testing.B) {
	for i := 0; i < b.N; i++ {
		testZip64(b, 1<<26)
	}
}

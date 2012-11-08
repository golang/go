// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"io/ioutil"
	"os"
	"reflect"
	"testing"
	"time"
)

func TestFileInfoHeader(t *testing.T) {
	fi, err := os.Lstat("testdata/small.txt")
	if err != nil {
		t.Fatal(err)
	}
	h, err := FileInfoHeader(fi, "")
	if err != nil {
		t.Fatalf("on small.txt: %v", err)
	}
	if g, e := h.Name, "small.txt"; g != e {
		t.Errorf("Name = %q; want %q", g, e)
	}
	if g, e := h.Mode, int64(fi.Mode().Perm())|c_ISREG; g != e {
		t.Errorf("Mode = %#o; want %#o", g, e)
	}
	if g, e := h.Size, int64(5); g != e {
		t.Errorf("Size = %v; want %v", g, e)
	}
	if g, e := h.ModTime, fi.ModTime(); !g.Equal(e) {
		t.Errorf("ModTime = %v; want %v", g, e)
	}
}

func TestFileInfoHeaderSymlink(t *testing.T) {
	h, err := FileInfoHeader(symlink{}, "some-target")
	if err != nil {
		t.Fatal(err)
	}
	if g, e := h.Name, "some-symlink"; g != e {
		t.Errorf("Name = %q; want %q", g, e)
	}
	if g, e := h.Linkname, "some-target"; g != e {
		t.Errorf("Linkname = %q; want %q", g, e)
	}
}

type symlink struct{}

func (symlink) Name() string       { return "some-symlink" }
func (symlink) Size() int64        { return 0 }
func (symlink) Mode() os.FileMode  { return os.ModeSymlink }
func (symlink) ModTime() time.Time { return time.Time{} }
func (symlink) IsDir() bool        { return false }
func (symlink) Sys() interface{}   { return nil }

func TestRoundTrip(t *testing.T) {
	data := []byte("some file contents")

	var b bytes.Buffer
	tw := NewWriter(&b)
	hdr := &Header{
		Name:    "file.txt",
		Uid:     1 << 21, // too big for 8 octal digits
		Size:    int64(len(data)),
		ModTime: time.Now(),
	}
	// tar only supports second precision.
	hdr.ModTime = hdr.ModTime.Add(-time.Duration(hdr.ModTime.Nanosecond()) * time.Nanosecond)
	if err := tw.WriteHeader(hdr); err != nil {
		t.Fatalf("tw.WriteHeader: %v", err)
	}
	if _, err := tw.Write(data); err != nil {
		t.Fatalf("tw.Write: %v", err)
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tw.Close: %v", err)
	}

	// Read it back.
	tr := NewReader(&b)
	rHdr, err := tr.Next()
	if err != nil {
		t.Fatalf("tr.Next: %v", err)
	}
	if !reflect.DeepEqual(rHdr, hdr) {
		t.Errorf("Header mismatch.\n got %+v\nwant %+v", rHdr, hdr)
	}
	rData, err := ioutil.ReadAll(tr)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !bytes.Equal(rData, data) {
		t.Errorf("Data mismatch.\n got %q\nwant %q", rData, data)
	}
}

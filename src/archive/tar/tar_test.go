// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestFileInfoHeader(t *testing.T) {
	fi, err := os.Stat("testdata/small.txt")
	if err != nil {
		t.Fatal(err)
	}
	h, err := FileInfoHeader(fi, "")
	if err != nil {
		t.Fatalf("FileInfoHeader: %v", err)
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
	// FileInfoHeader should error when passing nil FileInfo
	if _, err := FileInfoHeader(nil, ""); err == nil {
		t.Fatalf("Expected error when passing nil to FileInfoHeader")
	}
}

func TestFileInfoHeaderDir(t *testing.T) {
	fi, err := os.Stat("testdata")
	if err != nil {
		t.Fatal(err)
	}
	h, err := FileInfoHeader(fi, "")
	if err != nil {
		t.Fatalf("FileInfoHeader: %v", err)
	}
	if g, e := h.Name, "testdata/"; g != e {
		t.Errorf("Name = %q; want %q", g, e)
	}
	// Ignoring c_ISGID for golang.org/issue/4867
	if g, e := h.Mode&^c_ISGID, int64(fi.Mode().Perm())|c_ISDIR; g != e {
		t.Errorf("Mode = %#o; want %#o", g, e)
	}
	if g, e := h.Size, int64(0); g != e {
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

type headerRoundTripTest struct {
	h  *Header
	fm os.FileMode
}

func TestHeaderRoundTrip(t *testing.T) {
	vectors := []headerRoundTripTest{{
		// regular file.
		h: &Header{
			Name:     "test.txt",
			Mode:     0644 | c_ISREG,
			Size:     12,
			ModTime:  time.Unix(1360600916, 0),
			Typeflag: TypeReg,
		},
		fm: 0644,
	}, {
		// symbolic link.
		h: &Header{
			Name:     "link.txt",
			Mode:     0777 | c_ISLNK,
			Size:     0,
			ModTime:  time.Unix(1360600852, 0),
			Typeflag: TypeSymlink,
		},
		fm: 0777 | os.ModeSymlink,
	}, {
		// character device node.
		h: &Header{
			Name:     "dev/null",
			Mode:     0666 | c_ISCHR,
			Size:     0,
			ModTime:  time.Unix(1360578951, 0),
			Typeflag: TypeChar,
		},
		fm: 0666 | os.ModeDevice | os.ModeCharDevice,
	}, {
		// block device node.
		h: &Header{
			Name:     "dev/sda",
			Mode:     0660 | c_ISBLK,
			Size:     0,
			ModTime:  time.Unix(1360578954, 0),
			Typeflag: TypeBlock,
		},
		fm: 0660 | os.ModeDevice,
	}, {
		// directory.
		h: &Header{
			Name:     "dir/",
			Mode:     0755 | c_ISDIR,
			Size:     0,
			ModTime:  time.Unix(1360601116, 0),
			Typeflag: TypeDir,
		},
		fm: 0755 | os.ModeDir,
	}, {
		// fifo node.
		h: &Header{
			Name:     "dev/initctl",
			Mode:     0600 | c_ISFIFO,
			Size:     0,
			ModTime:  time.Unix(1360578949, 0),
			Typeflag: TypeFifo,
		},
		fm: 0600 | os.ModeNamedPipe,
	}, {
		// setuid.
		h: &Header{
			Name:     "bin/su",
			Mode:     0755 | c_ISREG | c_ISUID,
			Size:     23232,
			ModTime:  time.Unix(1355405093, 0),
			Typeflag: TypeReg,
		},
		fm: 0755 | os.ModeSetuid,
	}, {
		// setguid.
		h: &Header{
			Name:     "group.txt",
			Mode:     0750 | c_ISREG | c_ISGID,
			Size:     0,
			ModTime:  time.Unix(1360602346, 0),
			Typeflag: TypeReg,
		},
		fm: 0750 | os.ModeSetgid,
	}, {
		// sticky.
		h: &Header{
			Name:     "sticky.txt",
			Mode:     0600 | c_ISREG | c_ISVTX,
			Size:     7,
			ModTime:  time.Unix(1360602540, 0),
			Typeflag: TypeReg,
		},
		fm: 0600 | os.ModeSticky,
	}, {
		// hard link.
		h: &Header{
			Name:     "hard.txt",
			Mode:     0644 | c_ISREG,
			Size:     0,
			Linkname: "file.txt",
			ModTime:  time.Unix(1360600916, 0),
			Typeflag: TypeLink,
		},
		fm: 0644,
	}, {
		// More information.
		h: &Header{
			Name:     "info.txt",
			Mode:     0600 | c_ISREG,
			Size:     0,
			Uid:      1000,
			Gid:      1000,
			ModTime:  time.Unix(1360602540, 0),
			Uname:    "slartibartfast",
			Gname:    "users",
			Typeflag: TypeReg,
		},
		fm: 0600,
	}}

	for i, v := range vectors {
		fi := v.h.FileInfo()
		h2, err := FileInfoHeader(fi, "")
		if err != nil {
			t.Error(err)
			continue
		}
		if strings.Contains(fi.Name(), "/") {
			t.Errorf("FileInfo of %q contains slash: %q", v.h.Name, fi.Name())
		}
		name := path.Base(v.h.Name)
		if fi.IsDir() {
			name += "/"
		}
		if got, want := h2.Name, name; got != want {
			t.Errorf("i=%d: Name: got %v, want %v", i, got, want)
		}
		if got, want := h2.Size, v.h.Size; got != want {
			t.Errorf("i=%d: Size: got %v, want %v", i, got, want)
		}
		if got, want := h2.Uid, v.h.Uid; got != want {
			t.Errorf("i=%d: Uid: got %d, want %d", i, got, want)
		}
		if got, want := h2.Gid, v.h.Gid; got != want {
			t.Errorf("i=%d: Gid: got %d, want %d", i, got, want)
		}
		if got, want := h2.Uname, v.h.Uname; got != want {
			t.Errorf("i=%d: Uname: got %q, want %q", i, got, want)
		}
		if got, want := h2.Gname, v.h.Gname; got != want {
			t.Errorf("i=%d: Gname: got %q, want %q", i, got, want)
		}
		if got, want := h2.Linkname, v.h.Linkname; got != want {
			t.Errorf("i=%d: Linkname: got %v, want %v", i, got, want)
		}
		if got, want := h2.Typeflag, v.h.Typeflag; got != want {
			t.Logf("%#v %#v", v.h, fi.Sys())
			t.Errorf("i=%d: Typeflag: got %q, want %q", i, got, want)
		}
		if got, want := h2.Mode, v.h.Mode; got != want {
			t.Errorf("i=%d: Mode: got %o, want %o", i, got, want)
		}
		if got, want := fi.Mode(), v.fm; got != want {
			t.Errorf("i=%d: fi.Mode: got %o, want %o", i, got, want)
		}
		if got, want := h2.AccessTime, v.h.AccessTime; got != want {
			t.Errorf("i=%d: AccessTime: got %v, want %v", i, got, want)
		}
		if got, want := h2.ChangeTime, v.h.ChangeTime; got != want {
			t.Errorf("i=%d: ChangeTime: got %v, want %v", i, got, want)
		}
		if got, want := h2.ModTime, v.h.ModTime; got != want {
			t.Errorf("i=%d: ModTime: got %v, want %v", i, got, want)
		}
		if sysh, ok := fi.Sys().(*Header); !ok || sysh != v.h {
			t.Errorf("i=%d: Sys didn't return original *Header", i)
		}
	}
}

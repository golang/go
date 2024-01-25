// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"io/fs"
	"math"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

type testError struct{ error }

type fileOps []any // []T where T is (string | int64)

// testFile is an io.ReadWriteSeeker where the IO operations performed
// on it must match the list of operations in ops.
type testFile struct {
	ops fileOps
	pos int64
}

func (f *testFile) Read(b []byte) (int, error) {
	if len(b) == 0 {
		return 0, nil
	}
	if len(f.ops) == 0 {
		return 0, io.EOF
	}
	s, ok := f.ops[0].(string)
	if !ok {
		return 0, errors.New("unexpected Read operation")
	}

	n := copy(b, s)
	if len(s) > n {
		f.ops[0] = s[n:]
	} else {
		f.ops = f.ops[1:]
	}
	f.pos += int64(len(b))
	return n, nil
}

func (f *testFile) Write(b []byte) (int, error) {
	if len(b) == 0 {
		return 0, nil
	}
	if len(f.ops) == 0 {
		return 0, errors.New("unexpected Write operation")
	}
	s, ok := f.ops[0].(string)
	if !ok {
		return 0, errors.New("unexpected Write operation")
	}

	if !strings.HasPrefix(s, string(b)) {
		return 0, testError{fmt.Errorf("got Write(%q), want Write(%q)", b, s)}
	}
	if len(s) > len(b) {
		f.ops[0] = s[len(b):]
	} else {
		f.ops = f.ops[1:]
	}
	f.pos += int64(len(b))
	return len(b), nil
}

func (f *testFile) Seek(pos int64, whence int) (int64, error) {
	if pos == 0 && whence == io.SeekCurrent {
		return f.pos, nil
	}
	if len(f.ops) == 0 {
		return 0, errors.New("unexpected Seek operation")
	}
	s, ok := f.ops[0].(int64)
	if !ok {
		return 0, errors.New("unexpected Seek operation")
	}

	if s != pos || whence != io.SeekCurrent {
		return 0, testError{fmt.Errorf("got Seek(%d, %d), want Seek(%d, %d)", pos, whence, s, io.SeekCurrent)}
	}
	f.pos += s
	f.ops = f.ops[1:]
	return f.pos, nil
}

func equalSparseEntries(x, y []sparseEntry) bool {
	return (len(x) == 0 && len(y) == 0) || reflect.DeepEqual(x, y)
}

func TestSparseEntries(t *testing.T) {
	vectors := []struct {
		in   []sparseEntry
		size int64

		wantValid    bool          // Result of validateSparseEntries
		wantAligned  []sparseEntry // Result of alignSparseEntries
		wantInverted []sparseEntry // Result of invertSparseEntries
	}{{
		in: []sparseEntry{}, size: 0,
		wantValid:    true,
		wantInverted: []sparseEntry{{0, 0}},
	}, {
		in: []sparseEntry{}, size: 5000,
		wantValid:    true,
		wantInverted: []sparseEntry{{0, 5000}},
	}, {
		in: []sparseEntry{{0, 5000}}, size: 5000,
		wantValid:    true,
		wantAligned:  []sparseEntry{{0, 5000}},
		wantInverted: []sparseEntry{{5000, 0}},
	}, {
		in: []sparseEntry{{1000, 4000}}, size: 5000,
		wantValid:    true,
		wantAligned:  []sparseEntry{{1024, 3976}},
		wantInverted: []sparseEntry{{0, 1000}, {5000, 0}},
	}, {
		in: []sparseEntry{{0, 3000}}, size: 5000,
		wantValid:    true,
		wantAligned:  []sparseEntry{{0, 2560}},
		wantInverted: []sparseEntry{{3000, 2000}},
	}, {
		in: []sparseEntry{{3000, 2000}}, size: 5000,
		wantValid:    true,
		wantAligned:  []sparseEntry{{3072, 1928}},
		wantInverted: []sparseEntry{{0, 3000}, {5000, 0}},
	}, {
		in: []sparseEntry{{2000, 2000}}, size: 5000,
		wantValid:    true,
		wantAligned:  []sparseEntry{{2048, 1536}},
		wantInverted: []sparseEntry{{0, 2000}, {4000, 1000}},
	}, {
		in: []sparseEntry{{0, 2000}, {8000, 2000}}, size: 10000,
		wantValid:    true,
		wantAligned:  []sparseEntry{{0, 1536}, {8192, 1808}},
		wantInverted: []sparseEntry{{2000, 6000}, {10000, 0}},
	}, {
		in: []sparseEntry{{0, 2000}, {2000, 2000}, {4000, 0}, {4000, 3000}, {7000, 1000}, {8000, 0}, {8000, 2000}}, size: 10000,
		wantValid:    true,
		wantAligned:  []sparseEntry{{0, 1536}, {2048, 1536}, {4096, 2560}, {7168, 512}, {8192, 1808}},
		wantInverted: []sparseEntry{{10000, 0}},
	}, {
		in: []sparseEntry{{0, 0}, {1000, 0}, {2000, 0}, {3000, 0}, {4000, 0}, {5000, 0}}, size: 5000,
		wantValid:    true,
		wantInverted: []sparseEntry{{0, 5000}},
	}, {
		in: []sparseEntry{{1, 0}}, size: 0,
		wantValid: false,
	}, {
		in: []sparseEntry{{-1, 0}}, size: 100,
		wantValid: false,
	}, {
		in: []sparseEntry{{0, -1}}, size: 100,
		wantValid: false,
	}, {
		in: []sparseEntry{{0, 0}}, size: -100,
		wantValid: false,
	}, {
		in: []sparseEntry{{math.MaxInt64, 3}, {6, -5}}, size: 35,
		wantValid: false,
	}, {
		in: []sparseEntry{{1, 3}, {6, -5}}, size: 35,
		wantValid: false,
	}, {
		in: []sparseEntry{{math.MaxInt64, math.MaxInt64}}, size: math.MaxInt64,
		wantValid: false,
	}, {
		in: []sparseEntry{{3, 3}}, size: 5,
		wantValid: false,
	}, {
		in: []sparseEntry{{2, 0}, {1, 0}, {0, 0}}, size: 3,
		wantValid: false,
	}, {
		in: []sparseEntry{{1, 3}, {2, 2}}, size: 10,
		wantValid: false,
	}}

	for i, v := range vectors {
		gotValid := validateSparseEntries(v.in, v.size)
		if gotValid != v.wantValid {
			t.Errorf("test %d, validateSparseEntries() = %v, want %v", i, gotValid, v.wantValid)
		}
		if !v.wantValid {
			continue
		}
		gotAligned := alignSparseEntries(append([]sparseEntry{}, v.in...), v.size)
		if !equalSparseEntries(gotAligned, v.wantAligned) {
			t.Errorf("test %d, alignSparseEntries():\ngot  %v\nwant %v", i, gotAligned, v.wantAligned)
		}
		gotInverted := invertSparseEntries(append([]sparseEntry{}, v.in...), v.size)
		if !equalSparseEntries(gotInverted, v.wantInverted) {
			t.Errorf("test %d, inverseSparseEntries():\ngot  %v\nwant %v", i, gotInverted, v.wantInverted)
		}
	}
}

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
	if g, e := h.Mode, int64(fi.Mode().Perm()); g != e {
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
	if g, e := h.Mode&^c_ISGID, int64(fi.Mode().Perm()); g != e {
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
	testenv.MustHaveSymlink(t)

	tmpdir := t.TempDir()

	link := filepath.Join(tmpdir, "link")
	target := tmpdir
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Lstat(link)
	if err != nil {
		t.Fatal(err)
	}

	h, err := FileInfoHeader(fi, target)
	if err != nil {
		t.Fatal(err)
	}
	if g, e := h.Name, fi.Name(); g != e {
		t.Errorf("Name = %q; want %q", g, e)
	}
	if g, e := h.Linkname, target; g != e {
		t.Errorf("Linkname = %q; want %q", g, e)
	}
	if g, e := h.Typeflag, byte(TypeSymlink); g != e {
		t.Errorf("Typeflag = %v; want %v", g, e)
	}
}

func TestRoundTrip(t *testing.T) {
	data := []byte("some file contents")

	var b bytes.Buffer
	tw := NewWriter(&b)
	hdr := &Header{
		Name:       "file.txt",
		Uid:        1 << 21, // Too big for 8 octal digits
		Size:       int64(len(data)),
		ModTime:    time.Now().Round(time.Second),
		PAXRecords: map[string]string{"uid": "2097152"},
		Format:     FormatPAX,
		Typeflag:   TypeReg,
	}
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
	rData, err := io.ReadAll(tr)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !bytes.Equal(rData, data) {
		t.Errorf("Data mismatch.\n got %q\nwant %q", rData, data)
	}
}

type headerRoundTripTest struct {
	h  *Header
	fm fs.FileMode
}

func TestHeaderRoundTrip(t *testing.T) {
	vectors := []headerRoundTripTest{{
		// regular file.
		h: &Header{
			Name:     "test.txt",
			Mode:     0644,
			Size:     12,
			ModTime:  time.Unix(1360600916, 0),
			Typeflag: TypeReg,
		},
		fm: 0644,
	}, {
		// symbolic link.
		h: &Header{
			Name:     "link.txt",
			Mode:     0777,
			Size:     0,
			ModTime:  time.Unix(1360600852, 0),
			Typeflag: TypeSymlink,
		},
		fm: 0777 | fs.ModeSymlink,
	}, {
		// character device node.
		h: &Header{
			Name:     "dev/null",
			Mode:     0666,
			Size:     0,
			ModTime:  time.Unix(1360578951, 0),
			Typeflag: TypeChar,
		},
		fm: 0666 | fs.ModeDevice | fs.ModeCharDevice,
	}, {
		// block device node.
		h: &Header{
			Name:     "dev/sda",
			Mode:     0660,
			Size:     0,
			ModTime:  time.Unix(1360578954, 0),
			Typeflag: TypeBlock,
		},
		fm: 0660 | fs.ModeDevice,
	}, {
		// directory.
		h: &Header{
			Name:     "dir/",
			Mode:     0755,
			Size:     0,
			ModTime:  time.Unix(1360601116, 0),
			Typeflag: TypeDir,
		},
		fm: 0755 | fs.ModeDir,
	}, {
		// fifo node.
		h: &Header{
			Name:     "dev/initctl",
			Mode:     0600,
			Size:     0,
			ModTime:  time.Unix(1360578949, 0),
			Typeflag: TypeFifo,
		},
		fm: 0600 | fs.ModeNamedPipe,
	}, {
		// setuid.
		h: &Header{
			Name:     "bin/su",
			Mode:     0755 | c_ISUID,
			Size:     23232,
			ModTime:  time.Unix(1355405093, 0),
			Typeflag: TypeReg,
		},
		fm: 0755 | fs.ModeSetuid,
	}, {
		// setguid.
		h: &Header{
			Name:     "group.txt",
			Mode:     0750 | c_ISGID,
			Size:     0,
			ModTime:  time.Unix(1360602346, 0),
			Typeflag: TypeReg,
		},
		fm: 0750 | fs.ModeSetgid,
	}, {
		// sticky.
		h: &Header{
			Name:     "sticky.txt",
			Mode:     0600 | c_ISVTX,
			Size:     7,
			ModTime:  time.Unix(1360602540, 0),
			Typeflag: TypeReg,
		},
		fm: 0600 | fs.ModeSticky,
	}, {
		// hard link.
		h: &Header{
			Name:     "hard.txt",
			Mode:     0644,
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
			Mode:     0600,
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

func TestHeaderAllowedFormats(t *testing.T) {
	vectors := []struct {
		header  *Header           // Input header
		paxHdrs map[string]string // Expected PAX headers that may be needed
		formats Format            // Expected formats that can encode the header
	}{{
		header:  &Header{},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{Size: 077777777777},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{Size: 077777777777, Format: FormatUSTAR},
		formats: FormatUSTAR,
	}, {
		header:  &Header{Size: 077777777777, Format: FormatPAX},
		formats: FormatUSTAR | FormatPAX,
	}, {
		header:  &Header{Size: 077777777777, Format: FormatGNU},
		formats: FormatGNU,
	}, {
		header:  &Header{Size: 077777777777 + 1},
		paxHdrs: map[string]string{paxSize: "8589934592"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{Size: 077777777777 + 1, Format: FormatPAX},
		paxHdrs: map[string]string{paxSize: "8589934592"},
		formats: FormatPAX,
	}, {
		header:  &Header{Size: 077777777777 + 1, Format: FormatGNU},
		paxHdrs: map[string]string{paxSize: "8589934592"},
		formats: FormatGNU,
	}, {
		header:  &Header{Mode: 07777777},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{Mode: 07777777 + 1},
		formats: FormatGNU,
	}, {
		header:  &Header{Devmajor: -123},
		formats: FormatGNU,
	}, {
		header:  &Header{Devmajor: 1<<56 - 1},
		formats: FormatGNU,
	}, {
		header:  &Header{Devmajor: 1 << 56},
		formats: FormatUnknown,
	}, {
		header:  &Header{Devmajor: -1 << 56},
		formats: FormatGNU,
	}, {
		header:  &Header{Devmajor: -1<<56 - 1},
		formats: FormatUnknown,
	}, {
		header:  &Header{Name: "用戶名", Devmajor: -1 << 56},
		formats: FormatGNU,
	}, {
		header:  &Header{Size: math.MaxInt64},
		paxHdrs: map[string]string{paxSize: "9223372036854775807"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{Size: math.MinInt64},
		paxHdrs: map[string]string{paxSize: "-9223372036854775808"},
		formats: FormatUnknown,
	}, {
		header:  &Header{Uname: "0123456789abcdef0123456789abcdef"},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{Uname: "0123456789abcdef0123456789abcdefx"},
		paxHdrs: map[string]string{paxUname: "0123456789abcdef0123456789abcdefx"},
		formats: FormatPAX,
	}, {
		header:  &Header{Name: "foobar"},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{Name: strings.Repeat("a", nameSize)},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{Name: strings.Repeat("a", nameSize+1)},
		paxHdrs: map[string]string{paxPath: strings.Repeat("a", nameSize+1)},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{Linkname: "用戶名"},
		paxHdrs: map[string]string{paxLinkpath: "用戶名"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{Linkname: strings.Repeat("用戶名\x00", nameSize)},
		paxHdrs: map[string]string{paxLinkpath: strings.Repeat("用戶名\x00", nameSize)},
		formats: FormatUnknown,
	}, {
		header:  &Header{Linkname: "\x00hello"},
		paxHdrs: map[string]string{paxLinkpath: "\x00hello"},
		formats: FormatUnknown,
	}, {
		header:  &Header{Uid: 07777777},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{Uid: 07777777 + 1},
		paxHdrs: map[string]string{paxUid: "2097152"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{Xattrs: nil},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{Xattrs: map[string]string{"foo": "bar"}},
		paxHdrs: map[string]string{paxSchilyXattr + "foo": "bar"},
		formats: FormatPAX,
	}, {
		header:  &Header{Xattrs: map[string]string{"foo": "bar"}, Format: FormatGNU},
		paxHdrs: map[string]string{paxSchilyXattr + "foo": "bar"},
		formats: FormatUnknown,
	}, {
		header:  &Header{Xattrs: map[string]string{"用戶名": "\x00hello"}},
		paxHdrs: map[string]string{paxSchilyXattr + "用戶名": "\x00hello"},
		formats: FormatPAX,
	}, {
		header:  &Header{Xattrs: map[string]string{"foo=bar": "baz"}},
		formats: FormatUnknown,
	}, {
		header:  &Header{Xattrs: map[string]string{"foo": ""}},
		paxHdrs: map[string]string{paxSchilyXattr + "foo": ""},
		formats: FormatPAX,
	}, {
		header:  &Header{ModTime: time.Unix(0, 0)},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{ModTime: time.Unix(077777777777, 0)},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{ModTime: time.Unix(077777777777+1, 0)},
		paxHdrs: map[string]string{paxMtime: "8589934592"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{ModTime: time.Unix(math.MaxInt64, 0)},
		paxHdrs: map[string]string{paxMtime: "9223372036854775807"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{ModTime: time.Unix(math.MaxInt64, 0), Format: FormatUSTAR},
		paxHdrs: map[string]string{paxMtime: "9223372036854775807"},
		formats: FormatUnknown,
	}, {
		header:  &Header{ModTime: time.Unix(-1, 0)},
		paxHdrs: map[string]string{paxMtime: "-1"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{ModTime: time.Unix(1, 500)},
		paxHdrs: map[string]string{paxMtime: "1.0000005"},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{ModTime: time.Unix(1, 0)},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{ModTime: time.Unix(1, 0), Format: FormatPAX},
		formats: FormatUSTAR | FormatPAX,
	}, {
		header:  &Header{ModTime: time.Unix(1, 500), Format: FormatUSTAR},
		paxHdrs: map[string]string{paxMtime: "1.0000005"},
		formats: FormatUSTAR,
	}, {
		header:  &Header{ModTime: time.Unix(1, 500), Format: FormatPAX},
		paxHdrs: map[string]string{paxMtime: "1.0000005"},
		formats: FormatPAX,
	}, {
		header:  &Header{ModTime: time.Unix(1, 500), Format: FormatGNU},
		paxHdrs: map[string]string{paxMtime: "1.0000005"},
		formats: FormatGNU,
	}, {
		header:  &Header{ModTime: time.Unix(-1, 500)},
		paxHdrs: map[string]string{paxMtime: "-0.9999995"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{ModTime: time.Unix(-1, 500), Format: FormatGNU},
		paxHdrs: map[string]string{paxMtime: "-0.9999995"},
		formats: FormatGNU,
	}, {
		header:  &Header{AccessTime: time.Unix(0, 0)},
		paxHdrs: map[string]string{paxAtime: "0"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{AccessTime: time.Unix(0, 0), Format: FormatUSTAR},
		paxHdrs: map[string]string{paxAtime: "0"},
		formats: FormatUnknown,
	}, {
		header:  &Header{AccessTime: time.Unix(0, 0), Format: FormatPAX},
		paxHdrs: map[string]string{paxAtime: "0"},
		formats: FormatPAX,
	}, {
		header:  &Header{AccessTime: time.Unix(0, 0), Format: FormatGNU},
		paxHdrs: map[string]string{paxAtime: "0"},
		formats: FormatGNU,
	}, {
		header:  &Header{AccessTime: time.Unix(-123, 0)},
		paxHdrs: map[string]string{paxAtime: "-123"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{AccessTime: time.Unix(-123, 0), Format: FormatPAX},
		paxHdrs: map[string]string{paxAtime: "-123"},
		formats: FormatPAX,
	}, {
		header:  &Header{ChangeTime: time.Unix(123, 456)},
		paxHdrs: map[string]string{paxCtime: "123.000000456"},
		formats: FormatPAX | FormatGNU,
	}, {
		header:  &Header{ChangeTime: time.Unix(123, 456), Format: FormatUSTAR},
		paxHdrs: map[string]string{paxCtime: "123.000000456"},
		formats: FormatUnknown,
	}, {
		header:  &Header{ChangeTime: time.Unix(123, 456), Format: FormatGNU},
		paxHdrs: map[string]string{paxCtime: "123.000000456"},
		formats: FormatGNU,
	}, {
		header:  &Header{ChangeTime: time.Unix(123, 456), Format: FormatPAX},
		paxHdrs: map[string]string{paxCtime: "123.000000456"},
		formats: FormatPAX,
	}, {
		header:  &Header{Name: "foo/", Typeflag: TypeDir},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}, {
		header:  &Header{Name: "foo/", Typeflag: TypeReg},
		formats: FormatUnknown,
	}, {
		header:  &Header{Name: "foo/", Typeflag: TypeSymlink},
		formats: FormatUSTAR | FormatPAX | FormatGNU,
	}}

	for i, v := range vectors {
		formats, paxHdrs, err := v.header.allowedFormats()
		if formats != v.formats {
			t.Errorf("test %d, allowedFormats(): got %v, want %v", i, formats, v.formats)
		}
		if formats&FormatPAX > 0 && !reflect.DeepEqual(paxHdrs, v.paxHdrs) && !(len(paxHdrs) == 0 && len(v.paxHdrs) == 0) {
			t.Errorf("test %d, allowedFormats():\ngot  %v\nwant %s", i, paxHdrs, v.paxHdrs)
		}
		if (formats != FormatUnknown) && (err != nil) {
			t.Errorf("test %d, unexpected error: %v", i, err)
		}
		if (formats == FormatUnknown) && (err == nil) {
			t.Errorf("test %d, got nil-error, want non-nil error", i)
		}
	}
}

func Benchmark(b *testing.B) {
	type file struct {
		hdr  *Header
		body []byte
	}

	vectors := []struct {
		label string
		files []file
	}{{
		"USTAR",
		[]file{{
			&Header{Name: "bar", Mode: 0640, Size: int64(3)},
			[]byte("foo"),
		}, {
			&Header{Name: "world", Mode: 0640, Size: int64(5)},
			[]byte("hello"),
		}},
	}, {
		"GNU",
		[]file{{
			&Header{Name: "bar", Mode: 0640, Size: int64(3), Devmajor: -1},
			[]byte("foo"),
		}, {
			&Header{Name: "world", Mode: 0640, Size: int64(5), Devmajor: -1},
			[]byte("hello"),
		}},
	}, {
		"PAX",
		[]file{{
			&Header{Name: "bar", Mode: 0640, Size: int64(3), Xattrs: map[string]string{"foo": "bar"}},
			[]byte("foo"),
		}, {
			&Header{Name: "world", Mode: 0640, Size: int64(5), Xattrs: map[string]string{"foo": "bar"}},
			[]byte("hello"),
		}},
	}}

	b.Run("Writer", func(b *testing.B) {
		for _, v := range vectors {
			b.Run(v.label, func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					// Writing to io.Discard because we want to
					// test purely the writer code and not bring in disk performance into this.
					tw := NewWriter(io.Discard)
					for _, file := range v.files {
						if err := tw.WriteHeader(file.hdr); err != nil {
							b.Errorf("unexpected WriteHeader error: %v", err)
						}
						if _, err := tw.Write(file.body); err != nil {
							b.Errorf("unexpected Write error: %v", err)
						}
					}
					if err := tw.Close(); err != nil {
						b.Errorf("unexpected Close error: %v", err)
					}
				}
			})
		}
	})

	b.Run("Reader", func(b *testing.B) {
		for _, v := range vectors {
			var buf bytes.Buffer
			var r bytes.Reader

			// Write the archive to a byte buffer.
			tw := NewWriter(&buf)
			for _, file := range v.files {
				tw.WriteHeader(file.hdr)
				tw.Write(file.body)
			}
			tw.Close()
			b.Run(v.label, func(b *testing.B) {
				b.ReportAllocs()
				// Read from the byte buffer.
				for i := 0; i < b.N; i++ {
					r.Reset(buf.Bytes())
					tr := NewReader(&r)
					if _, err := tr.Next(); err != nil {
						b.Errorf("unexpected Next error: %v", err)
					}
					if _, err := io.Copy(io.Discard, tr); err != nil {
						b.Errorf("unexpected Copy error : %v", err)
					}
				}
			})
		}
	})

}

const (
	testUid = 10
	testGid = 20
)

var _ fileInfoNames = fileInfoNames{}

type fileInfoNames struct{}

func (f *fileInfoNames) Name() string {
	return "tmp"
}

func (f *fileInfoNames) Size() int64 {
	return 0
}

func (f *fileInfoNames) Mode() fs.FileMode {
	return 0777
}

func (f *fileInfoNames) ModTime() time.Time {
	return time.Time{}
}

func (f *fileInfoNames) IsDir() bool {
	return false
}

func (f *fileInfoNames) Sys() any {
	return nil
}

func (f *fileInfoNames) Uname() (string, error) {
	return "Uname", nil
}

func (f *fileInfoNames) Gname() (string, error) {
	return "Gname", nil
}

func (f *fileInfoNames) Uid() (int, error) {
	return testUid, nil
}

func (f *fileInfoNames) Gid() (int, error) {
	return testGid, nil
}

func TestFileInfoHeaderUseFileInfoNames(t *testing.T) {
	info := &fileInfoNames{}
	header, err := FileInfoHeader(info, "")
	if err != nil {
		t.Fatal(err)
	}
	if header.Uname != "Uname" {
		t.Fatalf("header.Uname: got %s, want %s", header.Uname, "Uname")
	}
	if header.Gname != "Gname" {
		t.Fatalf("header.Gname: got %s, want %s", header.Gname, "Gname")
	}
	if header.Uid != testUid {
		t.Fatalf("header.Uid: got %d, want %d", header.Uid, testUid)
	}
	if header.Gid != testGid {
		t.Fatalf("header.Gid: got %d, want %d", header.Gid, testGid)
	}
}

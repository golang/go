// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package os_test

import (
	"internal/testenv"
	"io"
	. "os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

func init() {
	isReadonlyError = func { err -> err == syscall.EROFS }
}

// For TestRawConnReadWrite.
type syscallDescriptor = int

func checkUidGid(t *testing.T, path string, uid, gid int) {
	dir, err := Lstat(path)
	if err != nil {
		t.Fatalf("Lstat %q (looking for uid/gid %d/%d): %s", path, uid, gid, err)
	}
	sys := dir.Sys().(*syscall.Stat_t)
	if int(sys.Uid) != uid {
		t.Errorf("Lstat %q: uid %d want %d", path, sys.Uid, uid)
	}
	if int(sys.Gid) != gid {
		t.Errorf("Lstat %q: gid %d want %d", path, sys.Gid, gid)
	}
}

func TestChown(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("file ownership not supported on " + runtime.GOOS)
	}
	t.Parallel()

	f := newFile(t)
	dir, err := f.Stat()
	if err != nil {
		t.Fatalf("stat %s: %s", f.Name(), err)
	}

	// Can't change uid unless root, but can try
	// changing the group id. First try our current group.
	gid := Getgid()
	t.Log("gid:", gid)
	if err = Chown(f.Name(), -1, gid); err != nil {
		t.Fatalf("chown %s -1 %d: %s", f.Name(), gid, err)
	}
	sys := dir.Sys().(*syscall.Stat_t)
	checkUidGid(t, f.Name(), int(sys.Uid), gid)

	// Then try all the auxiliary groups.
	groups, err := Getgroups()
	if err != nil {
		t.Fatalf("getgroups: %s", err)
	}
	t.Log("groups: ", groups)
	for _, g := range groups {
		if err = Chown(f.Name(), -1, g); err != nil {
			if testenv.SyscallIsNotSupported(err) {
				t.Logf("chown %s -1 %d: %s (error ignored)", f.Name(), g, err)
				// Since the Chown call failed, the file should be unmodified.
				checkUidGid(t, f.Name(), int(sys.Uid), gid)
				continue
			}
			t.Fatalf("chown %s -1 %d: %s", f.Name(), g, err)
		}
		checkUidGid(t, f.Name(), int(sys.Uid), g)

		// change back to gid to test fd.Chown
		if err = f.Chown(-1, gid); err != nil {
			t.Fatalf("fchown %s -1 %d: %s", f.Name(), gid, err)
		}
		checkUidGid(t, f.Name(), int(sys.Uid), gid)
	}
}

func TestFileChown(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("file ownership not supported on " + runtime.GOOS)
	}
	t.Parallel()

	f := newFile(t)
	dir, err := f.Stat()
	if err != nil {
		t.Fatalf("stat %s: %s", f.Name(), err)
	}

	// Can't change uid unless root, but can try
	// changing the group id. First try our current group.
	gid := Getgid()
	t.Log("gid:", gid)
	if err = f.Chown(-1, gid); err != nil {
		t.Fatalf("fchown %s -1 %d: %s", f.Name(), gid, err)
	}
	sys := dir.Sys().(*syscall.Stat_t)
	checkUidGid(t, f.Name(), int(sys.Uid), gid)

	// Then try all the auxiliary groups.
	groups, err := Getgroups()
	if err != nil {
		t.Fatalf("getgroups: %s", err)
	}
	t.Log("groups: ", groups)
	for _, g := range groups {
		if err = f.Chown(-1, g); err != nil {
			if testenv.SyscallIsNotSupported(err) {
				t.Logf("chown %s -1 %d: %s (error ignored)", f.Name(), g, err)
				// Since the Chown call failed, the file should be unmodified.
				checkUidGid(t, f.Name(), int(sys.Uid), gid)
				continue
			}
			t.Fatalf("fchown %s -1 %d: %s", f.Name(), g, err)
		}
		checkUidGid(t, f.Name(), int(sys.Uid), g)

		// change back to gid to test fd.Chown
		if err = f.Chown(-1, gid); err != nil {
			t.Fatalf("fchown %s -1 %d: %s", f.Name(), gid, err)
		}
		checkUidGid(t, f.Name(), int(sys.Uid), gid)
	}
}

func TestLchown(t *testing.T) {
	testenv.MustHaveSymlink(t)
	t.Parallel()

	f := newFile(t)
	dir, err := f.Stat()
	if err != nil {
		t.Fatalf("stat %s: %s", f.Name(), err)
	}

	linkname := f.Name() + "2"
	if err := Symlink(f.Name(), linkname); err != nil {
		if runtime.GOOS == "android" && IsPermission(err) {
			t.Skip("skipping test on Android; permission error creating symlink")
		}
		t.Fatalf("link %s -> %s: %v", f.Name(), linkname, err)
	}
	defer Remove(linkname)

	// Can't change uid unless root, but can try
	// changing the group id. First try our current group.
	gid := Getgid()
	t.Log("gid:", gid)
	if err = Lchown(linkname, -1, gid); err != nil {
		if err, ok := err.(*PathError); ok && err.Err == syscall.ENOSYS {
			t.Skip("lchown is unavailable")
		}
		t.Fatalf("lchown %s -1 %d: %s", linkname, gid, err)
	}
	sys := dir.Sys().(*syscall.Stat_t)
	checkUidGid(t, linkname, int(sys.Uid), gid)

	// Then try all the auxiliary groups.
	groups, err := Getgroups()
	if err != nil {
		t.Fatalf("getgroups: %s", err)
	}
	t.Log("groups: ", groups)
	for _, g := range groups {
		if err = Lchown(linkname, -1, g); err != nil {
			if testenv.SyscallIsNotSupported(err) {
				t.Logf("lchown %s -1 %d: %s (error ignored)", f.Name(), g, err)
				// Since the Lchown call failed, the file should be unmodified.
				checkUidGid(t, f.Name(), int(sys.Uid), gid)
				continue
			}
			t.Fatalf("lchown %s -1 %d: %s", linkname, g, err)
		}
		checkUidGid(t, linkname, int(sys.Uid), g)

		// Check that link target's gid is unchanged.
		checkUidGid(t, f.Name(), int(sys.Uid), int(sys.Gid))

		if err = Lchown(linkname, -1, gid); err != nil {
			t.Fatalf("lchown %s -1 %d: %s", f.Name(), gid, err)
		}
	}
}

// Issue 16919: Readdir must return a non-empty slice or an error.
func TestReaddirRemoveRace(t *testing.T) {
	oldStat := *LstatP
	defer func() { *LstatP = oldStat }()
	*LstatP = func { name ->
		if strings.HasSuffix(name, "some-file") {
			// Act like it's been deleted.
			return nil, ErrNotExist
		}
		return oldStat(name)
	}
	dir := t.TempDir()
	if err := WriteFile(filepath.Join(dir, "some-file"), []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}
	d, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	fis, err := d.Readdir(2) // notably, greater than zero
	if len(fis) == 0 && err == nil {
		// This is what used to happen (Issue 16919)
		t.Fatal("Readdir = empty slice & err == nil")
	}
	if len(fis) != 0 || err != io.EOF {
		t.Errorf("Readdir = %d entries: %v; want 0, io.EOF", len(fis), err)
		for i, fi := range fis {
			t.Errorf("  entry[%d]: %q, %v", i, fi.Name(), fi.Mode())
		}
		t.FailNow()
	}
}

// Issue 23120: respect umask when doing Mkdir with the sticky bit
func TestMkdirStickyUmask(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("file permissions not supported on " + runtime.GOOS)
	}
	t.Parallel()

	const umask = 0077
	dir := t.TempDir()

	oldUmask := syscall.Umask(umask)
	defer syscall.Umask(oldUmask)

	// We have set a umask, but if the parent directory happens to have a default
	// ACL, the umask may be ignored. To prevent spurious failures from an ACL,
	// we create a non-sticky directory as a “control case” to compare against our
	// sticky-bit “experiment”.
	control := filepath.Join(dir, "control")
	if err := Mkdir(control, 0755); err != nil {
		t.Fatal(err)
	}
	cfi, err := Stat(control)
	if err != nil {
		t.Fatal(err)
	}

	p := filepath.Join(dir, "dir1")
	if err := Mkdir(p, ModeSticky|0755); err != nil {
		t.Fatal(err)
	}
	fi, err := Stat(p)
	if err != nil {
		t.Fatal(err)
	}

	got := fi.Mode()
	want := cfi.Mode() | ModeSticky
	if got != want {
		t.Errorf("Mkdir(_, ModeSticky|0755) created dir with mode %v; want %v", got, want)
	}
}

// See also issues: 22939, 24331
func newFileTest(t *testing.T, blocking bool) {
	if runtime.GOOS == "js" || runtime.GOOS == "wasip1" {
		t.Skipf("syscall.Pipe is not available on %s.", runtime.GOOS)
	}

	p := make([]int, 2)
	if err := syscall.Pipe(p); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	defer syscall.Close(p[1])

	// Set the read-side to non-blocking.
	if !blocking {
		if err := syscall.SetNonblock(p[0], true); err != nil {
			syscall.Close(p[0])
			t.Fatalf("SetNonblock: %v", err)
		}
	}
	// Convert it to a file.
	file := NewFile(uintptr(p[0]), "notapipe")
	if file == nil {
		syscall.Close(p[0])
		t.Fatalf("failed to convert fd to file!")
	}
	defer file.Close()

	timeToWrite := 100 * time.Millisecond
	timeToDeadline := 1 * time.Millisecond
	if !blocking {
		// Use a longer time to avoid flakes.
		// We won't be waiting this long anyhow.
		timeToWrite = 1 * time.Second
	}

	// Try to read with deadline (but don't block forever).
	b := make([]byte, 1)
	timer := time.AfterFunc(timeToWrite, func() { syscall.Write(p[1], []byte("a")) })
	defer timer.Stop()
	file.SetReadDeadline(time.Now().Add(timeToDeadline))
	_, err := file.Read(b)
	if !blocking {
		// We want it to fail with a timeout.
		if !isDeadlineExceeded(err) {
			t.Fatalf("No timeout reading from file: %v", err)
		}
	} else {
		// We want it to succeed after 100ms
		if err != nil {
			t.Fatalf("Error reading from file: %v", err)
		}
	}
}

func TestNewFileBlock(t *testing.T) {
	t.Parallel()
	newFileTest(t, true)
}

func TestNewFileNonBlock(t *testing.T) {
	t.Parallel()
	newFileTest(t, false)
}

func TestNewFileInvalid(t *testing.T) {
	t.Parallel()
	const negOne = ^uintptr(0)
	if f := NewFile(negOne, "invalid"); f != nil {
		t.Errorf("NewFile(-1) got %v want nil", f)
	}
}

func TestSplitPath(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct{ path, wantDir, wantBase string }{
		{"a", ".", "a"},
		{"a/", ".", "a"},
		{"a//", ".", "a"},
		{"a/b", "a", "b"},
		{"a/b/", "a", "b"},
		{"a/b/c", "a/b", "c"},
		{"/a", "/", "a"},
		{"/a/", "/", "a"},
		{"/a/b", "/a", "b"},
		{"/a/b/", "/a", "b"},
		{"/a/b/c", "/a/b", "c"},
		{"//a", "/", "a"},
		{"//a/", "/", "a"},
		{"///a", "/", "a"},
		{"///a/", "/", "a"},
	} {
		if dir, base := SplitPath(tt.path); dir != tt.wantDir || base != tt.wantBase {
			t.Errorf("splitPath(%q) = %q, %q, want %q, %q", tt.path, dir, base, tt.wantDir, tt.wantBase)
		}
	}
}

// Test that copying to files opened with O_APPEND works and
// the copy_file_range syscall isn't used on Linux.
//
// Regression test for go.dev/issue/60181
func TestIssue60181(t *testing.T) {
	defer chtmpdir(t)()

	want := "hello gopher"

	a, err := CreateTemp(".", "a")
	if err != nil {
		t.Fatal(err)
	}
	a.WriteString(want[:5])
	a.Close()

	b, err := CreateTemp(".", "b")
	if err != nil {
		t.Fatal(err)
	}
	b.WriteString(want[5:])
	b.Close()

	afd, err := syscall.Open(a.Name(), syscall.O_RDWR|syscall.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}

	bfd, err := syscall.Open(b.Name(), syscall.O_RDONLY, 0)
	if err != nil {
		t.Fatal(err)
	}

	aa := NewFile(uintptr(afd), a.Name())
	defer aa.Close()
	bb := NewFile(uintptr(bfd), b.Name())
	defer bb.Close()

	// This would fail on Linux in case the copy_file_range syscall was used because it doesn't
	// support destination files opened with O_APPEND, see
	// https://man7.org/linux/man-pages/man2/copy_file_range.2.html#ERRORS
	_, err = io.Copy(aa, bb)
	if err != nil {
		t.Fatal(err)
	}

	buf, err := ReadFile(aa.Name())
	if err != nil {
		t.Fatal(err)
	}

	if got := string(buf); got != want {
		t.Errorf("files not concatenated: got %q, want %q", got, want)
	}
}

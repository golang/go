// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd js,wasm linux netbsd openbsd solaris

package os_test

import (
	"io"
	"io/ioutil"
	. "os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

func init() {
	isReadonlyError = func(err error) bool { return err == syscall.EROFS }
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
	// Use TempDir() to make sure we're on a local file system,
	// so that the group ids returned by Getgroups will be allowed
	// on the file. On NFS, the Getgroups groups are
	// basically useless.
	f := newFile("TestChown", t)
	defer Remove(f.Name())
	defer f.Close()
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
	// Use TempDir() to make sure we're on a local file system,
	// so that the group ids returned by Getgroups will be allowed
	// on the file. On NFS, the Getgroups groups are
	// basically useless.
	f := newFile("TestFileChown", t)
	defer Remove(f.Name())
	defer f.Close()
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
	// Use TempDir() to make sure we're on a local file system,
	// so that the group ids returned by Getgroups will be allowed
	// on the file. On NFS, the Getgroups groups are
	// basically useless.
	f := newFile("TestLchown", t)
	defer Remove(f.Name())
	defer f.Close()
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
			t.Fatalf("lchown %s -1 %d: %s", linkname, g, err)
		}
		checkUidGid(t, linkname, int(sys.Uid), g)

		// Check that link target's gid is unchanged.
		checkUidGid(t, f.Name(), int(sys.Uid), int(sys.Gid))
	}
}

// Issue 16919: Readdir must return a non-empty slice or an error.
func TestReaddirRemoveRace(t *testing.T) {
	oldStat := *LstatP
	defer func() { *LstatP = oldStat }()
	*LstatP = func(name string) (FileInfo, error) {
		if strings.HasSuffix(name, "some-file") {
			// Act like it's been deleted.
			return nil, ErrNotExist
		}
		return oldStat(name)
	}
	dir := newDir("TestReaddirRemoveRace", t)
	defer RemoveAll(dir)
	if err := ioutil.WriteFile(filepath.Join(dir, "some-file"), []byte("hello"), 0644); err != nil {
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
	const umask = 0077
	dir := newDir("TestMkdirStickyUmask", t)
	defer RemoveAll(dir)
	oldUmask := syscall.Umask(umask)
	defer syscall.Umask(oldUmask)
	p := filepath.Join(dir, "dir1")
	if err := Mkdir(p, ModeSticky|0755); err != nil {
		t.Fatal(err)
	}
	fi, err := Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if mode := fi.Mode(); (mode&umask) != 0 || (mode&^ModePerm) != (ModeDir|ModeSticky) {
		t.Errorf("unexpected mode %s", mode)
	}
}

// See also issues: 22939, 24331
func newFileTest(t *testing.T, blocking bool) {
	if runtime.GOOS == "js" {
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

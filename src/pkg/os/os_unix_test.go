// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package os_test

import (
	. "os"
	"runtime"
	"syscall"
	"testing"
)

func checkUidGid(t *testing.T, path string, uid, gid int) {
	dir, err := Stat(path)
	if err != nil {
		t.Fatalf("Stat %q (looking for uid/gid %d/%d): %s", path, uid, gid, err)
	}
	sys := dir.Sys().(*syscall.Stat_t)
	if int(sys.Uid) != uid {
		t.Errorf("Stat %q: uid %d want %d", path, sys.Uid, uid)
	}
	if int(sys.Gid) != gid {
		t.Errorf("Stat %q: gid %d want %d", path, sys.Gid, gid)
	}
}

func TestChown(t *testing.T) {
	// Chown is not supported under windows os Plan 9.
	// Plan9 provides a native ChownPlan9 version instead.
	if runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
		return
	}
	// Use TempDir() to make sure we're on a local file system,
	// so that the group ids returned by Getgroups will be allowed
	// on the file.  On NFS, the Getgroups groups are
	// basically useless.
	f := newFile("TestChown", t)
	defer Remove(f.Name())
	defer f.Close()
	dir, err := f.Stat()
	if err != nil {
		t.Fatalf("stat %s: %s", f.Name(), err)
	}

	// Can't change uid unless root, but can try
	// changing the group id.  First try our current group.
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

func TestReaddirWithBadLstat(t *testing.T) {
	handle, err := Open(sfdir)
	failfile := sfdir + "/" + sfname
	if err != nil {
		t.Fatalf("Couldn't open %s: %s", sfdir, err)
	}

	*LstatP = func(file string) (FileInfo, error) {
		if file == failfile {
			var fi FileInfo
			return fi, ErrInvalid
		}
		return Lstat(file)
	}
	defer func() { *LstatP = Lstat }()

	dirs, err := handle.Readdir(-1)
	if err != ErrInvalid {
		t.Fatalf("Expected Readdir to return ErrInvalid, got %v", err)
	}
	foundfail := false
	for _, dir := range dirs {
		if dir.Name() == sfname {
			foundfail = true
			if dir.Sys() != nil {
				t.Errorf("Expected Readdir for %s should not contain Sys", failfile)
			}
		} else {
			if dir.Sys() == nil {
				t.Errorf("Readdir for every file other than %s should contain Sys, but %s/%s didn't either", failfile, sfdir, dir.Name())
			}
		}
	}
	if !foundfail {
		t.Fatalf("Expected %s from Readdir, but didn't find it", failfile)
	}
}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux

package unix_test

import (
	"os"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

func TestXattr(t *testing.T) {
	defer chtmpdir(t)()

	f := "xattr1"
	touch(t, f)

	xattrName := "user.test"
	xattrDataSet := "gopher"
	err := unix.Setxattr(f, xattrName, []byte(xattrDataSet), 0)
	if err == unix.ENOTSUP || err == unix.EOPNOTSUPP {
		t.Skip("filesystem does not support extended attributes, skipping test")
	} else if err != nil {
		t.Fatalf("Setxattr: %v", err)
	}

	// find size
	size, err := unix.Listxattr(f, nil)
	if err != nil {
		t.Fatalf("Listxattr: %v", err)
	}

	if size <= 0 {
		t.Fatalf("Listxattr returned an empty list of attributes")
	}

	buf := make([]byte, size)
	read, err := unix.Listxattr(f, buf)
	if err != nil {
		t.Fatalf("Listxattr: %v", err)
	}

	xattrs := stringsFromByteSlice(buf[:read])

	xattrWant := xattrName
	if runtime.GOOS == "freebsd" {
		// On FreeBSD, the namespace is stored separately from the xattr
		// name and Listxattr doesn't return the namespace prefix.
		xattrWant = strings.TrimPrefix(xattrWant, "user.")
	}
	found := false
	for _, name := range xattrs {
		if name == xattrWant {
			found = true
		}
	}

	if !found {
		t.Errorf("Listxattr did not return previously set attribute '%s'", xattrName)
	}

	// find size
	size, err = unix.Getxattr(f, xattrName, nil)
	if err != nil {
		t.Fatalf("Getxattr: %v", err)
	}

	if size <= 0 {
		t.Fatalf("Getxattr returned an empty attribute")
	}

	xattrDataGet := make([]byte, size)
	_, err = unix.Getxattr(f, xattrName, xattrDataGet)
	if err != nil {
		t.Fatalf("Getxattr: %v", err)
	}

	got := string(xattrDataGet)
	if got != xattrDataSet {
		t.Errorf("Getxattr: expected attribute value %s, got %s", xattrDataSet, got)
	}

	err = unix.Removexattr(f, xattrName)
	if err != nil {
		t.Fatalf("Removexattr: %v", err)
	}

	n := "nonexistent"
	err = unix.Lsetxattr(n, xattrName, []byte(xattrDataSet), 0)
	if err != unix.ENOENT {
		t.Errorf("Lsetxattr: expected %v on non-existent file, got %v", unix.ENOENT, err)
	}

	_, err = unix.Lgetxattr(n, xattrName, nil)
	if err != unix.ENOENT {
		t.Errorf("Lgetxattr: %v", err)
	}

	s := "symlink1"
	err = os.Symlink(n, s)
	if err != nil {
		t.Fatal(err)
	}

	err = unix.Lsetxattr(s, xattrName, []byte(xattrDataSet), 0)
	if err != nil {
		// Linux and Android doen't support xattrs on symlinks according
		// to xattr(7), so just test that we get the proper error.
		if (runtime.GOOS != "linux" && runtime.GOOS != "android") || err != unix.EPERM {
			t.Fatalf("Lsetxattr: %v", err)
		}
	}
}

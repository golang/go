// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || wasip1

package unix_test

import (
	"internal/syscall/unix"
	"os"
	"runtime"
	"testing"
)

// TestFchmodAtSymlinkNofollow verifies that Fchmodat honors the AT_SYMLINK_NOFOLLOW flag.
func TestFchmodatSymlinkNofollow(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("wasip1 doesn't support chmod")
	}

	dir := t.TempDir()
	filename := dir + "/file"
	linkname := dir + "/symlink"
	if err := os.WriteFile(filename, nil, 0o100); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filename, linkname); err != nil {
		t.Fatal(err)
	}

	parent, err := os.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close()

	lstatMode := func(path string) os.FileMode {
		st, err := os.Lstat(path)
		if err != nil {
			t.Fatal(err)
		}
		return st.Mode()
	}

	// Fchmodat with no flags follows symlinks.
	const mode1 = 0o200
	if err := unix.Fchmodat(int(parent.Fd()), "symlink", mode1, 0); err != nil {
		t.Fatal(err)
	}
	if got, want := lstatMode(filename), os.FileMode(mode1); got != want {
		t.Errorf("after Fchmodat(parent, symlink, %v, 0); mode = %v, want %v", mode1, got, want)
	}

	// Fchmodat with AT_SYMLINK_NOFOLLOW does not follow symlinks.
	// The Fchmodat call may fail or chmod the symlink itself, depending on the kernel version.
	const mode2 = 0o400
	unix.Fchmodat(int(parent.Fd()), "symlink", mode2, unix.AT_SYMLINK_NOFOLLOW)
	if got, want := lstatMode(filename), os.FileMode(mode1); got != want {
		t.Errorf("after Fchmodat(parent, symlink, %v, AT_SYMLINK_NOFOLLOW); mode = %v, want %v", mode1, got, want)
	}
}

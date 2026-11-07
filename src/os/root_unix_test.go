// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package os_test

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
)

func TestRootChown(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("Chown not supported on " + runtime.GOOS)
	}

	// Look up the current default uid/gid.
	f := newFile(t)
	dir, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	sys := dir.Sys().(*syscall.Stat_t)

	groups, err := os.Getgroups()
	if err != nil {
		t.Fatal(err)
	}
	groups = append(groups, os.Getgid())
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			if target != "" {
				if err := os.WriteFile(target, nil, 0o666); err != nil {
					t.Fatal(err)
				}
			}
			for _, gid := range groups {
				err := root.Chown(test.open, -1, gid)
				if errEndsTest(t, err, test.wantError, "root.Chown(%q, -1, %v)", test.open, gid) {
					return
				}
				checkUidGid(t, target, int(sys.Uid), gid)
			}
		})
	}
}

func TestRootLchown(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("Lchown not supported on " + runtime.GOOS)
	}

	// Look up the current default uid/gid.
	f := newFile(t)
	dir, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	sys := dir.Sys().(*syscall.Stat_t)

	groups, err := os.Getgroups()
	if err != nil {
		t.Fatal(err)
	}
	groups = append(groups, os.Getgid())
	for _, test := range rootTestCases {
		test.run(t, func(t *testing.T, target string, root *os.Root) {
			if test.ltarget != "" {
				test.wantError = false
				target = filepath.Join(root.Name(), test.ltarget)
			} else if target != "" {
				if err := os.WriteFile(target, nil, 0o666); err != nil {
					t.Fatal(err)
				}
			}
			for _, gid := range groups {
				err := root.Lchown(test.open, -1, gid)
				if errEndsTest(t, err, test.wantError, "root.Lchown(%q, -1, %v)", test.open, gid) {
					return
				}
				checkUidGid(t, target, int(sys.Uid), gid)
			}
		})
	}
}

func TestRootConsistencyChown(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("Chown not supported on " + runtime.GOOS)
	}
	groups, err := os.Getgroups()
	if err != nil {
		t.Fatal(err)
	}
	var gid int
	if len(groups) == 0 {
		gid = os.Getgid()
	} else {
		gid = groups[0]
	}
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			chown := os.Chown
			lstat := os.Lstat
			if r != nil {
				chown = r.Chown
				lstat = r.Lstat
			}
			err := chown(path, -1, gid)
			if err != nil {
				return "", err
			}
			fi, err := lstat(path)
			if err != nil {
				return "", err
			}
			sys := fi.Sys().(*syscall.Stat_t)
			return fmt.Sprintf("%v %v", sys.Uid, sys.Gid), nil
		})
	}
}

func TestRootConsistencyLchown(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		t.Skip("Lchown not supported on " + runtime.GOOS)
	}
	groups, err := os.Getgroups()
	if err != nil {
		t.Fatal(err)
	}
	var gid int
	if len(groups) == 0 {
		gid = os.Getgid()
	} else {
		gid = groups[0]
	}
	for _, test := range rootConsistencyTestCases {
		test.run(t, func(t *testing.T, path string, r *os.Root) (string, error) {
			lchown := os.Lchown
			lstat := os.Lstat
			if r != nil {
				lchown = r.Lchown
				lstat = r.Lstat
			}
			err := lchown(path, -1, gid)
			if err != nil {
				return "", err
			}
			fi, err := lstat(path)
			if err != nil {
				return "", err
			}
			sys := fi.Sys().(*syscall.Stat_t)
			return fmt.Sprintf("%v %v", sys.Uid, sys.Gid), nil
		})
	}
}

func TestRootNoReadPermission(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("O_PATH intermediate directories are Linux-specific")
	}
	if os.Getuid() == 0 {
		t.Skip("test requires an unprivileged user")
	}

	dir := t.TempDir()
	parent := filepath.Join(dir, "parent")
	if err := os.Mkdir(parent, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(parent, 0o300); err != nil {
		t.Fatal(err)
	}
	// Restore permissions so the test's temp dir can be removed.
	defer os.Chmod(parent, 0o700)

	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	// Creating an entry in a directory requires write and search
	// permission on the directory, not read permission.
	// Root operations must preserve those permission semantics
	// instead of additionally requiring read permission on
	// intermediate directories (issue #81605).
	if err := root.Mkdir("parent/child", 0o700); err != nil {
		t.Errorf("Root.Mkdir: %v", err)
	}
	if err := root.MkdirAll("parent/a/b", 0o700); err != nil {
		t.Errorf("Root.MkdirAll: %v", err)
	}
	f, err := root.Create("parent/file")
	if err != nil {
		t.Errorf("Root.Create: %v", err)
	} else {
		f.Close()
	}
	if _, err := root.Stat("parent/child"); err != nil {
		t.Errorf("Root.Stat: %v", err)
	}
	if err := root.Symlink("child", "parent/link"); err != nil {
		t.Errorf("Root.Symlink: %v", err)
	}
	if _, err := root.Readlink("parent/link"); err != nil {
		t.Errorf("Root.Readlink: %v", err)
	}
}

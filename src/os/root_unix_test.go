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

func TestRootOpenFileNoFollow(t *testing.T) {
	dir := t.TempDir()
	for name, content := range map[string]string{
		"target":     "intact",
		"dir/target": "intact",
	} {
		name = filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(name), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(name, []byte(content), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	for link, target := range map[string]string{
		"link":     "target",
		"dangling": "missing",
		"dirlink":  "dir",
	} {
		if err := os.Symlink(target, filepath.Join(dir, link)); err != nil {
			t.Skipf("skipping: cannot create symlink: %v", err)
		}
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	// O_NOFOLLOW applies to the final path component only.
	// Root.OpenFile should fail on a symlink there, with the same error
	// as os.OpenFile, and should follow symlinks in earlier components.
	for _, test := range []struct {
		name    string
		flag    int
		wantErr bool
	}{
		{"link", os.O_WRONLY, false},
		{"link", os.O_WRONLY | syscall.O_NOFOLLOW, true},
		{"link", os.O_RDONLY | syscall.O_NOFOLLOW, true},
		{"link", os.O_WRONLY | os.O_TRUNC | syscall.O_NOFOLLOW, true},
		{"dangling", os.O_WRONLY | os.O_CREATE | syscall.O_NOFOLLOW, true},
		{"dirlink", os.O_RDONLY | syscall.O_NOFOLLOW, true},
		{"dirlink", os.O_RDONLY | syscall.O_DIRECTORY | syscall.O_NOFOLLOW, true},
		{"target", os.O_RDWR | syscall.O_NOFOLLOW, false},
		{"dirlink/target", os.O_RDWR | syscall.O_NOFOLLOW, false},
		{"dir", os.O_RDONLY | syscall.O_DIRECTORY | syscall.O_NOFOLLOW, false},
	} {
		name := fmt.Sprintf("%v/%#x", test.name, test.flag)
		t.Run(name, func(t *testing.T) {
			f, err := root.OpenFile(test.name, test.flag, 0o666)
			if err == nil {
				f.Close()
			}
			if got := err != nil; got != test.wantErr {
				t.Fatalf("root.OpenFile(%q, %#x) = %v, want error: %v", test.name, test.flag, err, test.wantErr)
			}
			if !test.wantErr {
				return
			}
			// Match the error os.OpenFile returns.
			wantF, wantErr := os.OpenFile(filepath.Join(dir, test.name), test.flag, 0o666)
			if wantErr == nil {
				wantF.Close()
				t.Fatalf("os.OpenFile(%q, %#x) unexpectedly succeeded", test.name, test.flag)
			}
			pe, ok := err.(*os.PathError)
			wantPE, wantOK := wantErr.(*os.PathError)
			if !ok || !wantOK || pe.Err != wantPE.Err {
				t.Errorf("root.OpenFile(%q, %#x) error = %v, want %v", test.name, test.flag, err, wantErr)
			}
			// Nothing may have been created or modified through the link.
			if _, err := os.Lstat(filepath.Join(dir, "missing")); err == nil {
				t.Errorf("root.OpenFile(%q, %#x) created the link target", test.name, test.flag)
			}
			if got, err := os.ReadFile(filepath.Join(dir, "target")); err != nil || string(got) != "intact" {
				t.Errorf("target = %q, %v; want %q", got, err, "intact")
			}
		})
	}
}

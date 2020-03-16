// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"internal/testenv"
	"io/ioutil"
	. "os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
)

var isReadonlyError = func(error) bool { return false }

func TestMkdirAll(t *testing.T) {
	tmpDir := TempDir()
	path := tmpDir + "/_TestMkdirAll_/dir/./dir2"
	err := MkdirAll(path, 0777)
	if err != nil {
		t.Fatalf("MkdirAll %q: %s", path, err)
	}
	defer RemoveAll(tmpDir + "/_TestMkdirAll_")

	// Already exists, should succeed.
	err = MkdirAll(path, 0777)
	if err != nil {
		t.Fatalf("MkdirAll %q (second time): %s", path, err)
	}

	// Make file.
	fpath := path + "/file"
	f, err := Create(fpath)
	if err != nil {
		t.Fatalf("create %q: %s", fpath, err)
	}
	defer f.Close()

	// Can't make directory named after file.
	err = MkdirAll(fpath, 0777)
	if err == nil {
		t.Fatalf("MkdirAll %q: no error", fpath)
	}
	perr, ok := err.(*PathError)
	if !ok {
		t.Fatalf("MkdirAll %q returned %T, not *PathError", fpath, err)
	}
	if filepath.Clean(perr.Path) != filepath.Clean(fpath) {
		t.Fatalf("MkdirAll %q returned wrong error path: %q not %q", fpath, filepath.Clean(perr.Path), filepath.Clean(fpath))
	}

	// Can't make subdirectory of file.
	ffpath := fpath + "/subdir"
	err = MkdirAll(ffpath, 0777)
	if err == nil {
		t.Fatalf("MkdirAll %q: no error", ffpath)
	}
	perr, ok = err.(*PathError)
	if !ok {
		t.Fatalf("MkdirAll %q returned %T, not *PathError", ffpath, err)
	}
	if filepath.Clean(perr.Path) != filepath.Clean(fpath) {
		t.Fatalf("MkdirAll %q returned wrong error path: %q not %q", ffpath, filepath.Clean(perr.Path), filepath.Clean(fpath))
	}

	if runtime.GOOS == "windows" {
		path := tmpDir + `\_TestMkdirAll_\dir\.\dir2\`
		err := MkdirAll(path, 0777)
		if err != nil {
			t.Fatalf("MkdirAll %q: %s", path, err)
		}
	}
}

func TestMkdirAllWithSymlink(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tmpDir, err := ioutil.TempDir("", "TestMkdirAllWithSymlink-")
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(tmpDir)

	dir := tmpDir + "/dir"
	err = Mkdir(dir, 0755)
	if err != nil {
		t.Fatalf("Mkdir %s: %s", dir, err)
	}

	link := tmpDir + "/link"
	err = Symlink("dir", link)
	if err != nil {
		t.Fatalf("Symlink %s: %s", link, err)
	}

	path := link + "/foo"
	err = MkdirAll(path, 0755)
	if err != nil {
		t.Errorf("MkdirAll %q: %s", path, err)
	}
}

func TestMkdirAllAtSlash(t *testing.T) {
	switch runtime.GOOS {
	case "android", "plan9", "windows":
		t.Skipf("skipping on %s", runtime.GOOS)
	case "darwin":
		switch runtime.GOARCH {
		case "arm", "arm64":
			t.Skipf("skipping on darwin/%s, mkdir returns EPERM", runtime.GOARCH)
		}
	}
	RemoveAll("/_go_os_test")
	const dir = "/_go_os_test/dir"
	err := MkdirAll(dir, 0777)
	if err != nil {
		pathErr, ok := err.(*PathError)
		// common for users not to be able to write to /
		if ok && (pathErr.Err == syscall.EACCES || isReadonlyError(pathErr.Err)) {
			t.Skipf("could not create %v: %v", dir, err)
		}
		t.Fatalf(`MkdirAll "/_go_os_test/dir": %v, %s`, err, pathErr.Err)
	}
	RemoveAll("/_go_os_test")
}

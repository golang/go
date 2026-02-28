// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"internal/testenv"
	. "os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
)

var isReadonlyError = func(error) bool { return false }

func TestMkdirAll(t *testing.T) {
	testMaybeRooted(t, func(t *testing.T, r *Root) {
		mkdirAll := MkdirAll
		create := Create
		if r != nil {
			mkdirAll = r.MkdirAll
			create = r.Create
		}

		path := "_TestMkdirAll_/dir/./dir2"
		err := mkdirAll(path, 0777)
		if err != nil {
			t.Fatalf("MkdirAll %q: %s", path, err)
		}

		// Already exists, should succeed.
		err = mkdirAll(path, 0777)
		if err != nil {
			t.Fatalf("MkdirAll %q (second time): %s", path, err)
		}

		// Make file.
		fpath := path + "/file"
		f, err := create(fpath)
		if err != nil {
			t.Fatalf("create %q: %s", fpath, err)
		}
		defer f.Close()

		// Can't make directory named after file.
		err = mkdirAll(fpath, 0777)
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
		err = mkdirAll(ffpath, 0777)
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
			path := `_TestMkdirAll_\dir\.\dir2\`
			err := mkdirAll(path, 0777)
			if err != nil {
				t.Fatalf("MkdirAll %q: %s", path, err)
			}
		}
	})
}

func TestMkdirAllAbsPath(t *testing.T) {
	t.Parallel()
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "/a/b/c")
	if err := MkdirAll(path, 0o777); err != nil {
		t.Fatal(err)
	}
	st, err := Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !st.IsDir() {
		t.Fatalf("after MkdirAll(%q, 0o777), %q is not a directory", path, path)
	}
}

func TestMkdirAllWithSymlink(t *testing.T) {
	testenv.MustHaveSymlink(t)
	t.Parallel()

	tmpDir := t.TempDir()
	dir := tmpDir + "/dir"
	if err := Mkdir(dir, 0755); err != nil {
		t.Fatalf("Mkdir %s: %s", dir, err)
	}

	link := tmpDir + "/link"
	if err := Symlink("dir", link); err != nil {
		t.Fatalf("Symlink %s: %s", link, err)
	}

	path := link + "/foo"
	if err := MkdirAll(path, 0755); err != nil {
		t.Errorf("MkdirAll %q: %s", path, err)
	}
}

func TestMkdirAllAtSlash(t *testing.T) {
	switch runtime.GOOS {
	case "android", "ios", "plan9", "windows":
		t.Skipf("skipping on %s", runtime.GOOS)
	}
	if testenv.Builder() == "" {
		t.Skipf("skipping non-hermetic test outside of Go builders")
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

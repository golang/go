// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// testStatAndLstat verifies that all os.Stat, os.Lstat os.File.Stat and os.Readdir work.
func testStatAndLstat(t *testing.T, path string, isLink bool, statCheck, lstatCheck func(*testing.T, string, fs.FileInfo)) {
	// test os.Stat
	sfi, err := os.Stat(path)
	if err != nil {
		t.Error(err)
		return
	}
	statCheck(t, path, sfi)

	// test os.Lstat
	lsfi, err := os.Lstat(path)
	if err != nil {
		t.Error(err)
		return
	}
	lstatCheck(t, path, lsfi)

	if isLink {
		if os.SameFile(sfi, lsfi) {
			t.Errorf("stat and lstat of %q should not be the same", path)
		}
	} else {
		if !os.SameFile(sfi, lsfi) {
			t.Errorf("stat and lstat of %q should be the same", path)
		}
	}

	// test os.File.Stat
	f, err := os.Open(path)
	if err != nil {
		t.Error(err)
		return
	}
	defer f.Close()

	sfi2, err := f.Stat()
	if err != nil {
		t.Error(err)
		return
	}
	statCheck(t, path, sfi2)

	if !os.SameFile(sfi, sfi2) {
		t.Errorf("stat of open %q file and stat of %q should be the same", path, path)
	}

	if isLink {
		if os.SameFile(sfi2, lsfi) {
			t.Errorf("stat of opened %q file and lstat of %q should not be the same", path, path)
		}
	} else {
		if !os.SameFile(sfi2, lsfi) {
			t.Errorf("stat of opened %q file and lstat of %q should be the same", path, path)
		}
	}

	// test fs.FileInfo returned by os.Readdir
	if len(path) > 0 && os.IsPathSeparator(path[len(path)-1]) {
		// skip os.Readdir test of directories with slash at the end
		return
	}
	parentdir := filepath.Dir(path)
	parent, err := os.Open(parentdir)
	if err != nil {
		t.Error(err)
		return
	}
	defer parent.Close()

	fis, err := parent.Readdir(-1)
	if err != nil {
		t.Error(err)
		return
	}
	var lsfi2 fs.FileInfo
	base := filepath.Base(path)
	for _, fi2 := range fis {
		if fi2.Name() == base {
			lsfi2 = fi2
			break
		}
	}
	if lsfi2 == nil {
		t.Errorf("failed to find %q in its parent", path)
		return
	}
	lstatCheck(t, path, lsfi2)

	if !os.SameFile(lsfi, lsfi2) {
		t.Errorf("lstat of %q file in %q directory and %q should be the same", lsfi2.Name(), parentdir, path)
	}
}

// testIsDir verifies that fi refers to directory.
func testIsDir(t *testing.T, path string, fi fs.FileInfo) {
	t.Helper()
	if !fi.IsDir() {
		t.Errorf("%q should be a directory", path)
	}
	if fi.Mode()&fs.ModeSymlink != 0 {
		t.Errorf("%q should not be a symlink", path)
	}
}

// testIsSymlink verifies that fi refers to symlink.
func testIsSymlink(t *testing.T, path string, fi fs.FileInfo) {
	t.Helper()
	if fi.IsDir() {
		t.Errorf("%q should not be a directory", path)
	}
	if fi.Mode()&fs.ModeSymlink == 0 {
		t.Errorf("%q should be a symlink", path)
	}
}

// testIsFile verifies that fi refers to file.
func testIsFile(t *testing.T, path string, fi fs.FileInfo) {
	t.Helper()
	if fi.IsDir() {
		t.Errorf("%q should not be a directory", path)
	}
	if fi.Mode()&fs.ModeSymlink != 0 {
		t.Errorf("%q should not be a symlink", path)
	}
}

func testDirStats(t *testing.T, path string) {
	testStatAndLstat(t, path, false, testIsDir, testIsDir)
}

func testFileStats(t *testing.T, path string) {
	testStatAndLstat(t, path, false, testIsFile, testIsFile)
}

func testSymlinkStats(t *testing.T, path string, isdir bool) {
	if isdir {
		testStatAndLstat(t, path, true, testIsDir, testIsSymlink)
	} else {
		testStatAndLstat(t, path, true, testIsFile, testIsSymlink)
	}
}

func testSymlinkSameFile(t *testing.T, path, link string) {
	pathfi, err := os.Stat(path)
	if err != nil {
		t.Error(err)
		return
	}

	linkfi, err := os.Stat(link)
	if err != nil {
		t.Error(err)
		return
	}
	if !os.SameFile(pathfi, linkfi) {
		t.Errorf("os.Stat(%q) and os.Stat(%q) are not the same file", path, link)
	}

	linkfi, err = os.Lstat(link)
	if err != nil {
		t.Error(err)
		return
	}
	if os.SameFile(pathfi, linkfi) {
		t.Errorf("os.Stat(%q) and os.Lstat(%q) are the same file", path, link)
	}
}

func TestDirAndSymlinkStats(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tmpdir, err := os.MkdirTemp("", "TestDirAndSymlinkStats")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	dir := filepath.Join(tmpdir, "dir")
	err = os.Mkdir(dir, 0777)
	if err != nil {
		t.Fatal(err)
	}
	testDirStats(t, dir)

	dirlink := filepath.Join(tmpdir, "link")
	err = os.Symlink(dir, dirlink)
	if err != nil {
		t.Fatal(err)
	}
	testSymlinkStats(t, dirlink, true)
	testSymlinkSameFile(t, dir, dirlink)

	linklink := filepath.Join(tmpdir, "linklink")
	err = os.Symlink(dirlink, linklink)
	if err != nil {
		t.Fatal(err)
	}
	testSymlinkStats(t, linklink, true)
	testSymlinkSameFile(t, dir, linklink)
}

func TestFileAndSymlinkStats(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tmpdir, err := os.MkdirTemp("", "TestFileAndSymlinkStats")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	file := filepath.Join(tmpdir, "file")
	err = os.WriteFile(file, []byte(""), 0644)
	if err != nil {
		t.Fatal(err)
	}
	testFileStats(t, file)

	filelink := filepath.Join(tmpdir, "link")
	err = os.Symlink(file, filelink)
	if err != nil {
		t.Fatal(err)
	}
	testSymlinkStats(t, filelink, false)
	testSymlinkSameFile(t, file, filelink)

	linklink := filepath.Join(tmpdir, "linklink")
	err = os.Symlink(filelink, linklink)
	if err != nil {
		t.Fatal(err)
	}
	testSymlinkStats(t, linklink, false)
	testSymlinkSameFile(t, file, linklink)
}

// see issue 27225 for details
func TestSymlinkWithTrailingSlash(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tmpdir, err := os.MkdirTemp("", "TestSymlinkWithTrailingSlash")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	dir := filepath.Join(tmpdir, "dir")
	err = os.Mkdir(dir, 0777)
	if err != nil {
		t.Fatal(err)
	}
	dirlink := filepath.Join(tmpdir, "link")
	err = os.Symlink(dir, dirlink)
	if err != nil {
		t.Fatal(err)
	}
	dirlinkWithSlash := dirlink + string(os.PathSeparator)

	if runtime.GOOS == "windows" {
		testSymlinkStats(t, dirlinkWithSlash, true)
	} else {
		testDirStats(t, dirlinkWithSlash)
	}

	fi1, err := os.Stat(dir)
	if err != nil {
		t.Error(err)
		return
	}
	fi2, err := os.Stat(dirlinkWithSlash)
	if err != nil {
		t.Error(err)
		return
	}
	if !os.SameFile(fi1, fi2) {
		t.Errorf("os.Stat(%q) and os.Stat(%q) are not the same file", dir, dirlinkWithSlash)
	}
}

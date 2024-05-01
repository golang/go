// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"errors"
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

type testStatAndLstatParams struct {
	isLink     bool
	statCheck  func(*testing.T, string, fs.FileInfo)
	lstatCheck func(*testing.T, string, fs.FileInfo)
}

// testStatAndLstat verifies that all os.Stat, os.Lstat os.File.Stat and os.Readdir work.
func testStatAndLstat(t *testing.T, path string, params testStatAndLstatParams) {
	// test os.Stat
	sfi, err := os.Stat(path)
	if err != nil {
		t.Error(err)
		return
	}
	params.statCheck(t, path, sfi)

	// test os.Lstat
	lsfi, err := os.Lstat(path)
	if err != nil {
		t.Error(err)
		return
	}
	params.lstatCheck(t, path, lsfi)

	if params.isLink {
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
	params.statCheck(t, path, sfi2)

	if !os.SameFile(sfi, sfi2) {
		t.Errorf("stat of open %q file and stat of %q should be the same", path, path)
	}

	if params.isLink {
		if os.SameFile(sfi2, lsfi) {
			t.Errorf("stat of opened %q file and lstat of %q should not be the same", path, path)
		}
	} else {
		if !os.SameFile(sfi2, lsfi) {
			t.Errorf("stat of opened %q file and lstat of %q should be the same", path, path)
		}
	}

	parentdir, base := filepath.Split(path)
	if parentdir == "" || base == "" {
		// skip os.Readdir test of files without directory or file name component,
		// such as directories with slash at the end or Windows device names.
		return
	}

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
	params.lstatCheck(t, path, lsfi2)

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
	params := testStatAndLstatParams{
		isLink:     false,
		statCheck:  testIsDir,
		lstatCheck: testIsDir,
	}
	testStatAndLstat(t, path, params)
}

func testFileStats(t *testing.T, path string) {
	params := testStatAndLstatParams{
		isLink:     false,
		statCheck:  testIsFile,
		lstatCheck: testIsFile,
	}
	testStatAndLstat(t, path, params)
}

func testSymlinkStats(t *testing.T, path string, isdir bool) {
	params := testStatAndLstatParams{
		isLink:     true,
		lstatCheck: testIsSymlink,
	}
	if isdir {
		params.statCheck = testIsDir
	} else {
		params.statCheck = testIsFile
	}
	testStatAndLstat(t, path, params)
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

func testSymlinkSameFileOpen(t *testing.T, link string) {
	f, err := os.Open(link)
	if err != nil {
		t.Error(err)
		return
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		t.Error(err)
		return
	}

	fi2, err := os.Stat(link)
	if err != nil {
		t.Error(err)
		return
	}

	if !os.SameFile(fi, fi2) {
		t.Errorf("os.Open(%q).Stat() and os.Stat(%q) are not the same file", link, link)
	}
}

func TestDirAndSymlinkStats(t *testing.T) {
	testenv.MustHaveSymlink(t)
	t.Parallel()

	tmpdir := t.TempDir()
	dir := filepath.Join(tmpdir, "dir")
	if err := os.Mkdir(dir, 0777); err != nil {
		t.Fatal(err)
	}
	testDirStats(t, dir)

	dirlink := filepath.Join(tmpdir, "link")
	if err := os.Symlink(dir, dirlink); err != nil {
		t.Fatal(err)
	}
	testSymlinkStats(t, dirlink, true)
	testSymlinkSameFile(t, dir, dirlink)
	testSymlinkSameFileOpen(t, dirlink)

	linklink := filepath.Join(tmpdir, "linklink")
	if err := os.Symlink(dirlink, linklink); err != nil {
		t.Fatal(err)
	}
	testSymlinkStats(t, linklink, true)
	testSymlinkSameFile(t, dir, linklink)
	testSymlinkSameFileOpen(t, linklink)
}

func TestFileAndSymlinkStats(t *testing.T) {
	testenv.MustHaveSymlink(t)
	t.Parallel()

	tmpdir := t.TempDir()
	file := filepath.Join(tmpdir, "file")
	if err := os.WriteFile(file, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}
	testFileStats(t, file)

	filelink := filepath.Join(tmpdir, "link")
	if err := os.Symlink(file, filelink); err != nil {
		t.Fatal(err)
	}
	testSymlinkStats(t, filelink, false)
	testSymlinkSameFile(t, file, filelink)
	testSymlinkSameFileOpen(t, filelink)

	linklink := filepath.Join(tmpdir, "linklink")
	if err := os.Symlink(filelink, linklink); err != nil {
		t.Fatal(err)
	}
	testSymlinkStats(t, linklink, false)
	testSymlinkSameFile(t, file, linklink)
	testSymlinkSameFileOpen(t, linklink)
}

// see issue 27225 for details
func TestSymlinkWithTrailingSlash(t *testing.T) {
	testenv.MustHaveSymlink(t)
	t.Parallel()

	tmpdir := t.TempDir()
	dir := filepath.Join(tmpdir, "dir")
	if err := os.Mkdir(dir, 0777); err != nil {
		t.Fatal(err)
	}
	dirlink := filepath.Join(tmpdir, "link")
	if err := os.Symlink(dir, dirlink); err != nil {
		t.Fatal(err)
	}
	dirlinkWithSlash := dirlink + string(os.PathSeparator)

	testDirStats(t, dirlinkWithSlash)

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

func TestStatConsole(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("skipping on non-Windows")
	}
	t.Parallel()
	consoleNames := []string{
		"CONIN$",
		"CONOUT$",
		"CON",
	}
	for _, name := range consoleNames {
		params := testStatAndLstatParams{
			isLink:     false,
			statCheck:  testIsFile,
			lstatCheck: testIsFile,
		}
		testStatAndLstat(t, name, params)
		testStatAndLstat(t, `\\.\`+name, params)
	}
}

func TestClosedStat(t *testing.T) {
	// Historically we do not seem to match ErrClosed on non-Unix systems.
	switch runtime.GOOS {
	case "windows", "plan9":
		t.Skipf("skipping on %s", runtime.GOOS)
	}

	t.Parallel()
	f, err := os.Open("testdata/hello")
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	_, err = f.Stat()
	if err == nil {
		t.Error("Stat succeeded on closed File")
	} else if !errors.Is(err, os.ErrClosed) {
		t.Errorf("error from Stat on closed file did not match ErrClosed: %q, type %T", err, err)
	}
}

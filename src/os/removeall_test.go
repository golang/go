// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"errors"
	"fmt"
	"io/ioutil"
	. "os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRemoveAll(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "TestRemoveAll-")
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(tmpDir)

	if err := RemoveAll(""); err != nil {
		t.Errorf("RemoveAll(\"\"): %v; want nil", err)
	}

	file := filepath.Join(tmpDir, "file")
	path := filepath.Join(tmpDir, "_TestRemoveAll_")
	fpath := filepath.Join(path, "file")
	dpath := filepath.Join(path, "dir")

	// Make a regular file and remove
	fd, err := Create(file)
	if err != nil {
		t.Fatalf("create %q: %s", file, err)
	}
	fd.Close()
	if err = RemoveAll(file); err != nil {
		t.Fatalf("RemoveAll %q (first): %s", file, err)
	}
	if _, err = Lstat(file); err == nil {
		t.Fatalf("Lstat %q succeeded after RemoveAll (first)", file)
	}

	// Make directory with 1 file and remove.
	if err := MkdirAll(path, 0777); err != nil {
		t.Fatalf("MkdirAll %q: %s", path, err)
	}
	fd, err = Create(fpath)
	if err != nil {
		t.Fatalf("create %q: %s", fpath, err)
	}
	fd.Close()
	if err = RemoveAll(path); err != nil {
		t.Fatalf("RemoveAll %q (second): %s", path, err)
	}
	if _, err = Lstat(path); err == nil {
		t.Fatalf("Lstat %q succeeded after RemoveAll (second)", path)
	}

	// Make directory with file and subdirectory and remove.
	if err = MkdirAll(dpath, 0777); err != nil {
		t.Fatalf("MkdirAll %q: %s", dpath, err)
	}
	fd, err = Create(fpath)
	if err != nil {
		t.Fatalf("create %q: %s", fpath, err)
	}
	fd.Close()
	fd, err = Create(dpath + "/file")
	if err != nil {
		t.Fatalf("create %q: %s", fpath, err)
	}
	fd.Close()
	if err = RemoveAll(path); err != nil {
		t.Fatalf("RemoveAll %q (third): %s", path, err)
	}
	if _, err := Lstat(path); err == nil {
		t.Fatalf("Lstat %q succeeded after RemoveAll (third)", path)
	}

	// Chmod is not supported under Windows and test fails as root.
	if runtime.GOOS != "windows" && Getuid() != 0 {
		// Make directory with file and subdirectory and trigger error.
		if err = MkdirAll(dpath, 0777); err != nil {
			t.Fatalf("MkdirAll %q: %s", dpath, err)
		}

		for _, s := range []string{fpath, dpath + "/file1", path + "/zzz"} {
			fd, err = Create(s)
			if err != nil {
				t.Fatalf("create %q: %s", s, err)
			}
			fd.Close()
		}
		if err = Chmod(dpath, 0); err != nil {
			t.Fatalf("Chmod %q 0: %s", dpath, err)
		}

		// No error checking here: either RemoveAll
		// will or won't be able to remove dpath;
		// either way we want to see if it removes fpath
		// and path/zzz. Reasons why RemoveAll might
		// succeed in removing dpath as well include:
		//	* running as root
		//	* running on a file system without permissions (FAT)
		RemoveAll(path)
		Chmod(dpath, 0777)

		for _, s := range []string{fpath, path + "/zzz"} {
			if _, err = Lstat(s); err == nil {
				t.Fatalf("Lstat %q succeeded after partial RemoveAll", s)
			}
		}
	}
	if err = RemoveAll(path); err != nil {
		t.Fatalf("RemoveAll %q after partial RemoveAll: %s", path, err)
	}
	if _, err = Lstat(path); err == nil {
		t.Fatalf("Lstat %q succeeded after RemoveAll (final)", path)
	}
}

// Test RemoveAll on a large directory.
func TestRemoveAllLarge(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	tmpDir, err := ioutil.TempDir("", "TestRemoveAll-")
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "_TestRemoveAllLarge_")

	// Make directory with 1000 files and remove.
	if err := MkdirAll(path, 0777); err != nil {
		t.Fatalf("MkdirAll %q: %s", path, err)
	}
	for i := 0; i < 1000; i++ {
		fpath := fmt.Sprintf("%s/file%d", path, i)
		fd, err := Create(fpath)
		if err != nil {
			t.Fatalf("create %q: %s", fpath, err)
		}
		fd.Close()
	}
	if err := RemoveAll(path); err != nil {
		t.Fatalf("RemoveAll %q: %s", path, err)
	}
	if _, err := Lstat(path); err == nil {
		t.Fatalf("Lstat %q succeeded after RemoveAll", path)
	}
}

func TestRemoveAllLongPath(t *testing.T) {
	switch runtime.GOOS {
	case "aix", "darwin", "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "illumos", "solaris":
		break
	default:
		t.Skip("skipping for not implemented platforms")
	}

	prevDir, err := Getwd()
	if err != nil {
		t.Fatalf("Could not get wd: %s", err)
	}

	startPath, err := ioutil.TempDir("", "TestRemoveAllLongPath-")
	if err != nil {
		t.Fatalf("Could not create TempDir: %s", err)
	}
	defer RemoveAll(startPath)

	err = Chdir(startPath)
	if err != nil {
		t.Fatalf("Could not chdir %s: %s", startPath, err)
	}

	// Removing paths with over 4096 chars commonly fails
	for i := 0; i < 41; i++ {
		name := strings.Repeat("a", 100)

		err = Mkdir(name, 0755)
		if err != nil {
			t.Fatalf("Could not mkdir %s: %s", name, err)
		}

		err = Chdir(name)
		if err != nil {
			t.Fatalf("Could not chdir %s: %s", name, err)
		}
	}

	err = Chdir(prevDir)
	if err != nil {
		t.Fatalf("Could not chdir %s: %s", prevDir, err)
	}

	err = RemoveAll(startPath)
	if err != nil {
		t.Errorf("RemoveAll could not remove long file path %s: %s", startPath, err)
	}
}

func TestRemoveAllDot(t *testing.T) {
	prevDir, err := Getwd()
	if err != nil {
		t.Fatalf("Could not get wd: %s", err)
	}
	tempDir, err := ioutil.TempDir("", "TestRemoveAllDot-")
	if err != nil {
		t.Fatalf("Could not create TempDir: %s", err)
	}
	defer RemoveAll(tempDir)

	err = Chdir(tempDir)
	if err != nil {
		t.Fatalf("Could not chdir to tempdir: %s", err)
	}

	err = RemoveAll(".")
	if err == nil {
		t.Errorf("RemoveAll succeed to remove .")
	}

	err = Chdir(prevDir)
	if err != nil {
		t.Fatalf("Could not chdir %s: %s", prevDir, err)
	}
}

func TestRemoveAllDotDot(t *testing.T) {
	t.Parallel()

	tempDir, err := ioutil.TempDir("", "TestRemoveAllDotDot-")
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(tempDir)

	subdir := filepath.Join(tempDir, "x")
	subsubdir := filepath.Join(subdir, "y")
	if err := MkdirAll(subsubdir, 0777); err != nil {
		t.Fatal(err)
	}
	if err := RemoveAll(filepath.Join(subsubdir, "..")); err != nil {
		t.Error(err)
	}
	for _, dir := range []string{subsubdir, subdir} {
		if _, err := Stat(dir); err == nil {
			t.Errorf("%s: exists after RemoveAll", dir)
		}
	}
}

// Issue #29178.
func TestRemoveReadOnlyDir(t *testing.T) {
	t.Parallel()

	tempDir, err := ioutil.TempDir("", "TestRemoveReadOnlyDir-")
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(tempDir)

	subdir := filepath.Join(tempDir, "x")
	if err := Mkdir(subdir, 0); err != nil {
		t.Fatal(err)
	}

	// If an error occurs make it more likely that removing the
	// temporary directory will succeed.
	defer Chmod(subdir, 0777)

	if err := RemoveAll(subdir); err != nil {
		t.Fatal(err)
	}

	if _, err := Stat(subdir); err == nil {
		t.Error("subdirectory was not removed")
	}
}

// Issue #29983.
func TestRemoveAllButReadOnlyAndPathError(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "js", "windows":
		t.Skipf("skipping test on %s", runtime.GOOS)
	}

	if Getuid() == 0 {
		t.Skip("skipping test when running as root")
	}

	t.Parallel()

	tempDir, err := ioutil.TempDir("", "TestRemoveAllButReadOnly-")
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(tempDir)

	dirs := []string{
		"a",
		"a/x",
		"a/x/1",
		"b",
		"b/y",
		"b/y/2",
		"c",
		"c/z",
		"c/z/3",
	}
	readonly := []string{
		"b",
	}
	inReadonly := func(d string) bool {
		for _, ro := range readonly {
			if d == ro {
				return true
			}
			dd, _ := filepath.Split(d)
			if filepath.Clean(dd) == ro {
				return true
			}
		}
		return false
	}

	for _, dir := range dirs {
		if err := Mkdir(filepath.Join(tempDir, dir), 0777); err != nil {
			t.Fatal(err)
		}
	}
	for _, dir := range readonly {
		d := filepath.Join(tempDir, dir)
		if err := Chmod(d, 0555); err != nil {
			t.Fatal(err)
		}

		// Defer changing the mode back so that the deferred
		// RemoveAll(tempDir) can succeed.
		defer Chmod(d, 0777)
	}

	err = RemoveAll(tempDir)
	if err == nil {
		t.Fatal("RemoveAll succeeded unexpectedly")
	}

	// The error should be of type *PathError.
	// see issue 30491 for details.
	if pathErr, ok := err.(*PathError); ok {
		if g, w := pathErr.Path, filepath.Join(tempDir, "b", "y"); g != w {
			t.Errorf("got %q, expected pathErr.path %q", g, w)
		}
	} else {
		t.Errorf("got %T, expected *os.PathError", err)
	}

	for _, dir := range dirs {
		_, err := Stat(filepath.Join(tempDir, dir))
		if inReadonly(dir) {
			if err != nil {
				t.Errorf("file %q was deleted but should still exist", dir)
			}
		} else {
			if err == nil {
				t.Errorf("file %q still exists but should have been deleted", dir)
			}
		}
	}
}

func TestRemoveUnreadableDir(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "js", "windows":
		t.Skipf("skipping test on %s", runtime.GOOS)
	}

	if Getuid() == 0 {
		t.Skip("skipping test when running as root")
	}

	t.Parallel()

	tempDir, err := ioutil.TempDir("", "TestRemoveAllButReadOnly-")
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(tempDir)

	target := filepath.Join(tempDir, "d0", "d1", "d2")
	if err := MkdirAll(target, 0755); err != nil {
		t.Fatal(err)
	}
	if err := Chmod(target, 0300); err != nil {
		t.Fatal(err)
	}
	if err := RemoveAll(filepath.Join(tempDir, "d0")); err != nil {
		t.Fatal(err)
	}
}

// Issue 29921
func TestRemoveAllWithMoreErrorThanReqSize(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	defer func(oldHook func(error) error) {
		*RemoveAllTestHook = oldHook
	}(*RemoveAllTestHook)

	*RemoveAllTestHook = func(err error) error {
		return errors.New("error from RemoveAllTestHook")
	}

	tmpDir, err := ioutil.TempDir("", "TestRemoveAll-")
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(tmpDir)

	path := filepath.Join(tmpDir, "_TestRemoveAllWithMoreErrorThanReqSize_")

	// Make directory with 1025 files and remove.
	if err := MkdirAll(path, 0777); err != nil {
		t.Fatalf("MkdirAll %q: %s", path, err)
	}
	for i := 0; i < 1025; i++ {
		fpath := filepath.Join(path, fmt.Sprintf("file%d", i))
		fd, err := Create(fpath)
		if err != nil {
			t.Fatalf("create %q: %s", fpath, err)
		}
		fd.Close()
	}

	// This call should not hang
	if err := RemoveAll(path); err == nil {
		t.Fatal("Want error from RemoveAllTestHook, got nil")
	}

	// We hook to inject error, but the actual files must be deleted
	if _, err := Lstat(path); err == nil {
		t.Fatal("directory must be deleted even with removeAllTetHook run")
	}
}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
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

	// Determine if we should run the following test.
	testit := true
	if runtime.GOOS == "windows" {
		// Chmod is not supported under windows.
		testit = false
	} else {
		// Test fails as root.
		testit = Getuid() != 0
	}
	if testit {
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
	case "aix", "darwin", "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "solaris":
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
	switch runtime.GOOS {
	case "aix", "darwin", "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "solaris":
		break
	default:
		t.Skip("skipping for not implemented platforms")
	}

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

	err = RemoveAll("..")
	if err == nil {
		t.Errorf("RemoveAll succeed to remove ..")
	}

	err = Chdir(prevDir)
	if err != nil {
		t.Fatalf("Could not chdir %s: %s", prevDir, err)
	}
}

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package os_test

import (
	"errors"
	. "os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
)

func TestGetwdDeep(t *testing.T) {
	testGetwdDeep(t, false)
}

func TestGetwdDeepWithPWDSet(t *testing.T) {
	testGetwdDeep(t, true)
}

// testGetwdDeep checks that os.Getwd is able to return paths
// longer than syscall.PathMax (with or without PWD set).
func testGetwdDeep(t *testing.T, setPWD bool) {
	tempDir := t.TempDir()

	dir := tempDir
	t.Chdir(dir)

	if setPWD {
		t.Setenv("PWD", dir)
	} else {
		// When testing os.Getwd, setting PWD to empty string
		// is the same as unsetting it, but the latter would
		// be more complicated since we don't have t.Unsetenv.
		t.Setenv("PWD", "")
	}

	name := strings.Repeat("a", 200)
	for {
		if err := Mkdir(name, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := Chdir(name); err != nil {
			t.Fatal(err)
		}
		if setPWD {
			dir += "/" + name
			if err := Setenv("PWD", dir); err != nil {
				t.Fatal(err)
			}
			t.Logf(" $PWD len: %d", len(dir))
		}

		wd, err := Getwd()
		t.Logf("Getwd len: %d", len(wd))
		if err != nil {
			// We can get an EACCES error if we can't read up
			// to root, which happens on the Android builders.
			if errors.Is(err, syscall.EACCES) {
				t.Logf("ignoring EACCES error: %v", err)
				break
			}
			t.Fatal(err)
		}
		if setPWD && wd != dir {
			// It's possible for the stat of PWD to fail
			// with ENAMETOOLONG, and for getwd to fail for
			// the same reason, and it's possible for $TMPDIR
			// to contain a symlink. In that case the fallback
			// code will not return the same directory.
			if len(dir) > 1000 {
				symDir, err := filepath.EvalSymlinks(tempDir)
				if err == nil && symDir != tempDir {
					t.Logf("EvalSymlinks(%q) = %q", tempDir, symDir)
					if strings.Replace(dir, tempDir, symDir, 1) == wd {
						// Symlink confusion is OK.
						break
					}
				}
			}

			t.Fatalf("Getwd: got %q, want same value as $PWD: %q", wd, dir)
		}
		// Ideally the success criterion should be len(wd) > syscall.PathMax,
		// but the latter is not public for some platforms, so use Stat(wd).
		// When it fails with ENAMETOOLONG, it means:
		//  - wd is longer than PathMax;
		//  - Getwd have used the slow fallback code.
		//
		// To avoid an endless loop here in case Stat keeps working,
		// check if len(wd) is above the largest known PathMax among
		// all Unix platforms (4096, on Linux).
		if _, err := Stat(wd); err != nil || len(wd) > 4096 {
			t.Logf("Done; len(wd)=%d", len(wd))
			// Most systems return ENAMETOOLONG.
			// Dragonfly returns EFAULT.
			switch {
			case err == nil:
			case errors.Is(err, syscall.ENAMETOOLONG):
			case runtime.GOOS == "dragonfly" && errors.Is(err, syscall.EFAULT):
			default:
				t.Fatalf("unexpected Stat error: %v", err)
			}
			break
		}
	}
}

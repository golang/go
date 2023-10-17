// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec_test

import (
	"errors"
	"internal/testenv"
	"os"
	. "os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var pathVar string = func() string {
	if runtime.GOOS == "plan9" {
		return "path"
	}
	return "PATH"
}()

func TestLookPath(t *testing.T) {
	testenv.MustHaveExec(t)
	// Not parallel: uses Chdir and Setenv.

	tmpDir := filepath.Join(t.TempDir(), "testdir")
	if err := os.Mkdir(tmpDir, 0777); err != nil {
		t.Fatal(err)
	}

	executable := "execabs-test"
	if runtime.GOOS == "windows" {
		executable += ".exe"
	}
	if err := os.WriteFile(filepath.Join(tmpDir, executable), []byte{1, 2, 3}, 0777); err != nil {
		t.Fatal(err)
	}
	chdir(t, tmpDir)
	t.Setenv("PWD", tmpDir)
	t.Logf(". is %#q", tmpDir)

	origPath := os.Getenv(pathVar)

	// Add "." to PATH so that exec.LookPath looks in the current directory on all systems.
	// And try to trick it with "../testdir" too.
	for _, errdot := range []string{"1", "0"} {
		t.Run("GODEBUG=execerrdot="+errdot, func(t *testing.T) {
			t.Setenv("GODEBUG", "execerrdot="+errdot+",execwait=2")
			for _, dir := range []string{".", "../testdir"} {
				t.Run(pathVar+"="+dir, func(t *testing.T) {
					t.Setenv(pathVar, dir+string(filepath.ListSeparator)+origPath)
					good := dir + "/execabs-test"
					if found, err := LookPath(good); err != nil || !strings.HasPrefix(found, good) {
						t.Fatalf(`LookPath(%#q) = %#q, %v, want "%s...", nil`, good, found, err, good)
					}
					if runtime.GOOS == "windows" {
						good = dir + `\execabs-test`
						if found, err := LookPath(good); err != nil || !strings.HasPrefix(found, good) {
							t.Fatalf(`LookPath(%#q) = %#q, %v, want "%s...", nil`, good, found, err, good)
						}
					}

					_, err := LookPath("execabs-test")
					if errdot == "1" {
						if err == nil {
							t.Fatalf("LookPath didn't fail when finding a non-relative path")
						} else if !errors.Is(err, ErrDot) {
							t.Fatalf("LookPath returned unexpected error: want Is ErrDot, got %q", err)
						}
					} else {
						if err != nil {
							t.Fatalf("LookPath failed unexpectedly: %v", err)
						}
					}

					cmd := Command("execabs-test")
					if errdot == "1" {
						if cmd.Err == nil {
							t.Fatalf("Command didn't fail when finding a non-relative path")
						} else if !errors.Is(cmd.Err, ErrDot) {
							t.Fatalf("Command returned unexpected error: want Is ErrDot, got %q", cmd.Err)
						}
						cmd.Err = nil
					} else {
						if cmd.Err != nil {
							t.Fatalf("Command failed unexpectedly: %v", err)
						}
					}

					// Clearing cmd.Err should let the execution proceed,
					// and it should fail because it's not a valid binary.
					if err := cmd.Run(); err == nil {
						t.Fatalf("Run did not fail: expected exec error")
					} else if errors.Is(err, ErrDot) {
						t.Fatalf("Run returned unexpected error ErrDot: want error like ENOEXEC: %q", err)
					}
				})
			}
		})
	}

	// Test the behavior when the first entry in PATH is an absolute name for the
	// current directory.
	//
	// On Windows, "." may or may not be implicitly included before the explicit
	// %PATH%, depending on the process environment;
	// see https://go.dev/issue/4394.
	//
	// If the relative entry from "." resolves to the same executable as what
	// would be resolved from an absolute entry in %PATH% alone, LookPath should
	// return the absolute version of the path instead of ErrDot.
	// (See https://go.dev/issue/53536.)
	//
	// If PATH does not implicitly include "." (such as on Unix platforms, or on
	// Windows configured with NoDefaultCurrentDirectoryInExePath), then this
	// lookup should succeed regardless of the behavior for ".", so it may be
	// useful to run as a control case even on those platforms.
	t.Run(pathVar+"=$PWD", func(t *testing.T) {
		t.Setenv(pathVar, tmpDir+string(filepath.ListSeparator)+origPath)
		good := filepath.Join(tmpDir, "execabs-test")
		if found, err := LookPath(good); err != nil || !strings.HasPrefix(found, good) {
			t.Fatalf(`LookPath(%#q) = %#q, %v, want \"%s...\", nil`, good, found, err, good)
		}

		if found, err := LookPath("execabs-test"); err != nil || !strings.HasPrefix(found, good) {
			t.Fatalf(`LookPath(%#q) = %#q, %v, want \"%s...\", nil`, "execabs-test", found, err, good)
		}

		cmd := Command("execabs-test")
		if cmd.Err != nil {
			t.Fatalf("Command(%#q).Err = %v; want nil", "execabs-test", cmd.Err)
		}
	})

	t.Run(pathVar+"=$OTHER", func(t *testing.T) {
		// Control case: if the lookup returns ErrDot when PATH is empty, then we
		// know that PATH implicitly includes ".". If it does not, then we don't
		// expect to see ErrDot at all in this test (because the path will be
		// unambiguously absolute).
		wantErrDot := false
		t.Setenv(pathVar, "")
		if found, err := LookPath("execabs-test"); errors.Is(err, ErrDot) {
			wantErrDot = true
		} else if err == nil {
			t.Fatalf(`with PATH='', LookPath(%#q) = %#q; want non-nil error`, "execabs-test", found)
		}

		// Set PATH to include an explicit directory that contains a completely
		// independent executable that happens to have the same name as an
		// executable in ".". If "." is included implicitly, looking up the
		// (unqualified) executable name will return ErrDot; otherwise, the
		// executable in "." should have no effect and the lookup should
		// unambiguously resolve to the directory in PATH.

		dir := t.TempDir()
		executable := "execabs-test"
		if runtime.GOOS == "windows" {
			executable += ".exe"
		}
		if err := os.WriteFile(filepath.Join(dir, executable), []byte{1, 2, 3}, 0777); err != nil {
			t.Fatal(err)
		}
		t.Setenv(pathVar, dir+string(filepath.ListSeparator)+origPath)

		found, err := LookPath("execabs-test")
		if wantErrDot {
			wantFound := filepath.Join(".", executable)
			if found != wantFound || !errors.Is(err, ErrDot) {
				t.Fatalf(`LookPath(%#q) = %#q, %v, want %#q, Is ErrDot`, "execabs-test", found, err, wantFound)
			}
		} else {
			wantFound := filepath.Join(dir, executable)
			if found != wantFound || err != nil {
				t.Fatalf(`LookPath(%#q) = %#q, %v, want %#q, nil`, "execabs-test", found, err, wantFound)
			}
		}
	})
}

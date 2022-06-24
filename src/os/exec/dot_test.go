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

func TestLookPath(t *testing.T) {
	testenv.MustHaveExec(t)

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
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.Chdir(cwd); err != nil {
			panic(err)
		}
	}()
	if err = os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}
	origPath := os.Getenv("PATH")
	defer os.Setenv("PATH", origPath)

	// Add "." to PATH so that exec.LookPath looks in the current directory on all systems.
	// And try to trick it with "../testdir" too.
	for _, dir := range []string{".", "../testdir"} {
		os.Setenv("PATH", dir+string(filepath.ListSeparator)+origPath)
		t.Run("PATH="+dir, func(t *testing.T) {
			good := dir + "/execabs-test"
			if found, err := LookPath(good); err != nil || !strings.HasPrefix(found, good) {
				t.Fatalf("LookPath(%q) = %q, %v, want \"%s...\", nil", good, found, err, good)
			}
			if runtime.GOOS == "windows" {
				good = dir + `\execabs-test`
				if found, err := LookPath(good); err != nil || !strings.HasPrefix(found, good) {
					t.Fatalf("LookPath(%q) = %q, %v, want \"%s...\", nil", good, found, err, good)
				}
			}

			if _, err := LookPath("execabs-test"); err == nil {
				t.Fatalf("LookPath didn't fail when finding a non-relative path")
			} else if !errors.Is(err, ErrDot) {
				t.Fatalf("LookPath returned unexpected error: want Is ErrDot, got %q", err)
			}

			cmd := Command("execabs-test")
			if cmd.Err == nil {
				t.Fatalf("Command didn't fail when finding a non-relative path")
			} else if !errors.Is(cmd.Err, ErrDot) {
				t.Fatalf("Command returned unexpected error: want Is ErrDot, got %q", cmd.Err)
			}
			cmd.Err = nil

			// Clearing cmd.Err should let the execution proceed,
			// and it should fail because it's not a valid binary.
			if err := cmd.Run(); err == nil {
				t.Fatalf("Run did not fail: expected exec error")
			} else if errors.Is(err, ErrDot) {
				t.Fatalf("Run returned unexpected error ErrDot: want error like ENOEXEC: %q", err)
			}
		})
	}
}

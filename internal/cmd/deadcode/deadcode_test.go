// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.20

package main_test

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/txtar"
)

// Test runs the deadcode command on each scenario
// described by a testdata/*.txtar file.
func Test(t *testing.T) {
	testenv.NeedsTool(t, "go")
	if runtime.GOOS == "android" {
		t.Skipf("the dependencies are not available on android")
	}

	exe := buildDeadcode(t)

	matches, err := filepath.Glob("testdata/*.txtar")
	if err != nil {
		t.Fatal(err)
	}
	for _, filename := range matches {
		filename := filename
		t.Run(filename, func(t *testing.T) {
			t.Parallel()

			ar, err := txtar.ParseFile(filename)
			if err != nil {
				t.Fatal(err)
			}

			// Parse archive comment as directives of these forms:
			//
			//    deadcode args...		command-line arguments
			//  [!]want "quoted"		expected/unwanted string in output
			//
			var args []string
			want := make(map[string]bool) // string -> sense
			for _, line := range strings.Split(string(ar.Comment), "\n") {
				line = strings.TrimSpace(line)
				if line == "" || line[0] == '#' {
					continue // skip blanks and comments
				}

				fields := strings.Fields(line)
				switch kind := fields[0]; kind {
				case "deadcode":
					args = fields[1:] // lossy wrt spaces
				case "want", "!want":
					rest := line[len(kind):]
					str, err := strconv.Unquote(strings.TrimSpace(rest))
					if err != nil {
						t.Fatalf("bad %s directive <<%s>>", kind, line)
					}
					want[str] = kind[0] != '!'
				default:
					t.Fatalf("%s: invalid directive %q", filename, kind)
				}
			}

			// Write the archive files to the temp directory.
			tmpdir := t.TempDir()
			for _, f := range ar.Files {
				filename := filepath.Join(tmpdir, f.Name)
				if err := os.MkdirAll(filepath.Dir(filename), 0777); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filename, f.Data, 0666); err != nil {
					t.Fatal(err)
				}
			}

			// Run the command.
			cmd := exec.Command(exe, args...)
			cmd.Stdout = new(bytes.Buffer)
			cmd.Stderr = new(bytes.Buffer)
			cmd.Dir = tmpdir
			cmd.Env = append(os.Environ(), "GOPROXY=", "GO111MODULE=on")
			if err := cmd.Run(); err != nil {
				t.Fatalf("deadcode failed: %v (stderr=%s)", err, cmd.Stderr)
			}

			// Check each want directive.
			got := fmt.Sprint(cmd.Stdout)
			for str, sense := range want {
				ok := true
				if strings.Contains(got, str) != sense {
					if sense {
						t.Errorf("missing %q", str)
					} else {
						t.Errorf("unwanted %q", str)
					}
					ok = false
				}
				if !ok {
					t.Errorf("got: <<%s>>", got)
				}
			}
		})
	}
}

// buildDeadcode builds the deadcode executable.
// It returns its path, and a cleanup function.
func buildDeadcode(t *testing.T) string {
	bin := filepath.Join(t.TempDir(), "deadcode")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	cmd := exec.Command("go", "build", "-o", bin)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Building deadcode: %v\n%s", err, out)
	}
	return bin
}

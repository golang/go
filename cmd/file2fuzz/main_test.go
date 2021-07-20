// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

// The setup for this test is mostly cribbed from x/exp/txtar.

var buildBin struct {
	once sync.Once
	name string
	err  error
}

func binPath(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("go"); err != nil {
		t.Skipf("cannot build file2fuzz binary: %v", err)
	}

	buildBin.once.Do(func() {
		exe, err := ioutil.TempFile("", "file2fuzz-*.exe")
		if err != nil {
			buildBin.err = err
			return
		}
		exe.Close()
		buildBin.name = exe.Name()

		cmd := exec.Command("go", "build", "-o", buildBin.name, ".")
		out, err := cmd.CombinedOutput()
		if err != nil {
			buildBin.err = fmt.Errorf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, out)
		}
	})

	if buildBin.err != nil {
		if runtime.GOOS == "android" {
			t.Skipf("skipping test after failing to build file2fuzz binary: go_android_exec may have failed to copy needed dependencies (see https://golang.org/issue/37088)")
		}
		t.Fatal(buildBin.err)
	}
	return buildBin.name
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
	if buildBin.name != "" {
		os.Remove(buildBin.name)
	}
}

func file2fuzz(t *testing.T, dir string, args []string, stdin string) (string, bool) {
	t.Helper()
	cmd := exec.Command(binPath(t), args...)
	cmd.Dir = dir
	if stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), true
	}
	return string(out), false
}

func TestFile2Fuzz(t *testing.T) {
	type file struct {
		name    string
		dir     bool
		content string
	}
	tests := []struct {
		name           string
		args           []string
		stdin          string
		inputFiles     []file
		expectedStdout string
		expectedFiles  []file
		expectedError  string
	}{
		{
			name:           "stdin, stdout",
			stdin:          "hello",
			expectedStdout: "go test fuzz v1\n[]byte(\"hello\")",
		},
		{
			name:          "stdin, output file",
			stdin:         "hello",
			args:          []string{"-o", "output"},
			expectedFiles: []file{{name: "output", content: "go test fuzz v1\n[]byte(\"hello\")"}},
		},
		{
			name:          "stdin, output directory",
			stdin:         "hello",
			args:          []string{"-o", "output"},
			inputFiles:    []file{{name: "output", dir: true}},
			expectedFiles: []file{{name: "output/ffc7b87a0377262d4f77926bd235551d78e6037bbe970d81ec39ac1d95542f7b", content: "go test fuzz v1\n[]byte(\"hello\")"}},
		},
		{
			name:          "input file, output file",
			args:          []string{"-o", "output", "input"},
			inputFiles:    []file{{name: "input", content: "hello"}},
			expectedFiles: []file{{name: "output", content: "go test fuzz v1\n[]byte(\"hello\")"}},
		},
		{
			name:          "input file, output directory",
			args:          []string{"-o", "output", "input"},
			inputFiles:    []file{{name: "output", dir: true}, {name: "input", content: "hello"}},
			expectedFiles: []file{{name: "output/ffc7b87a0377262d4f77926bd235551d78e6037bbe970d81ec39ac1d95542f7b", content: "go test fuzz v1\n[]byte(\"hello\")"}},
		},
		{
			name:       "input files, output directory",
			args:       []string{"-o", "output", "input", "input-2"},
			inputFiles: []file{{name: "output", dir: true}, {name: "input", content: "hello"}, {name: "input-2", content: "hello :)"}},
			expectedFiles: []file{
				{name: "output/ffc7b87a0377262d4f77926bd235551d78e6037bbe970d81ec39ac1d95542f7b", content: "go test fuzz v1\n[]byte(\"hello\")"},
				{name: "output/28059db30ce420ff65b2c29b749804c69c601aeca21b3cbf0644244ff080d7a5", content: "go test fuzz v1\n[]byte(\"hello :)\")"},
			},
		},
		{
			name:          "input files, no output",
			args:          []string{"input", "input-2"},
			inputFiles:    []file{{name: "output", dir: true}, {name: "input", content: "hello"}, {name: "input-2", content: "hello :)"}},
			expectedError: "file2fuzz: -o required with multiple input files\n",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tmp, err := ioutil.TempDir(os.TempDir(), "file2fuzz")
			if err != nil {
				t.Fatalf("ioutil.TempDir failed: %s", err)
			}
			defer os.RemoveAll(tmp)
			for _, f := range tc.inputFiles {
				if f.dir {
					if err := os.Mkdir(filepath.Join(tmp, f.name), 0777); err != nil {
						t.Fatalf("failed to create test directory: %s", err)
					}
				} else {
					if err := ioutil.WriteFile(filepath.Join(tmp, f.name), []byte(f.content), 0666); err != nil {
						t.Fatalf("failed to create test input file: %s", err)
					}
				}
			}

			out, failed := file2fuzz(t, tmp, tc.args, tc.stdin)
			if failed && tc.expectedError == "" {
				t.Fatalf("file2fuzz failed unexpectedly: %s", out)
			} else if failed && out != tc.expectedError {
				t.Fatalf("file2fuzz returned unexpected error: got %q, want %q", out, tc.expectedError)
			}
			if !failed && out != tc.expectedStdout {
				t.Fatalf("file2fuzz unexpected stdout: got %q, want %q", out, tc.expectedStdout)
			}

			for _, f := range tc.expectedFiles {
				c, err := ioutil.ReadFile(filepath.Join(tmp, f.name))
				if err != nil {
					t.Fatalf("failed to read expected output file %q: %s", f.name, err)
				}
				if string(c) != f.content {
					t.Fatalf("expected output file %q contains unexpected content: got %s, want %s", f.name, string(c), f.content)
				}
			}
		})
	}
}

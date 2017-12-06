// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errorstest

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

func path(file string) string {
	return filepath.Join("src", file)
}

func check(t *testing.T, file string) {
	t.Run(file, func(t *testing.T) {
		t.Parallel()

		contents, err := ioutil.ReadFile(path(file))
		if err != nil {
			t.Fatal(err)
		}
		var errors []*regexp.Regexp
		for i, line := range bytes.Split(contents, []byte("\n")) {
			if bytes.HasSuffix(line, []byte("ERROR HERE")) {
				re := regexp.MustCompile(regexp.QuoteMeta(fmt.Sprintf("%s:%d:", file, i+1)))
				errors = append(errors, re)
				continue
			}

			frags := bytes.SplitAfterN(line, []byte("ERROR HERE: "), 2)
			if len(frags) == 1 {
				continue
			}
			re, err := regexp.Compile(string(frags[1]))
			if err != nil {
				t.Errorf("Invalid regexp after `ERROR HERE: `: %#q", frags[1])
				continue
			}
			errors = append(errors, re)
		}
		if len(errors) == 0 {
			t.Fatalf("cannot find ERROR HERE")
		}
		expect(t, file, errors)
	})
}

func expect(t *testing.T, file string, errors []*regexp.Regexp) {
	dir, err := ioutil.TempDir("", filepath.Base(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	dst := filepath.Join(dir, strings.TrimSuffix(file, ".go"))
	cmd := exec.Command("go", "build", "-gcflags=-L", "-o="+dst, path(file)) // TODO(gri) no need for -gcflags=-L if go tool is adjusted
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Errorf("expected cgo to fail but it succeeded")
	}

	lines := bytes.Split(out, []byte("\n"))
	for _, re := range errors {
		found := false
		for _, line := range lines {
			if re.Match(line) {
				t.Logf("found match for %#q: %q", re, line)
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected error output to contain %#q", re)
		}
	}

	if t.Failed() {
		t.Logf("actual output:\n%s", out)
	}
}

func sizeofLongDouble(t *testing.T) int {
	cmd := exec.Command("go", "run", path("long_double_size.go"))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%#q: %v:\n%s", strings.Join(cmd.Args, " "), err, out)
	}

	i, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		t.Fatalf("long_double_size.go printed invalid size: %s", out)
	}
	return i
}

func TestReportsTypeErrors(t *testing.T) {
	for _, file := range []string{
		"err1.go",
		"err2.go",
		"err3.go",
		"issue7757.go",
		"issue8442.go",
		"issue11097a.go",
		"issue11097b.go",
		"issue13129.go",
		"issue13423.go",
		"issue13467.go",
		"issue13635.go",
		"issue13830.go",
		"issue16116.go",
		"issue16591.go",
		"issue18452.go",
		"issue18889.go",
	} {
		check(t, file)
	}

	if sizeofLongDouble(t) > 8 {
		check(t, "err4.go")
	}
}

func TestToleratesOptimizationFlag(t *testing.T) {
	for _, cflags := range []string{
		"",
		"-O",
	} {
		cflags := cflags
		t.Run(cflags, func(t *testing.T) {
			t.Parallel()

			cmd := exec.Command("go", "build", path("issue14669.go"))
			cmd.Env = append(os.Environ(), "CGO_CFLAGS="+cflags)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Errorf("%#q: %v:\n%s", strings.Join(cmd.Args, " "), err, out)
			}
		})
	}
}

func TestMallocCrashesOnNil(t *testing.T) {
	t.Parallel()

	cmd := exec.Command("go", "run", path("malloc.go"))
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Logf("%#q:\n%s", strings.Join(cmd.Args, " "), out)
		t.Fatalf("succeeded unexpectedly")
	}
}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestUndefinedRelocErrors(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	out, err := exec.Command(testenv.GoToolPath(t), "build", "./testdata/issue10978").CombinedOutput()
	if err == nil {
		t.Fatal("expected build to fail")
	}

	wantErrors := map[string]int{
		// Main function has dedicated error message.
		"function main is undeclared in the main package": 1,

		// Single error reporting per each symbol.
		// This way, duplicated messages are not reported for
		// multiple relocations with a same name.
		"main.defined1: relocation target main.undefined not defined": 1,
		"main.defined2: relocation target main.undefined not defined": 1,
	}
	unexpectedErrors := map[string]int{}

	for _, l := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(l, "#") || l == "" {
			continue
		}
		matched := ""
		for want := range wantErrors {
			if strings.Contains(l, want) {
				matched = want
				break
			}
		}
		if matched != "" {
			wantErrors[matched]--
		} else {
			unexpectedErrors[l]++
		}
	}

	for want, n := range wantErrors {
		switch {
		case n > 0:
			t.Errorf("unmatched error: %s (x%d)", want, n)
		case n < 0:
			t.Errorf("extra errors: %s (x%d)", want, -n)
		}
	}
	for unexpected, n := range unexpectedErrors {
		t.Errorf("unexpected error: %s (x%d)", unexpected, n)
	}
}

const carchiveSrcText = `
package main

//export GoFunc
func GoFunc() {
	println(42)
}

func main() {
}
`

func TestArchiveBuildInvokeWithExec(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	// run this test on just a small set of platforms (no need to test it
	// across the board given the nature of the test).
	pair := runtime.GOOS + "-" + runtime.GOARCH
	switch pair {
	case "darwin-amd64", "darwin-arm64", "linux-amd64", "freebsd-amd64":
	default:
		t.Skip("no need for test on " + pair)
	}
	switch runtime.GOOS {
	case "openbsd", "windows":
		t.Skip("c-archive unsupported")
	}
	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	srcfile := filepath.Join(dir, "test.go")
	arfile := filepath.Join(dir, "test.a")
	if err := ioutil.WriteFile(srcfile, []byte(carchiveSrcText), 0666); err != nil {
		t.Fatal(err)
	}

	ldf := fmt.Sprintf("-ldflags=-v -tmpdir=%s", dir)
	argv := []string{"build", "-buildmode=c-archive", "-o", arfile, ldf, srcfile}
	out, err := exec.Command(testenv.GoToolPath(t), argv...).CombinedOutput()
	if err != nil {
		t.Fatalf("build failure: %s\n%s\n", err, string(out))
	}

	found := false
	const want = "invoking archiver with syscall.Exec"
	for _, l := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(l, want) {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("expected '%s' in -v output, got:\n%s\n", want, string(out))
	}
}

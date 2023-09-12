// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/build"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Unit tests for internal guru functions

func TestIssue17515(t *testing.T) {
	// Tests handling of symlinks in function guessImportPath
	// If we have Go code inside $HOME/go/src and create a symlink $HOME/src to it
	// there are 4 possible cases that need to be tested:
	// (1) absolute & absolute: GOPATH=$HOME/go/src file=$HOME/go/src/test/test.go
	// (2) absolute & symlink:  GOPATH=$HOME/go/src file=$HOME/src/test/test.go
	// (3) symlink & symlink:   GOPATH=$HOME/src file=$HOME/src/test/test.go
	// (4) symlink & absolute:  GOPATH=$HOME/src file= $HOME/go/src/test/test.go

	// Create a temporary home directory under /tmp
	home, err := os.MkdirTemp(os.TempDir(), "home")
	if err != nil {
		t.Errorf("Unable to create a temporary directory in %s", os.TempDir())
	}

	defer os.RemoveAll(home)

	// create filepath /tmp/home/go/src/test/test.go
	if err = os.MkdirAll(home+"/go/src/test", 0755); err != nil {
		t.Fatal(err)
	}

	var buildContext = build.Default

	// Success test cases
	type SuccessTest struct {
		gopath, filename, wantSrcdir string
	}

	successTests := []SuccessTest{
		{home + "/go", home + "/go/src/test/test.go", filepath.FromSlash(home + "/go/src")},
	}

	// Add symlink cases if not on Windows, Plan 9
	if runtime.GOOS != "windows" && runtime.GOOS != "plan9" {
		// symlink between /tmp/home/go/src and /tmp/home/src
		if err := os.Symlink(home+"/go/src", home+"/src"); err != nil {
			t.Fatal(err)
		}

		successTests = append(successTests, []SuccessTest{
			{home + "/go", home + "/src/test/test.go", filepath.FromSlash(home + "/go/src")},
			{home, home + "/go/src/test/test.go", filepath.FromSlash(home + "/src")},
			{home, home + "/src/test/test.go", filepath.FromSlash(home + "/src")},
		}...)
	}

	for _, test := range successTests {
		buildContext.GOPATH = test.gopath
		srcdir, importPath, err := guessImportPath(test.filename, &buildContext)
		if srcdir != test.wantSrcdir || importPath != "test" || err != nil {
			t.Errorf("guessImportPath(%q, %q) = %q, %q, %q; want %q, %q, %q",
				test.filename, test.gopath, srcdir, importPath, err, test.wantSrcdir, "test", "nil")
		}
	}
	// Function to format expected error message
	errFormat := func(fpath string) string {
		return fmt.Sprintf("can't evaluate symlinks of %s", fpath)
	}

	// Failure test cases
	type FailTest struct {
		gopath, filename, wantErr string
	}

	failTests := []FailTest{
		{home + "/go", home + "/go/src/fake/test.go", errFormat(filepath.FromSlash(home + "/go/src/fake"))},
	}

	if runtime.GOOS != "windows" && runtime.GOOS != "plan9" {
		failTests = append(failTests, []FailTest{
			{home + "/go", home + "/src/fake/test.go", errFormat(filepath.FromSlash(home + "/src/fake"))},
			{home, home + "/src/fake/test.go", errFormat(filepath.FromSlash(home + "/src/fake"))},
			{home, home + "/go/src/fake/test.go", errFormat(filepath.FromSlash(home + "/go/src/fake"))},
		}...)
	}

	for _, test := range failTests {
		buildContext.GOPATH = test.gopath
		srcdir, importPath, err := guessImportPath(test.filename, &buildContext)
		if !strings.HasPrefix(fmt.Sprint(err), test.wantErr) {
			t.Errorf("guessImportPath(%q, %q) = %q, %q, %q; want %q, %q, %q",
				test.filename, test.gopath, srcdir, importPath, err, "", "", test.wantErr)
		}
	}
}

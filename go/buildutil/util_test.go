// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildutil_test

import (
	"go/build"
	"io/ioutil"
	"os"
	"runtime"
	"testing"

	"golang.org/x/tools/go/buildutil"
)

func TestContainingPackage(t *testing.T) {
	// unvirtualized:
	goroot := runtime.GOROOT()
	gopath := gopathContainingTools(t)

	type Test struct {
		gopath, filename, wantPkg string
	}

	tests := []Test{
		{gopath, goroot + "/src/fmt/print.go", "fmt"},
		{gopath, goroot + "/src/encoding/json/foo.go", "encoding/json"},
		{gopath, goroot + "/src/encoding/missing/foo.go", "(not found)"},
		{gopath, gopath + "/src/golang.org/x/tools/go/buildutil/util_test.go",
			"golang.org/x/tools/go/buildutil"},
	}

	if runtime.GOOS != "windows" && runtime.GOOS != "plan9" {
		// Make a symlink to gopath for test
		tmp, err := ioutil.TempDir(os.TempDir(), "go")
		if err != nil {
			t.Errorf("Unable to create a temporary directory in %s", os.TempDir())
		}

		defer os.RemoveAll(tmp)

		// symlink between $GOPATH/src and /tmp/go/src
		// in order to test all possible symlink cases
		if err := os.Symlink(gopath+"/src", tmp+"/src"); err != nil {
			t.Fatal(err)
		}
		tests = append(tests, []Test{
			{gopath, tmp + "/src/golang.org/x/tools/go/buildutil/util_test.go", "golang.org/x/tools/go/buildutil"},
			{tmp, gopath + "/src/golang.org/x/tools/go/buildutil/util_test.go", "golang.org/x/tools/go/buildutil"},
			{tmp, tmp + "/src/golang.org/x/tools/go/buildutil/util_test.go", "golang.org/x/tools/go/buildutil"},
		}...)
	}

	for _, test := range tests {
		var got string
		var buildContext = build.Default
		buildContext.GOPATH = test.gopath
		bp, err := buildutil.ContainingPackage(&buildContext, ".", test.filename)
		if err != nil {
			got = "(not found)"
		} else {
			got = bp.ImportPath
		}
		if got != test.wantPkg {
			t.Errorf("ContainingPackage(%q) = %s, want %s", test.filename, got, test.wantPkg)
		}
	}

	// TODO(adonovan): test on virtualized GOPATH too.
}

// gopathContainingTools returns the path of the GOPATH workspace
// with golang.org/x/tools, or fails the test if it can't locate it.
func gopathContainingTools(t *testing.T) string {
	p, err := build.Import("golang.org/x/tools", "", build.FindOnly)
	if err != nil {
		t.Fatal(err)
	}
	return p.Root
}

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Incomplete source tree on Android.

// +build !android

package buildutil_test

import (
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"golang.org/x/tools/go/buildutil"
)

func TestContainingPackage(t *testing.T) {
	// unvirtualized:
	goroot := runtime.GOROOT()
	gopath := filepath.SplitList(os.Getenv("GOPATH"))[0]

	// Make a symlink to gopath for test
	tmp, err := ioutil.TempDir(os.TempDir(), "go")
	if err != nil {
		t.Errorf("Unable to create a temporary directory in %s", os.TempDir())
	}

	// symlink between $GOPATH/src and /tmp/go/src
	// in order to test all possible symlink cases
	if err := os.Symlink(gopath+"/src", tmp+"/src"); err != nil {
		t.Fatal(err)
	}

	defer os.RemoveAll(tmp)

	for _, test := range []struct {
		gopath, filename, wantPkg string
	}{
		{gopath, goroot + "/src/fmt/print.go", "fmt"},
		{gopath, goroot + "/src/encoding/json/foo.go", "encoding/json"},
		{gopath, goroot + "/src/encoding/missing/foo.go", "(not found)"},
		{gopath, gopath + "/src/golang.org/x/tools/go/buildutil/util_test.go",
			"golang.org/x/tools/go/buildutil"},
		{gopath, tmp + "/src/golang.org/x/tools/go/buildutil/util_test.go",
			"golang.org/x/tools/go/buildutil"},
		{tmp, gopath + "/src/golang.org/x/tools/go/buildutil/util_test.go",
			"golang.org/x/tools/go/buildutil"},
		{tmp, tmp + "/src/golang.org/x/tools/go/buildutil/util_test.go",
			"golang.org/x/tools/go/buildutil"},
	} {
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

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Incomplete source tree on Android.

// +build !android

package buildutil_test

import (
	"go/build"
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

	tests := [][2]string{
		{goroot + "/src/fmt/print.go", "fmt"},
		{goroot + "/src/encoding/json/foo.go", "encoding/json"},
		{goroot + "/src/encoding/missing/foo.go", "(not found)"},
		{gopath + "/src/golang.org/x/tools/go/buildutil/util_test.go",
			"golang.org/x/tools/go/buildutil"},
	}
	for _, test := range tests {
		file, want := test[0], test[1]
		bp, err := buildutil.ContainingPackage(&build.Default, ".", file)
		got := bp.ImportPath
		if err != nil {
			got = "(not found)"
		}
		if got != want {
			t.Errorf("ContainingPackage(%q) = %s, want %s", file, got, want)
		}
	}

	// TODO(adonovan): test on virtualized GOPATH too.
}

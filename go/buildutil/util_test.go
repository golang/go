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
	// TODO(adonovan): simplify after Go 1.6.
	if buildutil.AllowVendor != 0 {
		tests = append(tests, [2]string{
			gopath + "/src/vendor/golang.org/x/net/http2/hpack/hpack.go",
			"vendor/golang.org/x/net/http2/hpack",
		})
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

func TestStripVendor(t *testing.T) {
	for _, test := range []struct {
		path, want string
	}{
		{"", ""},
		{"a", "a"},
		{"a/b", "a/b"},
		{"a/vendor/b", "b"},
		{"a/b/vendor/c/d", "c/d"},
		{"vendor/a/b", "a/b"},
		{"a/vendor", "a/vendor"},
	} {
		if got := buildutil.StripVendor(test.path); got != test.want {
			t.Errorf("StripVendor(%q) = %q, want %q",
				test.path, got, test.want)
		}
	}
}

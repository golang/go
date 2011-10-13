// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"path/filepath"
	"runtime"
	"testing"
)

func runTest(t *testing.T, path, pkg string) {
	exitCode = 0
	*pkgName = pkg
	*recursive = false

	if pkg == "" {
		processFiles([]string{path}, true)
	} else {
		processDirectory(path)
	}

	if exitCode != 0 {
		t.Errorf("processing %s failed: exitCode = %d", path, exitCode)
	}
}

var tests = []struct {
	path string
	pkg  string
}{
	// individual files
	{"testdata/test1.go", ""},

	// directories
	{filepath.Join(runtime.GOROOT(), "src/pkg/go/ast"), "ast"},
	{filepath.Join(runtime.GOROOT(), "src/pkg/go/doc"), "doc"},
	{filepath.Join(runtime.GOROOT(), "src/pkg/go/token"), "scanner"},
	{filepath.Join(runtime.GOROOT(), "src/pkg/go/scanner"), "scanner"},
	{filepath.Join(runtime.GOROOT(), "src/pkg/go/parser"), "parser"},
	{filepath.Join(runtime.GOROOT(), "src/pkg/exp/types"), "types"},
}

func Test(t *testing.T) {
	for _, test := range tests {
		runTest(t, test.path, test.pkg)
	}
}

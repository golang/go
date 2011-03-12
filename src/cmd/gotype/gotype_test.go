// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"path/filepath"
	"runtime"
	"testing"
)


func testDir(t *testing.T, dir, pkg string) {
	*pkgName = pkg
	*recursive = false
	processDirectory(dir)
	if exitCode != 0 {
		t.Errorf("processing %d failed: exitCode = %d", dir, exitCode)
	}
}


func Test(t *testing.T) {
	testDir(t, ".", "main")
	testDir(t, filepath.Join(runtime.GOROOT(), "src/pkg/go/ast"), "ast")
	testDir(t, filepath.Join(runtime.GOROOT(), "src/pkg/go/scanner"), "scanner")
	testDir(t, filepath.Join(runtime.GOROOT(), "src/pkg/go/parser"), "parser")
}

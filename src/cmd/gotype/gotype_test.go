// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"os"
	"path/filepath"
	"runtime"
	"path"
	"testing"
)


func testImporter(importPath string) (string, *ast.Scope, os.Error) {
	_, pkgName := path.Split(importPath) // filename is package name for std library
	return pkgName, ast.NewScope(nil), nil
}


func testDir(t *testing.T, dir, pkg string) {
	exitCode = 0
	*pkgName = pkg
	*recursive = false
	importer = testImporter
	processDirectory(dir)
	if exitCode != 0 {
		t.Errorf("processing %s failed: exitCode = %d", dir, exitCode)
	}
}


func Test(t *testing.T) {
	testDir(t, filepath.Join(runtime.GOROOT(), "src/pkg/go/ast"), "ast")
	testDir(t, filepath.Join(runtime.GOROOT(), "src/pkg/go/token"), "scanner")
	testDir(t, filepath.Join(runtime.GOROOT(), "src/pkg/go/scanner"), "scanner")
	testDir(t, filepath.Join(runtime.GOROOT(), "src/pkg/go/parser"), "parser")
}

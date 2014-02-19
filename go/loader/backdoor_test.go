// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import (
	"go/ast"
	"go/build"
	"go/token"
)

// PackageLocatorFunc exposes the address of parsePackageFiles to tests.
// This is a temporary hack until we expose a proper PackageLocator interface.
func PackageLocatorFunc() *func(ctxt *build.Context, fset *token.FileSet, path string, which string) ([]*ast.File, error) {
	return &parsePackageFiles
}

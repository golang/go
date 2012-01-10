// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package doc extracts source code documentation from a Go AST.
package doc

import "go/ast"

// PackageDoc is the documentation for an entire package.
type PackageDoc struct {
	Doc         string
	PackageName string
	ImportPath  string
	Filenames   []string
	Consts      []*ValueDoc
	Types       []*TypeDoc
	Vars        []*ValueDoc
	Funcs       []*FuncDoc
	Bugs        []string
}

// Value is the documentation for a (possibly grouped) var or const declaration.
type ValueDoc struct {
	Doc  string
	Decl *ast.GenDecl

	order int
}

// TypeDoc is the documentation for type declaration.
type TypeDoc struct {
	Doc       string
	Type      *ast.TypeSpec
	Decl      *ast.GenDecl
	Consts    []*ValueDoc // sorted list of constants of (mostly) this type
	Vars      []*ValueDoc // sorted list of variables of (mostly) this type
	Factories []*FuncDoc  // sorted list of functions returning this type
	Methods   []*FuncDoc  // sorted list of methods (including embedded ones) of this type

	methods  []*FuncDoc // top-level methods only
	embedded methodSet  // embedded methods only
	order    int
}

// Func is the documentation for a func declaration.
type FuncDoc struct {
	Doc  string
	Recv ast.Expr // TODO(rsc): Would like string here
	Name string
	Decl *ast.FuncDecl
}

// NewPackageDoc computes the package documentation for the given package
// and import path. If exportsOnly is set, only exported objects are
// included in the documentation.
func NewPackageDoc(pkg *ast.Package, importpath string, exportsOnly bool) *PackageDoc {
	var r docReader
	r.init(pkg.Name)
	filenames := make([]string, len(pkg.Files))
	i := 0
	for filename, f := range pkg.Files {
		if exportsOnly {
			r.fileExports(f)
		}
		r.addFile(f)
		filenames[i] = filename
		i++
	}
	return r.newDoc(importpath, filenames)
}

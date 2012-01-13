// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package doc extracts source code documentation from a Go AST.
package doc

import (
	"go/ast"
	"sort"
)

// Package is the documentation for an entire package.
type Package struct {
	Doc        string
	Name       string
	ImportPath string
	Imports    []string // TODO(gri) this field is not computed at the moment
	Filenames  []string
	Consts     []*Value
	Types      []*Type
	Vars       []*Value
	Funcs      []*Func
	Bugs       []string
}

// Value is the documentation for a (possibly grouped) var or const declaration.
type Value struct {
	Doc   string
	Names []string // var or const names in declaration order
	Decl  *ast.GenDecl

	order int
}

type Method struct {
	*Func
	// TODO(gri) The following fields are not set at the moment. 
	Recv  *Type // original receiver base type
	Level int   // embedding level; 0 means Func is not embedded
}

// Type is the documentation for type declaration.
type Type struct {
	Doc     string
	Name    string
	Type    *ast.TypeSpec
	Decl    *ast.GenDecl
	Consts  []*Value  // sorted list of constants of (mostly) this type
	Vars    []*Value  // sorted list of variables of (mostly) this type
	Funcs   []*Func   // sorted list of functions returning this type
	Methods []*Method // sorted list of methods (including embedded ones) of this type

	methods  []*Func   // top-level methods only
	embedded methodSet // embedded methods only
	order    int
}

// Func is the documentation for a func declaration.
type Func struct {
	Doc  string
	Name string
	// TODO(gri) remove Recv once we switch to new implementation
	Recv ast.Expr // TODO(rsc): Would like string here
	Decl *ast.FuncDecl
}

// Mode values control the operation of New.
type Mode int

const (
	// extract documentation for all package-level declarations,
	// not just exported ones
	AllDecls Mode = 1 << iota
)

// New computes the package documentation for the given package.
func New(pkg *ast.Package, importpath string, mode Mode) *Package {
	var r docReader
	r.init(pkg.Name, mode)
	filenames := make([]string, len(pkg.Files))
	// sort package files before reading them so that the
	// result is the same on different machines (32/64bit)
	i := 0
	for filename := range pkg.Files {
		filenames[i] = filename
		i++
	}
	sort.Strings(filenames)

	// process files in sorted order
	for _, filename := range filenames {
		f := pkg.Files[filename]
		if mode&AllDecls == 0 {
			r.fileExports(f)
		}
		r.addFile(f)
	}
	return r.newDoc(importpath, filenames)
}

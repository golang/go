// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package types declares the data structures for representing
// Go types and implements typechecking of package files.
//
// WARNING: THE TYPES API IS SUBJECT TO CHANGE.
//
package types

import (
	"go/ast"
	"go/token"
)

// A Context specifies the supporting context for type checking.
type Context struct {
	IntSize int64 // size in bytes of int and uint values
	PtrSize int64 // size in bytes of pointers

	// If Error is not nil, it is called with each error found
	// during type checking. Most error messages have accurate
	// position information; those error strings are formatted
	// filename:line:column: message.
	Error func(err error)

	// If Ident is not nil, it is called for each identifier id
	// denoting an Object in the files provided to Check, and
	// obj is the denoted object.
	// Ident is not called for fields and methods in struct or
	// interface types or composite literals, or for blank (_)
	// or dot (.) identifiers in dot-imports.
	// TODO(gri) Consider making Fields and Methods ordinary
	// Objects - than we could lift this restriction.
	Ident func(id *ast.Ident, obj Object)

	// If Expr is not nil, it is called for each expression x that is
	// type-checked: typ is the expression type, and val is the value
	// if x is constant, val is nil otherwise.
	//
	// Constants are represented as follows:
	//
	//	bool     ->  bool
	//	numeric  ->  int64, *big.Int, *big.Rat, Complex
	//	string   ->  string
	//	nil      ->  NilType
	//
	// Constant values are normalized, that is, they are represented
	// using the "smallest" possible type that can represent the value.
	// For instance, 1.0 is represented as an int64 because it can be
	// represented accurately as an int64.
	Expr func(x ast.Expr, typ Type, val interface{})

	// If Import is not nil, it is used instead of GcImport.
	Import Importer
}

// An Importer resolves import paths to Package objects.
// The imports map records the packages already imported,
// indexed by package id (canonical import path).
// An Importer must determine the canonical import path and
// check the map to see if it is already present in the imports map.
// If so, the Importer can return the map entry.  Otherwise, the
// Importer should load the package data for the given path into
// a new *Package, record pkg in the imports map, and then
// return pkg.
type Importer func(imports map[string]*Package, path string) (pkg *Package, err error)

// Default is the default context for type checking.
var Default = Context{
	// TODO(gri) Perhaps this should depend on GOARCH?
	IntSize: 8,
	PtrSize: 8,
}

// Check resolves and typechecks a set of package files within the given
// context. The package files' ASTs are augmented by assigning types to
// ast.Objects. If there are no errors, Check returns the package, otherwise
// it returns the first error. If the context's Error handler is nil,
// Check terminates as soon as the first error is encountered.
//
// CAUTION: At the moment, the returned *ast.Package only contains the package
//          name and scope - the other fields are not set up. The returned
//          *Package contains the name and imports (but no scope yet). Once
//          we have the scope moved from *ast.Scope to *Scope, only *Package
//          will be returned.
//
func (ctxt *Context) Check(fset *token.FileSet, files []*ast.File) (*Package, error) {
	return check(ctxt, fset, files)
}

// Check is shorthand for Default.Check.
func Check(fset *token.FileSet, files []*ast.File) (*Package, error) {
	return Default.Check(fset, files)
}

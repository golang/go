// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package types declares the data structures for representing
// Go types and implements typechecking of package files.
//
// WARNING: THE TYPES API IS SUBJECT TO SIGNIFICANT CHANGE.
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
	// during type checking.
	Error func(err error)

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
	Import ast.Importer
}

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
func (ctxt *Context) Check(fset *token.FileSet, files map[string]*ast.File) (*ast.Package, error) {
	return check(ctxt, fset, files)
}

// Check is shorthand for Default.Check.
func Check(fset *token.FileSet, files map[string]*ast.File) (*ast.Package, error) {
	return Default.Check(fset, files)
}

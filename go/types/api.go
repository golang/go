// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package types declares the data structures for representing
// Go types and implements typechecking of package files.
//
// WARNING: THE TYPES API IS SUBJECT TO CHANGE.
//
package types

// Most correct programs should now be accepted by the types package,
// but there are several known bugs which permit incorrect programs to
// pass without errors. Please do not file issues against these for now
// since they are known already:
//
// BUG(gri): Method sets are computed loosely and don't distinguish between ptr vs non-pointer receivers.
// BUG(gri): Method expressions and method values work accidentally and may not be fully checked.
// BUG(gri): Conversions of constants only change the type, not the value (e.g., int(1.1) is wrong).
// BUG(gri): Some built-ins don't check parameters fully, yet (e.g. append).
// BUG(gri): Use of labels is not checked.
// BUG(gri): Unused variables and imports are not reported.
// BUG(gri): Interface vs non-interface comparisons are not correctly implemented.
// BUG(gri): Switch statements don't check correct use of 'fallthrough'.
// BUG(gri): Switch statements don't check duplicate cases for all types for which it is required.
// BUG(gri): Some built-ins may not be callable if in statement-context.

// The API is still slightly in flux and the following changes are considered:
//
// API(gri): The GcImporter should probably be in its own package - it is only one of possible importers.

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
)

// A Context specifies the supporting context for type checking.
// An empty Context is a ready-to-use default context.
type Context struct {
	// If Error != nil, it is called with each error found
	// during type checking. The error strings of errors with
	// detailed position information are formatted as follows:
	// filename:line:column: message
	Error func(err error)

	// If Ident != nil, it is called for each identifier id
	// denoting an Object in the files provided to Check, and
	// obj is the denoted object.
	// Ident is not called for fields and methods in struct or
	// interface types or composite literals, or for blank (_)
	// or dot (.) identifiers in dot-imports.
	// TODO(gri) Consider making Fields and Methods ordinary
	// Objects - than we could lift this restriction.
	Ident func(id *ast.Ident, obj Object)

	// If Expr != nil, it is called exactly once for each expression x
	// that is type-checked: typ is the expression type, and val is the
	// value if x is constant, val is nil otherwise.
	//
	// If x is a literal value (constant, composite literal), typ is always
	// the dynamic type of x (never an interface type). Otherwise, typ is x's
	// static type (possibly an interface type).
	// TODO(gri): Should this hold for all constant expressions?
	Expr func(x ast.Expr, typ Type, val exact.Value)

	// If Import != nil, it is called for each imported package.
	// Otherwise, GcImporter is called.
	Import Importer

	// If Alignof != nil, it is called to determine the alignment
	// of the given type. Otherwise DefaultAlignmentof is called.
	// Alignof must implement the alignment guarantees required by
	// the spec.
	Alignof func(Type) int64

	// If Offsetsof != nil, it is called to determine the offsets
	// of the given struct fields, in bytes. Otherwise DefaultOffsetsof
	// is called. Offsetsof must implement the offset guarantees
	// required by the spec.
	Offsetsof func(fields []*Field) []int64

	// If Sizeof != nil, it is called to determine the size of the
	// given type. Otherwise, DefaultSizeof is called. Sizeof must
	// implement the size guarantees required by the spec.
	Sizeof func(Type) int64
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

// Check resolves and typechecks a set of package files within the given
// context. It returns the package and the first error encountered, if
// any. If the context's Error handler is nil, Check terminates as soon
// as the first error is encountered; otherwise it continues until the
// entire package is checked. If there are errors, the package may be
// only partially type-checked, and the resulting package may be incomplete
// (missing objects, imports, etc.).
func (ctxt *Context) Check(path string, fset *token.FileSet, files ...*ast.File) (*Package, error) {
	return check(ctxt, path, fset, files...)
}

// Check is shorthand for ctxt.Check where ctxt is a default (empty) context.
func Check(path string, fset *token.FileSet, files ...*ast.File) (*Package, error) {
	var ctxt Context
	return ctxt.Check(path, fset, files...)
}

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package types declares the data types and implements
// the algorithms for type-checking of Go packages.
// Use Check and Config.Check to invoke the type-checker.
//
// Type-checking consists of several interdependent phases:
//
// Name resolution maps each identifier (ast.Ident) in the program to the
// language object (Object) it denotes.
// Use Info.Objects, Info.Implicits for the results of name resolution.
//
// Constant folding computes the exact constant value (exact.Value) for
// every expression (ast.Expr) that is a compile-time constant.
// Use Info.Values for the results of constant folding.
//
// Type inference computes the type (Type) of every expression (ast.Expr)
// and checks for compliance with the language specification.
// Use Info.Types for the results of type evaluation.
//
package types

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
)

// Check type-checks a package and returns the resulting complete package
// object, or a nil package and the first error. The package is specified
// by a list of *ast.Files and corresponding file set, and the import path
// the package is identified with. The clean path must not be empty or dot (".").
//
// For more control over type-checking and results, use Config.Check.
func Check(path string, fset *token.FileSet, files []*ast.File) (*Package, error) {
	var conf Config
	pkg, err := conf.Check(path, fset, files, nil)
	if err != nil {
		return nil, err
	}
	return pkg, nil
}

// A Config specifies the configuration for type checking.
// The zero value for Config is a ready-to-use default configuration.
type Config struct {
	// If IgnoreFuncBodies is set, function bodies are not
	// type-checked.
	IgnoreFuncBodies bool

	// If FakeImportC is set, `import "C"` (for packages requiring Cgo)
	// declares an empty "C" package and errors are omitted for qualified
	// identifiers referring to package C (which won't find an object).
	// This feature is intended for the standard library cmd/api tool.
	//
	// Caution: Effects may be unpredictable due to follow-up errors.
	//          Do not use casually!
	FakeImportC bool

	// Packages is used to look up (and thus canonicalize) packages by
	// package path. If Packages is nil, it is set to a new empty map.
	// During type-checking, imported packages are added to the map.
	Packages map[string]*Package

	// If Error != nil, it is called with each error found
	// during type checking. The error strings of errors with
	// detailed position information are formatted as follows:
	// filename:line:column: message
	Error func(err error)

	// If Import != nil, it is called for each imported package.
	// Otherwise, GcImporter is called.
	// An importer resolves import paths to Packages.
	// The imports map records packages already known,
	// indexed by canonical package path. The type-checker will
	// invoke Import with Config.Packages.
	// An importer must determine the canonical package path and
	// check imports to see if it is already present in the map.
	// If so, the Importer can return the map entry.  Otherwise,
	// the importer must load the package data for the given path
	// into a new *Package, record it in imports map, and return
	// the package.
	Import func(imports map[string]*Package, path string) (pkg *Package, err error)

	// If Sizes != nil, it provides the sizing functions for package unsafe.
	// Otherwise &StdSize{WordSize: 8, MaxAlign: 8} is used instead.
	Sizes Sizes
}

// Info holds result type information for a type-checked package.
// Only the information for which a map is provided is collected.
// If the package has type errors, the collected information may
// be incomplete.
type Info struct {
	// Types maps expressions to their types. Identifiers on the
	// lhs of declarations are collected in Objects, not Types.
	//
	// For an expression denoting a predeclared built-in function
	// the recorded signature is call-site specific. If the call
	// result is not a constant, the recorded type is an argument-
	// specific signature. Otherwise, the recorded type is invalid.
	Types map[ast.Expr]Type

	// Values maps constant expressions to their values.
	Values map[ast.Expr]exact.Value

	// Objects maps identifiers to their corresponding objects (including
	// package names, dots "." of dot-imports, and blank "_" identifiers).
	// For identifiers that do not denote objects (e.g., the package name
	// in package clauses, blank identifiers on the lhs of assignments, or
	// symbolic variables t in t := x.(type) of type switch headers), the
	// corresponding objects are nil.
	Objects map[*ast.Ident]Object

	// Implicits maps nodes to their implicitly declared objects, if any.
	// The following node and object types may appear:
	//
	//	node               declared object
	//
	//	*ast.ImportSpec    *PkgName (imports w/o renames), or imported objects (dot-imports)
	//	*ast.CaseClause    type-specific *Var for each type switch case clause (incl. default)
	//      *ast.Field         anonymous struct field or parameter *Var
	//
	Implicits map[ast.Node]Object

	// Selections maps selector expressions to their corresponding selections.
	Selections map[*ast.SelectorExpr]*Selection

	// Scopes maps ast.Nodes to the scopes they define. Note that package scopes
	// are not associated with a specific node but with all files belonging to a
	// package. Thus, the package scope can be found in the type-checked package
	// object.
	//
	// The following node types may appear in Scopes:
	//
	//	*ast.File
	//	*ast.FuncType
	//	*ast.BlockStmt
	//	*ast.IfStmt
	//	*ast.SwitchStmt
	//	*ast.TypeSwitchStmt
	//	*ast.CaseClause
	//	*ast.CommClause
	//	*ast.ForStmt
	//	*ast.RangeStmt
	//
	Scopes map[ast.Node]*Scope
}

// Check type-checks a package and returns the resulting package object,
// the first error if any, and if info != nil, additional type information.
// The package is marked as complete if no errors occurred, otherwise it is
// incomplete.
//
// The package is specified by a list of *ast.Files and corresponding
// file set, and the package path the package is identified with.
// The clean path must not be empty or dot (".").
func (conf *Config) Check(path string, fset *token.FileSet, files []*ast.File, info *Info) (*Package, error) {
	pkg, err := conf.check(path, fset, files, info)
	if err == nil {
		pkg.complete = true
	}
	return pkg, err
}

// IsAssignableTo reports whether a value of type V is assignable to a variable of type T.
func IsAssignableTo(V, T Type) bool {
	x := operand{mode: value, typ: V}
	return x.isAssignableTo(nil, T) // config not needed for non-constant x
}

// Implements reports whether a value of type V implements T, as follows:
//
// 1) For non-interface types V, or if static is set, V implements T if all
// methods of T are present in V. Informally, this reports whether V is a
// subtype of T.
//
// 2) For interface types V, and if static is not set, V implements T if all
// methods of T which are also present in V have matching types. Informally,
// this indicates whether a type assertion x.(T) where x is of type V would
// be legal (the concrete dynamic type of x may implement T even if V does
// not statically implement it).
//
func Implements(V Type, T *Interface, static bool) bool {
	f, _ := MissingMethod(V, T, static)
	return f == nil
}

// BUG(gri): Interface vs non-interface comparisons are not correctly implemented.

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !gotypes

// This file contains stubs for the pieces of the tool that require the go/types package,
// to be used if go/types is not available.

package main

import (
	"go/ast"
	"go/token"
)

// Type is equivalent to go/types.Type. Repeating it here allows us to avoid
// depending on the go/types package.
type Type interface {
	String() string
}

func (pkg *Package) check(fs *token.FileSet, astFiles []*ast.File) error {
	return nil
}

func (pkg *Package) isStruct(c *ast.CompositeLit) (bool, string) {
	return true, "" // Assume true, so we do the check.
}

func (f *File) matchArgType(t printfArgType, arg ast.Expr) bool {
	return true // We can't tell without types.
}

func (f *File) numArgsInSignature(call *ast.CallExpr) int {
	return 0 // We don't know.
}

func (f *File) isErrorMethodCall(call *ast.CallExpr) bool {
	// Is it a selector expression? Otherwise it's a function call, not a method call.
	if _, ok := call.Fun.(*ast.SelectorExpr); !ok {
		return false
	}
	return true // Best guess we can make without types.
}

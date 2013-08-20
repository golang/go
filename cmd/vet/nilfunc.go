// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
This file contains the code to check for useless function comparisons.
A useless comparison is one like f == nil as opposed to f() == nil.
*/

package main

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/types"
)

func (f *File) checkNilFuncComparison(e *ast.BinaryExpr) {
	if !vet("nilfunc") {
		return
	}

	// Only want == or != comparisons.
	if e.Op != token.EQL && e.Op != token.NEQ {
		return
	}

	// Only want comparisons with a nil identifier on one side.
	var e2 ast.Expr
	switch {
	case f.isNil(e.X):
		e2 = e.Y
	case f.isNil(e.Y):
		e2 = e.X
	default:
		return
	}

	// Only want identifiers or selector expressions.
	var obj types.Object
	switch v := e2.(type) {
	case *ast.Ident:
		obj = f.pkg.idents[v]
	case *ast.SelectorExpr:
		obj = f.pkg.idents[v.Sel]
	default:
		return
	}

	// Only want functions.
	if _, ok := obj.(*types.Func); !ok {
		return
	}

	f.Badf(e.Pos(), "comparison of function %v %v nil is always %v", obj.Name(), e.Op, e.Op == token.NEQ)
}

// isNil reports whether the provided expression is the built-in nil
// identifier.
func (f *File) isNil(e ast.Expr) bool {
	return f.pkg.types[e] == types.Typ[types.UntypedNil]
}

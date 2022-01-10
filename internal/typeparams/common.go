// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typeparams provides functions to work indirectly with type parameter
// data stored in go/ast and go/types objects, while these API are guarded by a
// build constraint.
//
// This package exists to make it easier for tools to work with generic code,
// while also compiling against older Go versions.
package typeparams

import (
	"go/ast"
	"go/token"
	"go/types"
)

// UnpackIndexExpr extracts data from AST nodes that represent index
// expressions.
//
// For an ast.IndexExpr, the resulting indices slice will contain exactly one
// index expression. For an ast.IndexListExpr (go1.18+), it may have a variable
// number of index expressions.
//
// For nodes that don't represent index expressions, the first return value of
// UnpackIndexExpr will be nil.
func UnpackIndexExpr(n ast.Node) (x ast.Expr, lbrack token.Pos, indices []ast.Expr, rbrack token.Pos) {
	switch e := n.(type) {
	case *ast.IndexExpr:
		return e.X, e.Lbrack, []ast.Expr{e.Index}, e.Rbrack
	case *IndexListExpr:
		return e.X, e.Lbrack, e.Indices, e.Rbrack
	}
	return nil, token.NoPos, nil, token.NoPos
}

// PackIndexExpr returns an *ast.IndexExpr or *ast.IndexListExpr, depending on
// the cardinality of indices. Calling PackIndexExpr with len(indices) == 0
// will panic.
func PackIndexExpr(x ast.Expr, lbrack token.Pos, indices []ast.Expr, rbrack token.Pos) ast.Expr {
	switch len(indices) {
	case 0:
		panic("empty indices")
	case 1:
		return &ast.IndexExpr{
			X:      x,
			Lbrack: lbrack,
			Index:  indices[0],
			Rbrack: rbrack,
		}
	default:
		return &IndexListExpr{
			X:       x,
			Lbrack:  lbrack,
			Indices: indices,
			Rbrack:  rbrack,
		}
	}
}

// IsTypeParam reports whether t is a type parameter.
func IsTypeParam(t types.Type) bool {
	_, ok := t.(*TypeParam)
	return ok
}

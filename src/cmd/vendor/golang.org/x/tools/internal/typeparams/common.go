// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typeparams contains common utilities for writing tools that interact
// with generic Go code, as introduced with Go 1.18.
//
// Many of the types and functions in this package are proxies for the new APIs
// introduced in the standard library with Go 1.18. For example, the
// typeparams.Union type is an alias for go/types.Union, and the ForTypeSpec
// function returns the value of the go/ast.TypeSpec.TypeParams field. At Go
// versions older than 1.18 these helpers are implemented as stubs, allowing
// users of this package to write code that handles generic constructs inline,
// even if the Go version being used to compile does not support generics.
//
// Additionally, this package contains common utilities for working with the
// new generic constructs, to supplement the standard library APIs. Notably,
// the StructuralTerms API computes a minimal representation of the structural
// restrictions on a type parameter. In the future, this API may be available
// from go/types.
//
// See the example/README.md for a more detailed guide on how to update tools
// to support generics.
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

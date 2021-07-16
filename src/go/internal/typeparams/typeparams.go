// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import (
	"fmt"
	"go/ast"
	"go/token"
)

func PackIndexExpr(x ast.Expr, lbrack token.Pos, exprs []ast.Expr, rbrack token.Pos) ast.Expr {
	switch len(exprs) {
	case 0:
		panic("internal error: PackIndexExpr with empty expr slice")
	case 1:
		return &ast.IndexExpr{
			X:      x,
			Lbrack: lbrack,
			Index:  exprs[0],
			Rbrack: rbrack,
		}
	default:
		return &ast.MultiIndexExpr{
			X:       x,
			Lbrack:  lbrack,
			Indices: exprs,
			Rbrack:  rbrack,
		}
	}
}

// IndexExpr wraps an ast.IndexExpr or ast.MultiIndexExpr into the
// MultiIndexExpr interface.
//
// Orig holds the original ast.Expr from which this IndexExpr was derived.
type IndexExpr struct {
	Orig ast.Expr // the wrapped expr, which may be distinct from MultiIndexExpr below.
	*ast.MultiIndexExpr
}

func UnpackIndexExpr(n ast.Node) *IndexExpr {
	switch e := n.(type) {
	case *ast.IndexExpr:
		return &IndexExpr{e, &ast.MultiIndexExpr{
			X:       e.X,
			Lbrack:  e.Lbrack,
			Indices: []ast.Expr{e.Index},
			Rbrack:  e.Rbrack,
		}}
	case *ast.MultiIndexExpr:
		return &IndexExpr{e, e}
	}
	return nil
}

func Get(n ast.Node) *ast.FieldList {
	switch n := n.(type) {
	case *ast.TypeSpec:
		return n.TParams
	case *ast.FuncType:
		return n.TParams
	default:
		panic(fmt.Sprintf("node type %T has no type parameters", n))
	}
}

func Set(n ast.Node, params *ast.FieldList) {
	switch n := n.(type) {
	case *ast.TypeSpec:
		n.TParams = params
	case *ast.FuncType:
		n.TParams = params
	default:
		panic(fmt.Sprintf("node type %T has no type parameters", n))
	}
}

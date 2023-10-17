// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import (
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
		return &ast.IndexListExpr{
			X:       x,
			Lbrack:  lbrack,
			Indices: exprs,
			Rbrack:  rbrack,
		}
	}
}

// IndexExpr wraps an ast.IndexExpr or ast.IndexListExpr.
//
// Orig holds the original ast.Expr from which this IndexExpr was derived.
type IndexExpr struct {
	Orig ast.Expr // the wrapped expr, which may be distinct from the IndexListExpr below.
	*ast.IndexListExpr
}

func UnpackIndexExpr(n ast.Node) *IndexExpr {
	switch e := n.(type) {
	case *ast.IndexExpr:
		return &IndexExpr{e, &ast.IndexListExpr{
			X:       e.X,
			Lbrack:  e.Lbrack,
			Indices: []ast.Expr{e.Index},
			Rbrack:  e.Rbrack,
		}}
	case *ast.IndexListExpr:
		return &IndexExpr{e, e}
	}
	return nil
}

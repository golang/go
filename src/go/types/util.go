// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains various functionality that is
// different between go/types and types2. Factoring
// out this code allows more of the rest of the code
// to be shared.

package types

import (
	"go/ast"
	"go/constant"
	"go/token"
)

const isTypes2 = false

// cmpPos compares the positions p and q and returns a result r as follows:
//
// r <  0: p is before q
// r == 0: p and q are the same position (but may not be identical)
// r >  0: p is after q
//
// If p and q are in different files, p is before q if the filename
// of p sorts lexicographically before the filename of q.
func cmpPos(p, q token.Pos) int { return int(p - q) }

// hasDots reports whether the last argument in the call is followed by ...
func hasDots(call *ast.CallExpr) bool { return call.Ellipsis.IsValid() }

// dddErrPos returns the positioner for reporting an invalid ... use in a call.
func dddErrPos(call *ast.CallExpr) positioner { return atPos(call.Ellipsis) }

// isdddArray reports whether atyp is of the form [...]E.
func isdddArray(atyp *ast.ArrayType) bool {
	if atyp.Len != nil {
		if ddd, _ := atyp.Len.(*ast.Ellipsis); ddd != nil && ddd.Elt == nil {
			return true
		}
	}
	return false
}

// argErrPos returns positioner for reporting an invalid argument count.
func argErrPos(call *ast.CallExpr) positioner { return inNode(call, call.Rparen) }

// startPos returns the start position of node n.
func startPos(n ast.Node) token.Pos { return n.Pos() }

// endPos returns the position of the first character immediately after node n.
func endPos(n ast.Node) token.Pos { return n.End() }

// makeFromLiteral returns the constant value for the given literal string and kind.
func makeFromLiteral(lit string, kind token.Token) constant.Value {
	return constant.MakeFromLiteral(lit, kind, 0)
}

// unTilde returns the operand of ~x and true, or x and false.
func unTilde(x ast.Expr) (ast.Expr, bool) {
	if op, _ := x.(*ast.UnaryExpr); op != nil && op.Op == token.TILDE {
		return op.X, true
	}
	return x, false
}

// splitOr returns the operands of x | y, or nil, nil.
func splitOr(e ast.Expr) (x, y ast.Expr) {
	if op, _ := e.(*ast.BinaryExpr); op != nil && op.Op == token.OR {
		return op.X, op.Y
	}
	return nil, nil
}

// newIdent returns a new identifier with the given position and name.
func newIdent(pos token.Pos, name string) *ast.Ident {
	return &ast.Ident{NamePos: pos, Name: name}
}

// If x represents an index expression, unpackIndexExpr returns the indexed expression
// and, if unpackParams is set, its indices. If unpackParams is not set, indices is nil.
// If x is not an index expression, unpackIndexExpr returns (nil, nil).
func unpackIndexExpr(x ast.Expr, unpackParams bool) (indexed ast.Expr, indices []ast.Expr) {
	switch ix := x.(type) {
	case *ast.IndexExpr:
		indexed = ix.X
		if unpackParams {
			indices = []ast.Expr{ix.Index}
		}
	case *ast.IndexListExpr:
		indexed = ix.X
		if unpackParams {
			indices = ix.Indices
		}
	}
	return
}

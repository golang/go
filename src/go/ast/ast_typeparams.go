// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build typeparams
// +build typeparams

package ast

import "go/token"

type (
	// A FuncType node represents a function type.
	FuncType struct {
		Func    token.Pos  // position of "func" keyword (token.NoPos if there is no "func")
		TParams *FieldList // type parameters; or nil
		Params  *FieldList // (incoming) parameters; non-nil
		Results *FieldList // (outgoing) results; or nil
	}

	// A TypeSpec node represents a type declaration (TypeSpec production).
	TypeSpec struct {
		Doc     *CommentGroup // associated documentation; or nil
		Name    *Ident        // type name
		TParams *FieldList    // type parameters; or nil
		Assign  token.Pos     // position of '=', if any
		Type    Expr          // *Ident, *ParenExpr, *SelectorExpr, *StarExpr, or any of the *XxxTypes
		Comment *CommentGroup // line comments; or nil
	}

	// A ListExpr node represents a list of expressions separated by commas.
	// ListExpr nodes are used as index in IndexExpr nodes representing type
	// or function instantiations with more than one type argument.
	ListExpr struct {
		ElemList []Expr
	}
)

func (*ListExpr) exprNode() {}
func (x *ListExpr) Pos() token.Pos {
	if len(x.ElemList) > 0 {
		return x.ElemList[0].Pos()
	}
	return token.NoPos
}
func (x *ListExpr) End() token.Pos {
	if len(x.ElemList) > 0 {
		return x.ElemList[len(x.ElemList)-1].End()
	}
	return token.NoPos
}

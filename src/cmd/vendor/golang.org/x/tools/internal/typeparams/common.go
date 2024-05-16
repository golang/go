// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typeparams contains common utilities for writing tools that
// interact with generic Go code, as introduced with Go 1.18. It
// supplements the standard library APIs. Notably, the StructuralTerms
// API computes a minimal representation of the structural
// restrictions on a type parameter.
//
// An external version of these APIs is available in the
// golang.org/x/exp/typeparams module.
package typeparams

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/internal/aliases"
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
	case *ast.IndexListExpr:
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
		return &ast.IndexListExpr{
			X:       x,
			Lbrack:  lbrack,
			Indices: indices,
			Rbrack:  rbrack,
		}
	}
}

// IsTypeParam reports whether t is a type parameter (or an alias of one).
func IsTypeParam(t types.Type) bool {
	_, ok := aliases.Unalias(t).(*types.TypeParam)
	return ok
}

// GenericAssignableTo is a generalization of types.AssignableTo that
// implements the following rule for uninstantiated generic types:
//
// If V and T are generic named types, then V is considered assignable to T if,
// for every possible instantiation of V[A_1, ..., A_N], the instantiation
// T[A_1, ..., A_N] is valid and V[A_1, ..., A_N] implements T[A_1, ..., A_N].
//
// If T has structural constraints, they must be satisfied by V.
//
// For example, consider the following type declarations:
//
//	type Interface[T any] interface {
//		Accept(T)
//	}
//
//	type Container[T any] struct {
//		Element T
//	}
//
//	func (c Container[T]) Accept(t T) { c.Element = t }
//
// In this case, GenericAssignableTo reports that instantiations of Container
// are assignable to the corresponding instantiation of Interface.
func GenericAssignableTo(ctxt *types.Context, V, T types.Type) bool {
	V = aliases.Unalias(V)
	T = aliases.Unalias(T)

	// If V and T are not both named, or do not have matching non-empty type
	// parameter lists, fall back on types.AssignableTo.

	VN, Vnamed := V.(*types.Named)
	TN, Tnamed := T.(*types.Named)
	if !Vnamed || !Tnamed {
		return types.AssignableTo(V, T)
	}

	vtparams := VN.TypeParams()
	ttparams := TN.TypeParams()
	if vtparams.Len() == 0 || vtparams.Len() != ttparams.Len() || VN.TypeArgs().Len() != 0 || TN.TypeArgs().Len() != 0 {
		return types.AssignableTo(V, T)
	}

	// V and T have the same (non-zero) number of type params. Instantiate both
	// with the type parameters of V. This must always succeed for V, and will
	// succeed for T if and only if the type set of each type parameter of V is a
	// subset of the type set of the corresponding type parameter of T, meaning
	// that every instantiation of V corresponds to a valid instantiation of T.

	// Minor optimization: ensure we share a context across the two
	// instantiations below.
	if ctxt == nil {
		ctxt = types.NewContext()
	}

	var targs []types.Type
	for i := 0; i < vtparams.Len(); i++ {
		targs = append(targs, vtparams.At(i))
	}

	vinst, err := types.Instantiate(ctxt, V, targs, true)
	if err != nil {
		panic("type parameters should satisfy their own constraints")
	}

	tinst, err := types.Instantiate(ctxt, T, targs, true)
	if err != nil {
		return false
	}

	return types.AssignableTo(vinst, tinst)
}

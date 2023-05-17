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
// restrictions on a type parameter.
//
// An external version of these APIs is available in the
// golang.org/x/exp/typeparams module.
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

// OriginMethod returns the origin method associated with the method fn.
// For methods on a non-generic receiver base type, this is just
// fn. However, for methods with a generic receiver, OriginMethod returns the
// corresponding method in the method set of the origin type.
//
// As a special case, if fn is not a method (has no receiver), OriginMethod
// returns fn.
func OriginMethod(fn *types.Func) *types.Func {
	recv := fn.Type().(*types.Signature).Recv()
	if recv == nil {
		return fn
	}
	base := recv.Type()
	p, isPtr := base.(*types.Pointer)
	if isPtr {
		base = p.Elem()
	}
	named, isNamed := base.(*types.Named)
	if !isNamed {
		// Receiver is a *types.Interface.
		return fn
	}
	if ForNamed(named).Len() == 0 {
		// Receiver base has no type parameters, so we can avoid the lookup below.
		return fn
	}
	orig := NamedTypeOrigin(named)
	gfn, _, _ := types.LookupFieldOrMethod(orig, true, fn.Pkg(), fn.Name())
	return gfn.(*types.Func)
}

// GenericAssignableTo is a generalization of types.AssignableTo that
// implements the following rule for uninstantiated generic types:
//
// If V and T are generic named types, then V is considered assignable to T if,
// for every possible instantation of V[A_1, ..., A_N], the instantiation
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
func GenericAssignableTo(ctxt *Context, V, T types.Type) bool {
	// If V and T are not both named, or do not have matching non-empty type
	// parameter lists, fall back on types.AssignableTo.

	VN, Vnamed := V.(*types.Named)
	TN, Tnamed := T.(*types.Named)
	if !Vnamed || !Tnamed {
		return types.AssignableTo(V, T)
	}

	vtparams := ForNamed(VN)
	ttparams := ForNamed(TN)
	if vtparams.Len() == 0 || vtparams.Len() != ttparams.Len() || NamedTypeArgs(VN).Len() != 0 || NamedTypeArgs(TN).Len() != 0 {
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
		ctxt = NewContext()
	}

	var targs []types.Type
	for i := 0; i < vtparams.Len(); i++ {
		targs = append(targs, vtparams.At(i))
	}

	vinst, err := Instantiate(ctxt, V, targs, true)
	if err != nil {
		panic("type parameters should satisfy their own constraints")
	}

	tinst, err := Instantiate(ctxt, T, targs, true)
	if err != nil {
		return false
	}

	return types.AssignableTo(vinst, tinst)
}

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build typeparams && go1.18
// +build typeparams,go1.18

package typeparams

import (
	"go/ast"
	"go/types"
)

// NOTE: doc comments must be kept in sync with notypeparams.go.

// Enabled reports whether type parameters are enabled in the current build
// environment.
const Enabled = true

// GetIndexExprData extracts data from AST nodes that represent index
// expressions.
//
// For an ast.IndexExpr, the resulting IndexExprData will have exactly one
// index expression. For an ast.MultiIndexExpr (go1.18+), it may have a
// variable number of index expressions.
//
// For nodes that don't represent index expressions, GetIndexExprData returns
// nil.
func GetIndexExprData(n ast.Node) *IndexExprData {
	switch e := n.(type) {
	case *ast.IndexExpr:
		return &IndexExprData{
			X:       e.X,
			Lbrack:  e.Lbrack,
			Indices: []ast.Expr{e.Index},
			Rbrack:  e.Rbrack,
		}
	case *ast.MultiIndexExpr:
		return (*IndexExprData)(e)
	}
	return nil
}

// ForTypeDecl extracts the (possibly nil) type parameter node list from n.
func ForTypeDecl(n *ast.TypeSpec) *ast.FieldList {
	return n.TParams
}

// ForFuncDecl extracts the (possibly nil) type parameter node list from n.
func ForFuncDecl(n *ast.FuncDecl) *ast.FieldList {
	if n.Type != nil {
		return n.Type.TParams
	}
	return nil
}

// ForSignature extracts the (possibly empty) type parameter object list from
// sig.
func ForSignature(sig *types.Signature) []*types.TypeName {
	return tparamsSlice(sig.TParams())
}

// IsComparable reports if iface is the comparable interface.
func IsComparable(iface *types.Interface) bool {
	return iface.IsComparable()
}

// IsConstraint reports whether iface may only be used as a type parameter
// constraint (i.e. has a type set or is the comparable interface).
func IsConstraint(iface *types.Interface) bool {
	return iface.IsConstraint()
}

// ForNamed extracts the (possibly empty) type parameter object list from
// named.
func ForNamed(named *types.Named) []*types.TypeName {
	return tparamsSlice(named.TParams())
}

func tparamsSlice(tparams *types.TParamList) []*types.TypeName {
	length := tparams.Len()
	if length == 0 {
		return nil
	}

	result := make([]*types.TypeName, length)
	for i := 0; i < length; i++ {
		result[i] = tparams.At(i).Obj()
	}

	return result
}

// NamedTArgs extracts the (possibly empty) type argument list from named.
func NamedTArgs(named *types.Named) []types.Type {
	targs := named.TArgs()
	numArgs := targs.Len()

	typs := make([]types.Type, numArgs)
	for i := 0; i < numArgs; i++ {
		typs[i] = targs.At(i)
	}

	return typs
}

// InitInferred initializes info to record inferred type information.
func InitInferred(info *types.Info) {
	info.Inferred = make(map[ast.Expr]types.Inferred)
}

// GetInferred extracts inferred type information from info for e.
//
// The expression e may have an inferred type if it is an *ast.IndexExpr
// representing partial instantiation of a generic function type for which type
// arguments have been inferred using constraint type inference, or if it is an
// *ast.CallExpr for which type type arguments have be inferred using both
// constraint type inference and function argument inference.
func GetInferred(info *types.Info, e ast.Expr) ([]types.Type, *types.Signature) {
	if info.Inferred == nil {
		return nil, nil
	}
	inf := info.Inferred[e]

	length := inf.TArgs.Len()

	typs := make([]types.Type, length)
	for i := 0; i < length; i++ {
		typs[i] = inf.TArgs.At(i)
	}

	return typs, inf.Sig
}

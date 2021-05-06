// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build typeparams && go1.17
// +build typeparams,go1.17

package typeparams

import (
	"go/ast"
	"go/types"
)

// NOTE: doc comments must be kept in sync with notypeparams.go.

// Enabled reports whether type parameters are enabled in the current build
// environment.
const Enabled = true

// UnpackIndex extracts all index expressions from e. For non-generic code this
// is always one expression: e.Index, but may be more than one expression for
// generic type instantiation.
func UnpackIndex(e *ast.IndexExpr) []ast.Expr {
	if x, _ := e.Index.(*ast.ListExpr); x != nil {
		return x.ElemList
	}
	if e.Index != nil {
		return []ast.Expr{e.Index}
	}
	return nil
}

// IsListExpr reports whether n is an *ast.ListExpr, which is a new node type
// introduced to hold type arguments for generic type instantiation.
func IsListExpr(n ast.Node) bool {
	_, ok := n.(*ast.ListExpr)
	return ok
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
	return sig.TParams()
}

// HasTypeSet reports if iface has a type set.
func HasTypeSet(iface *types.Interface) bool {
	return iface.HasTypeList()
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
	return named.TParams()
}

// NamedTArgs extracts the (possibly empty) type argument list from named.
func NamedTArgs(named *types.Named) []types.Type {
	return named.TArgs()
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
	return inf.Targs, inf.Sig
}

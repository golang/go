// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !typeparams || !go1.17
// +build !typeparams !go1.17

package typeparams

import (
	"go/ast"
	"go/types"
)

// NOTE: doc comments must be kept in sync with typeparams.go.

// Enabled reports whether type parameters are enabled in the current build
// environment.
const Enabled = false

// UnpackIndex extracts all index expressions from e. For non-generic code this
// is always one expression: e.Index, but may be more than one expression for
// generic type instantiation.
func UnpackIndex(e *ast.IndexExpr) []ast.Expr {
	return []ast.Expr{e.Index}
}

// IsListExpr reports whether n is an *ast.ListExpr, which is a new node type
// introduced to hold type arguments for generic type instantiation.
func IsListExpr(n ast.Node) bool {
	return false
}

// ForTypeDecl extracts the (possibly nil) type parameter node list from n.
func ForTypeDecl(*ast.TypeSpec) *ast.FieldList {
	return nil
}

// ForFuncDecl extracts the (possibly nil) type parameter node list from n.
func ForFuncDecl(*ast.FuncDecl) *ast.FieldList {
	return nil
}

// ForSignature extracts the (possibly empty) type parameter object list from
// sig.
func ForSignature(*types.Signature) []*types.TypeName {
	return nil
}

// HasTypeSet reports if iface has a type set.
func HasTypeSet(*types.Interface) bool {
	return false
}

// IsComparable reports if iface is the comparable interface.
func IsComparable(*types.Interface) bool {
	return false
}

// IsConstraint reports whether iface may only be used as a type parameter
// constraint (i.e. has a type set or is the comparable interface).
func IsConstraint(*types.Interface) bool {
	return false
}

// ForNamed extracts the (possibly empty) type parameter object list from
// named.
func ForNamed(*types.Named) []*types.TypeName {
	return nil
}

// NamedTArgs extracts the (possibly empty) type argument list from named.
func NamedTArgs(*types.Named) []types.Type {
	return nil
}

// InitInferred initializes info to record inferred type information.
func InitInferred(*types.Info) {
}

// GetInferred extracts inferred type information from info for e.
//
// The expression e may have an inferred type if it is an *ast.IndexExpr
// representing partial instantiation of a generic function type for which type
// arguments have been inferred using constraint type inference, or if it is an
// *ast.CallExpr for which type type arguments have be inferred using both
// constraint type inference and function argument inference.
func GetInferred(*types.Info, ast.Expr) ([]types.Type, *types.Signature) {
	return nil, nil
}

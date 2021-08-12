// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !typeparams || !go1.18
// +build !typeparams !go1.18

package typeparams

import (
	"go/ast"
	"go/types"
)

// NOTE: doc comments must be kept in sync with typeparams.go.

// Enabled reports whether type parameters are enabled in the current build
// environment.
const Enabled = false

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
	if e, _ := n.(*ast.IndexExpr); e != nil {
		return &IndexExprData{
			X:       e.X,
			Lbrack:  e.Lbrack,
			Indices: []ast.Expr{e.Index},
			Rbrack:  e.Rbrack,
		}
	}
	return nil
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

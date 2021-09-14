// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package typeparams

import (
	"go/ast"
	"go/types"
)

// GetIndexExprData extracts data from AST nodes that represent index
// expressions.
//
// For an ast.IndexExpr, the resulting IndexExprData will have exactly one
// index expression. For an ast.IndexListExpr (go1.18+), it may have a
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
	case *ast.IndexListExpr:
		return (*IndexExprData)(e)
	}
	return nil
}

// ForTypeDecl returns n.TypeParams.
func ForTypeDecl(n *ast.TypeSpec) *ast.FieldList {
	return n.TypeParams
}

// ForFuncDecl returns n.Type.TypeParams.
func ForFuncDecl(n *ast.FuncDecl) *ast.FieldList {
	if n.Type != nil {
		return n.Type.TypeParams
	}
	return nil
}

// TypeParam is an alias for types.TypeParam
type TypeParam = types.TypeParam

// TypeParamList is an alias for types.TypeParamList
type TypeParamList = types.TypeParamList

// TypeList is an alias for types.TypeList
type TypeList = types.TypeList

// NewTypeParam calls types.NewTypeParam.
func NewTypeParam(name *types.TypeName, constraint types.Type) *TypeParam {
	return types.NewTypeParam(name, constraint)
}

// SetTypeParamConstraint calls tparam.SetConstraint(constraint).
func SetTypeParamConstraint(tparam *TypeParam, constraint types.Type) {
	tparam.SetConstraint(constraint)
}

// ForSignature returns sig.TypeParams()
func ForSignature(sig *types.Signature) *TypeParamList {
	return sig.TypeParams()
}

// SetForSignature calls sig.SetTypeParams(tparams)
func SetForSignature(sig *types.Signature, tparams []*TypeParam) {
	sig.SetTypeParams(tparams)
}

// RecvTypeParams returns sig.RecvTypeParams().
func RecvTypeParams(sig *types.Signature) *TypeParamList {
	return sig.RecvTypeParams()
}

// SetRecvTypeParams calls sig.SetRecvTypeParams(rparams).
func SetRecvTypeParams(sig *types.Signature, rparams []*TypeParam) {
	sig.SetRecvTypeParams(rparams)
}

// IsComparable calls iface.IsComparable().
func IsComparable(iface *types.Interface) bool {
	return iface.IsComparable()
}

// IsConstraint calls iface.IsConstraint().
func IsConstraint(iface *types.Interface) bool {
	return iface.IsConstraint()
}

// ForNamed extracts the (possibly empty) type parameter object list from
// named.
func ForNamed(named *types.Named) *TypeParamList {
	return named.TypeParams()
}

// SetForNamed sets the type params tparams on n. Each tparam must be of
// dynamic type *types.TypeParam.
func SetForNamed(n *types.Named, tparams []*TypeParam) {
	n.SetTypeParams(tparams)
}

// NamedTypeArgs extracts the (possibly empty) type argument list from named.
func NamedTypeArgs(named *types.Named) []types.Type {
	targs := named.TypeArgs()
	numArgs := targs.Len()

	typs := make([]types.Type, numArgs)
	for i := 0; i < numArgs; i++ {
		typs[i] = targs.At(i)
	}

	return typs
}

// Term is an alias for types.Term.
type Term = types.Term

// NewTerm calls types.NewTerm.
func NewTerm(tilde bool, typ types.Type) *Term {
	return types.NewTerm(tilde, typ)
}

// Union is an alias for types.Union
type Union = types.Union

// NewUnion calls types.NewUnion.
func NewUnion(terms []*Term) *Union {
	return types.NewUnion(terms)
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

// Environment is an alias for types.Environment.
type Environment = types.Environment

// Instantiate calls types.Instantiate.
func Instantiate(env *Environment, typ types.Type, targs []types.Type, validate bool) (types.Type, error) {
	return types.Instantiate(env, typ, targs, validate)
}

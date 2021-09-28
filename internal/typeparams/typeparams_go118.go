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

// ForTypeSpec returns n.TypeParams.
func ForTypeSpec(n *ast.TypeSpec) *ast.FieldList {
	if n == nil {
		return nil
	}
	return n.TypeParams
}

// ForFuncType returns n.TypeParams.
func ForFuncType(n *ast.FuncType) *ast.FieldList {
	if n == nil {
		return nil
	}
	return n.TypeParams
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

// NamedTypeArgs returns named.TypeArgs().
func NamedTypeArgs(named *types.Named) *TypeList {
	return named.TypeArgs()
}

// NamedTypeOrigin returns named.Orig().
func NamedTypeOrigin(named *types.Named) types.Type {
	return named.Origin()
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

// InitInstanceInfo initializes info to record information about type and
// function instances.
func InitInstanceInfo(info *types.Info) {
	info.Instances = make(map[*ast.Ident]types.Instance)
}

// GetInstance extracts information about the instantiation occurring at the
// identifier id. id should be the identifier denoting a parameterized type or
// function in an instantiation expression or function call.
func GetInstance(info *types.Info, id *ast.Ident) (*TypeList, types.Type) {
	if info.Instances != nil {
		inf := info.Instances[id]
		return inf.TypeArgs, inf.Type
	}
	return nil, nil
}

// Environment is an alias for types.Environment.
type Environment = types.Environment

// Instantiate calls types.Instantiate.
func Instantiate(env *Environment, typ types.Type, targs []types.Type, validate bool) (types.Type, error) {
	return types.Instantiate(env, typ, targs, validate)
}

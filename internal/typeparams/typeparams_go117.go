// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.18
// +build !go1.18

package typeparams

import (
	"go/ast"
	"go/types"
)

func unsupported() {
	panic("type parameters are unsupported at this go version")
}

// GetIndexExprData extracts data from *ast.IndexExpr nodes.
// For other nodes, GetIndexExprData returns nil.
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

// ForTypeDecl returns an empty field list, as type parameters on not supported
// at this Go version.
func ForTypeDecl(*ast.TypeSpec) *ast.FieldList {
	return nil
}

// ForFuncDecl returns an empty field list, as type parameters are not
// supported at this Go version.
func ForFuncDecl(*ast.FuncDecl) *ast.FieldList {
	return nil
}

// TypeParam is a placeholder type, as type parameters are not supported at
// this Go version. Its methods panic on use.
type TypeParam struct{ types.Type }

// TypeParamList is a placeholder for an empty type parameter list.
type TypeParamList struct{}

func (*TypeParamList) Len() int          { return 0 }
func (*TypeParamList) At(int) *TypeParam { unsupported(); return nil }

// TypeList is a placeholder for an empty type list.
type TypeList struct{}

func (*TypeList) Len() int          { return 0 }
func (*TypeList) At(int) types.Type { unsupported(); return nil }

// NewTypeParam is unsupported at this Go version, and panics.
func NewTypeParam(name *types.TypeName, constraint types.Type) *TypeParam {
	unsupported()
	return nil
}

// SetTypeParamConstraint is unsupported at this Go version, and panics.
func SetTypeParamConstraint(tparam *TypeParam, constraint types.Type) {
	unsupported()
}

// ForSignature returns an empty slice.
func ForSignature(*types.Signature) *TypeParamList {
	return nil
}

// SetForSignature panics if tparams is non-empty.
func SetForSignature(_ *types.Signature, tparams []*TypeParam) {
	if len(tparams) > 0 {
		unsupported()
	}
}

// RecvTypeParams returns a nil slice.
func RecvTypeParams(sig *types.Signature) *TypeParamList {
	return nil
}

// SetRecvTypeParams panics if rparams is non-empty.
func SetRecvTypeParams(sig *types.Signature, rparams []*TypeParam) {
	if len(rparams) > 0 {
		unsupported()
	}
}

// IsComparable returns false, as no interfaces are type-restricted at this Go
// version.
func IsComparable(*types.Interface) bool {
	return false
}

// IsConstraint returns false, as no interfaces are type-restricted at this Go
// version.
func IsConstraint(*types.Interface) bool {
	return false
}

// ForNamed returns an empty type parameter list, as type parameters are not
// supported at this Go version.
func ForNamed(*types.Named) *TypeParamList {
	return nil
}

// SetForNamed panics if tparams is non-empty.
func SetForNamed(_ *types.Named, tparams []*TypeParam) {
	if len(tparams) > 0 {
		unsupported()
	}
}

// NamedTypeArgs extracts the (possibly empty) type argument list from named.
func NamedTypeArgs(*types.Named) []types.Type {
	return nil
}

// Term is a placeholder type, as type parameters are not supported at this Go
// version. Its methods panic on use.
type Term struct{ types.Type }

// NewTerm is unsupported at this Go version, and panics.
func NewTerm(tilde bool, typ types.Type) *Term {
	unsupported()
	return nil
}

// Union is a placeholder type, as type parameters are not supported at this Go
// version. Its methods panic on use.
type Union struct{ types.Type }

// NewUnion is unsupported at this Go version, and panics.
func NewUnion(terms []*Term) *Union {
	unsupported()
	return nil
}

// InitInferred is a noop at this Go version.
func InitInferred(*types.Info) {
}

// GetInferred returns nothing, as type parameters are not supported at this Go
// version.
func GetInferred(*types.Info, ast.Expr) ([]types.Type, *types.Signature) {
	return nil, nil
}

// Environment is a placeholder type, as type parameters are not supported at
// this Go version.
type Environment struct{}

// Instantiate is unsupported on this Go version, and panics.
func Instantiate(env *Environment, typ types.Type, targs []types.Type, validate bool) (types.Type, error) {
	unsupported()
	return nil, nil
}

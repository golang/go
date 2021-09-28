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

// ForTypeSpec returns an empty field list, as type parameters on not supported
// at this Go version.
func ForTypeSpec(*ast.TypeSpec) *ast.FieldList {
	return nil
}

// ForFuncType returns an empty field list, as type parameters are not
// supported at this Go version.
func ForFuncType(*ast.FuncType) *ast.FieldList {
	return nil
}

// TypeParam is a placeholder type, as type parameters are not supported at
// this Go version. Its methods panic on use.
type TypeParam struct{ types.Type }

func (*TypeParam) Constraint() types.Type { unsupported(); return nil }
func (*TypeParam) Obj() *types.TypeName   { unsupported(); return nil }

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

// NamedTypeArgs returns nil.
func NamedTypeArgs(*types.Named) *TypeList {
	return nil
}

// NamedTypeOrigin is the identity method at this Go version.
func NamedTypeOrigin(named *types.Named) types.Type {
	return named
}

// Term is a placeholder type, as type parameters are not supported at this Go
// version. Its methods panic on use.
type Term struct{}

func (*Term) Tilde() bool            { unsupported(); return false }
func (*Term) Type() types.Type       { unsupported(); return nil }
func (*Term) String() string         { unsupported(); return "" }
func (*Term) Underlying() types.Type { unsupported(); return nil }

// NewTerm is unsupported at this Go version, and panics.
func NewTerm(tilde bool, typ types.Type) *Term {
	unsupported()
	return nil
}

// Union is a placeholder type, as type parameters are not supported at this Go
// version. Its methods panic on use.
type Union struct{ types.Type }

func (*Union) Len() int         { return 0 }
func (*Union) Term(i int) *Term { unsupported(); return nil }

// NewUnion is unsupported at this Go version, and panics.
func NewUnion(terms []*Term) *Union {
	unsupported()
	return nil
}

// InitInstanceInfo is a noop at this Go version.
func InitInstanceInfo(*types.Info) {}

// GetInstance returns nothing, as type parameters are not supported at this Go
// version.
func GetInstance(*types.Info, *ast.Ident) (*TypeList, types.Type) { return nil, nil }

// Environment is a placeholder type, as type parameters are not supported at
// this Go version.
type Environment struct{}

// Instantiate is unsupported on this Go version, and panics.
func Instantiate(env *Environment, typ types.Type, targs []types.Type, validate bool) (types.Type, error) {
	unsupported()
	return nil, nil
}

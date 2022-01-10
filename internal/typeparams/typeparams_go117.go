// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.18
// +build !go1.18

package typeparams

import (
	"go/ast"
	"go/token"
	"go/types"
)

func unsupported() {
	panic("type parameters are unsupported at this go version")
}

// IndexListExpr is a placeholder type, as type parameters are not supported at
// this Go version. Its methods panic on use.
type IndexListExpr struct {
	ast.Expr
	X       ast.Expr   // expression
	Lbrack  token.Pos  // position of "["
	Indices []ast.Expr // index expressions
	Rbrack  token.Pos  // position of "]"
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

func (*TypeParam) Index() int             { unsupported(); return 0 }
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

// NewSignatureType calls types.NewSignature, panicking if recvTypeParams or
// typeParams is non-empty.
func NewSignatureType(recv *types.Var, recvTypeParams, typeParams []*TypeParam, params, results *types.Tuple, variadic bool) *types.Signature {
	if len(recvTypeParams) != 0 || len(typeParams) != 0 {
		panic("signatures cannot have type parameters at this Go version")
	}
	return types.NewSignature(recv, params, results, variadic)
}

// ForSignature returns an empty slice.
func ForSignature(*types.Signature) *TypeParamList {
	return nil
}

// RecvTypeParams returns a nil slice.
func RecvTypeParams(sig *types.Signature) *TypeParamList {
	return nil
}

// IsComparable returns false, as no interfaces are type-restricted at this Go
// version.
func IsComparable(*types.Interface) bool {
	return false
}

// IsMethodSet returns true, as no interfaces are type-restricted at this Go
// version.
func IsMethodSet(*types.Interface) bool {
	return true
}

// IsImplicit returns false, as no interfaces are implicit at this Go version.
func IsImplicit(*types.Interface) bool {
	return false
}

// MarkImplicit does nothing, because this Go version does not have implicit
// interfaces.
func MarkImplicit(*types.Interface) {}

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

// Term holds information about a structural type restriction.
type Term struct {
	tilde bool
	typ   types.Type
}

func (m *Term) Tilde() bool      { return m.tilde }
func (m *Term) Type() types.Type { return m.typ }
func (m *Term) String() string {
	pre := ""
	if m.tilde {
		pre = "~"
	}
	return pre + m.typ.String()
}

// NewTerm is unsupported at this Go version, and panics.
func NewTerm(tilde bool, typ types.Type) *Term {
	return &Term{tilde, typ}
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

// Instance is a placeholder type, as type parameters are not supported at this
// Go version.
type Instance struct {
	TypeArgs *TypeList
	Type     types.Type
}

// GetInstances returns a nil map, as type parameters are not supported at this
// Go version.
func GetInstances(info *types.Info) map[*ast.Ident]Instance { return nil }

// Context is a placeholder type, as type parameters are not supported at
// this Go version.
type Context struct{}

// Instantiate is unsupported on this Go version, and panics.
func Instantiate(ctxt *Context, typ types.Type, targs []types.Type, validate bool) (types.Type, error) {
	unsupported()
	return nil, nil
}

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build typeparams
// +build typeparams

package types

import (
	"go/ast"
)

type (
	Inferred  = _Inferred
	Sum       = _Sum
	TypeParam = _TypeParam
)

func NewSum(types []Type) Type { return _NewSum(types) }

func (s *Signature) TParams() []*TypeName           { return s._TParams() }
func (s *Signature) SetTParams(tparams []*TypeName) { s._SetTParams(tparams) }

func (t *Interface) HasTypeList() bool  { return t._HasTypeList() }
func (t *Interface) IsComparable() bool { return t._IsComparable() }
func (t *Interface) IsConstraint() bool { return t._IsConstraint() }

func (t *Named) TParams() []*TypeName { return t._TParams() }
func (t *Named) TArgs() []Type        { return t._TArgs() }
func (t *Named) SetTArgs(args []Type) { t._SetTArgs(args) }

// Info is documented in api_notypeparams.go.
type Info struct {
	Types map[ast.Expr]TypeAndValue

	// Inferred maps calls of parameterized functions that use type inference to
	// the Inferred type arguments and signature of the function called. The
	// recorded "call" expression may be an *ast.CallExpr (as in f(x)), or an
	// *ast.IndexExpr (s in f[T]).
	Inferred map[ast.Expr]_Inferred

	Defs       map[*ast.Ident]Object
	Uses       map[*ast.Ident]Object
	Implicits  map[ast.Node]Object
	Selections map[*ast.SelectorExpr]*Selection
	Scopes     map[ast.Node]*Scope
	InitOrder  []*Initializer
}

func getInferred(info *Info) map[ast.Expr]_Inferred {
	return info.Inferred
}

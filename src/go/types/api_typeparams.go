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

func GetInferred(info *Info) map[ast.Expr]Inferred {
	return info._Inferred
}

func SetInferred(info *Info, inferred map[ast.Expr]Inferred) {
	info._Inferred = inferred
}

func NewSum(types []Type) Type { return _NewSum(types) }

func (s *Signature) TParams() []*TypeName           { return s._TParams() }
func (s *Signature) SetTParams(tparams []*TypeName) { s._SetTParams(tparams) }

func (t *Interface) HasTypeList() bool  { return t._HasTypeList() }
func (t *Interface) IsComparable() bool { return t._IsComparable() }
func (t *Interface) IsConstraint() bool { return t._IsConstraint() }

func (t *Named) TParams() []*TypeName { return t._TParams() }
func (t *Named) TArgs() []Type        { return t._TArgs() }
func (t *Named) SetTArgs(args []Type) { t._SetTArgs(args) }

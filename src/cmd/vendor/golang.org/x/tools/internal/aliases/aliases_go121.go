// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.22
// +build !go1.22

package aliases

import (
	"go/types"
)

// Alias is a placeholder for a go/types.Alias for <=1.21.
// It will never be created by go/types.
type Alias struct{}

func (*Alias) String() string                                { panic("unreachable") }
func (*Alias) Underlying() types.Type                        { panic("unreachable") }
func (*Alias) Obj() *types.TypeName                          { panic("unreachable") }
func Rhs(alias *Alias) types.Type                            { panic("unreachable") }
func TypeParams(alias *Alias) *types.TypeParamList           { panic("unreachable") }
func SetTypeParams(alias *Alias, tparams []*types.TypeParam) { panic("unreachable") }
func TypeArgs(alias *Alias) *types.TypeList                  { panic("unreachable") }
func Origin(alias *Alias) *Alias                             { panic("unreachable") }

// Unalias returns the type t for go <=1.21.
func Unalias(t types.Type) types.Type { return t }

func newAlias(name *types.TypeName, rhs types.Type, tparams []*types.TypeParam) *Alias {
	panic("unreachable")
}

// Enabled reports whether [NewAlias] should create [types.Alias] types.
//
// Before go1.22, this function always returns false.
func Enabled() bool { return false }

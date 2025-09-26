// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.25

package typesinternal

import "go/types"

type VarKind = types.VarKind

const (
	PackageVar = types.PackageVar
	LocalVar   = types.LocalVar
	RecvVar    = types.RecvVar
	ParamVar   = types.ParamVar
	ResultVar  = types.ResultVar
	FieldVar   = types.FieldVar
)

func GetVarKind(v *types.Var) VarKind       { return v.Kind() }
func SetVarKind(v *types.Var, kind VarKind) { v.SetKind(kind) }

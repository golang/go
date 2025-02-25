// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

// TODO(adonovan): when CL 645115 lands, define the go1.25 version of
// this API that actually does something.

import "go/types"

type VarKind uint8

const (
	_          VarKind = iota // (not meaningful)
	PackageVar                // a package-level variable
	LocalVar                  // a local variable
	RecvVar                   // a method receiver variable
	ParamVar                  // a function parameter variable
	ResultVar                 // a function result variable
	FieldVar                  // a struct field
)

func (kind VarKind) String() string {
	return [...]string{
		0:          "VarKind(0)",
		PackageVar: "PackageVar",
		LocalVar:   "LocalVar",
		RecvVar:    "RecvVar",
		ParamVar:   "ParamVar",
		ResultVar:  "ResultVar",
		FieldVar:   "FieldVar",
	}[kind]
}

// GetVarKind returns an invalid VarKind.
func GetVarKind(v *types.Var) VarKind { return 0 }

// SetVarKind has no effect.
func SetVarKind(v *types.Var, kind VarKind) {}

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package types

import (
	"go/ast"
)

type Inferred = _Inferred

func GetInferred(info *Info) map[ast.Expr]Inferred {
	return info._Inferred
}

func SetInferred(info *Info, inferred map[ast.Expr]Inferred) {
	info._Inferred = inferred
}

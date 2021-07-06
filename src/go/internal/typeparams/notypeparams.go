// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !typeparams
// +build !typeparams

package typeparams

import (
	"go/ast"
)

const Enabled = false

func PackExpr(list []ast.Expr) ast.Expr {
	switch len(list) {
	case 1:
		return list[0]
	default:
		// The parser should not attempt to pack multiple expressions into an
		// IndexExpr if type params are disabled.
		panic("multiple index expressions are unsupported without type params")
	}
}

func UnpackExpr(expr ast.Expr) []ast.Expr {
	return []ast.Expr{expr}
}

func IsListExpr(n ast.Node) bool {
	return false
}

func Get(ast.Node) *ast.FieldList {
	return nil
}

func Set(node ast.Node, params *ast.FieldList) {
}

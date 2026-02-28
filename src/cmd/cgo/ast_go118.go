// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !compiler_bootstrap
// +build !compiler_bootstrap

package main

import (
	"go/ast"
	"go/token"
)

func (f *File) walkUnexpected(x interface{}, context astContext, visit func(*File, interface{}, astContext)) {
	switch n := x.(type) {
	default:
		error_(token.NoPos, "unexpected type %T in walk", x)
		panic("unexpected type")

	case *ast.IndexListExpr:
		f.walk(&n.X, ctxExpr, visit)
		f.walk(n.Indices, ctxExpr, visit)
	}
}

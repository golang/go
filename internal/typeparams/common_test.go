// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams_test

import (
	"go/ast"
	"testing"

	"golang.org/x/tools/internal/typeparams"
)

func TestGetIndexExprData(t *testing.T) {
	x := &ast.Ident{}
	i := &ast.Ident{}

	want := &typeparams.IndexListExpr{X: x, Lbrack: 1, Indices: []ast.Expr{i}, Rbrack: 2}
	tests := map[ast.Node]bool{
		&ast.IndexExpr{X: x, Lbrack: 1, Index: i, Rbrack: 2}: true,
		want:         true,
		&ast.Ident{}: false,
	}

	for n, isIndexExpr := range tests {
		X, lbrack, indices, rbrack := typeparams.UnpackIndexExpr(n)
		if got := X != nil; got != isIndexExpr {
			t.Errorf("UnpackIndexExpr(%v) = %v, _, _, _; want nil: %t", n, x, !isIndexExpr)
		}
		if X == nil {
			continue
		}
		if X != x || lbrack != 1 || indices[0] != i || rbrack != 2 {
			t.Errorf("UnpackIndexExprData(%v) = %v, %v, %v, %v; want %+v", n, x, lbrack, indices, rbrack, want)
		}
	}
}

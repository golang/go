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

	tests := map[ast.Node]bool{
		&ast.IndexExpr{X: x, Lbrack: 1, Index: i, Rbrack: 2}: true,
		&ast.Ident{}: false,
	}
	want := &typeparams.IndexExprData{X: x, Lbrack: 1, Indices: []ast.Expr{i}, Rbrack: 2}

	for n, isIndexExpr := range tests {
		ix := typeparams.GetIndexExprData(n)
		if got := ix != nil; got != isIndexExpr {
			t.Errorf("GetIndexExprData(%+v) = %+v, want nil: %t", n, ix, !isIndexExpr)
		}
		if ix == nil {
			continue
		}
		if ix.X != x || ix.Lbrack != 1 || ix.Indices[0] != i || ix.Rbrack != 2 {
			t.Errorf("GetIndexExprData(%+v) = %+v, want %+v", n, ix, want)
		}
	}
}

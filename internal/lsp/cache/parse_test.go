// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"go/ast"
	"go/parser"
	"testing"
)

func TestArrayLength(t *testing.T) {
	tests := []struct {
		expr   string
		length int
	}{
		{`[...]int{0,1,2,3,4,5,6,7,8,9}`, 10},
		{`[...]int{9:0}`, 10},
		{`[...]int{19-10:0}`, 10},
		{`[...]int{19-10:0, 17-10:0, 18-10:0}`, 10},
	}

	for _, tt := range tests {
		expr, err := parser.ParseExpr(tt.expr)
		if err != nil {
			t.Fatal(err)
		}
		l, ok := arrayLength(expr.(*ast.CompositeLit))
		if !ok {
			t.Errorf("arrayLength did not recognize expression %#v", expr)
		}
		if l != tt.length {
			t.Errorf("arrayLength(%#v) = %v, want %v", expr, l, tt.length)
		}
	}
}

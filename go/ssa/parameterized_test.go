// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/internal/typeparams"
)

func TestIsParameterized(t *testing.T) {
	if !typeparams.Enabled {
		return
	}

	const source = `
package P
type A int
func (A) f()
func (*A) g()

type fer interface { f() }

func Apply[T fer](x T) T {
	x.f()
	return x
}

type V[T any] []T
func (v *V[T]) Push(x T) { *v = append(*v, x) }
`

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "hello.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}

	var conf types.Config
	pkg, err := conf.Check("P", fset, []*ast.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		expr string // type expression
		want bool   // expected isParameterized value
	}{
		{"A", false},
		{"*A", false},
		{"error", false},
		{"*error", false},
		{"struct{A}", false},
		{"*struct{A}", false},
		{"fer", false},
		{"Apply", true},
		{"Apply[A]", false},
		{"V", true},
		{"V[A]", false},
		{"*V[A]", false},
		{"(*V[A]).Push", false},
	} {
		tv, err := types.Eval(fset, pkg, 0, test.expr)
		if err != nil {
			t.Errorf("Eval(%s) failed: %v", test.expr, err)
		}

		param := tpWalker{seen: make(map[types.Type]bool)}
		if got := param.isParameterized(tv.Type); got != test.want {
			t.Logf("Eval(%s) returned the type %s", test.expr, tv.Type)
			t.Errorf("isParameterized(%s) = %v, want %v", test.expr, got, test.want)
		}
	}
}

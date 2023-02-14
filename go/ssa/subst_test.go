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

func TestSubst(t *testing.T) {
	if !typeparams.Enabled {
		return
	}

	const source = `
package P

type t0 int
func (t0) f()
type t1 interface{ f() }
type t2 interface{ g() }
type t3 interface{ ~int }

func Fn0[T t1](x T) T {
	x.f()
	return x
}

type A[T any] [4]T
type B[T any] []T
type C[T, S any] []struct{s S; t T}
type D[T, S any] *struct{s S; t *T}
type E[T, S any] interface{ F() (T, S) }
type F[K comparable, V any] map[K]V
type G[T any] chan *T
type H[T any] func() T
type I[T any] struct{x, y, z int; t T}
type J[T any] interface{ t1 }
type K[T any] interface{ t1; F() T }
type L[T any] interface{ F() T; J[T] }

var _ L[int] = Fn0[L[int]](nil)
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
		expr string   // type expression of Named parameterized type
		args []string // type expressions of args for named
		want string   // expected underlying value after substitution
	}{
		{"A", []string{"string"}, "[4]string"},
		{"A", []string{"int"}, "[4]int"},
		{"B", []string{"int"}, "[]int"},
		{"B", []string{"int8"}, "[]int8"},
		{"C", []string{"int8", "string"}, "[]struct{s string; t int8}"},
		{"C", []string{"string", "int8"}, "[]struct{s int8; t string}"},
		{"D", []string{"int16", "string"}, "*struct{s string; t *int16}"},
		{"E", []string{"int32", "string"}, "interface{F() (int32, string)}"},
		{"F", []string{"int64", "string"}, "map[int64]string"},
		{"G", []string{"uint64"}, "chan *uint64"},
		{"H", []string{"uintptr"}, "func() uintptr"},
		{"I", []string{"t0"}, "struct{x int; y int; z int; t P.t0}"},
		{"J", []string{"t0"}, "interface{P.t1}"},
		{"K", []string{"t0"}, "interface{F() P.t0; P.t1}"},
		{"L", []string{"t0"}, "interface{F() P.t0; P.J[P.t0]}"},
		{"L", []string{"L[t0]"}, "interface{F() P.L[P.t0]; P.J[P.L[P.t0]]}"},
	} {
		// Eval() expr for its type.
		tv, err := types.Eval(fset, pkg, 0, test.expr)
		if err != nil {
			t.Fatalf("Eval(%s) failed: %v", test.expr, err)
		}
		// Eval() test.args[i] to get the i'th type arg.
		var targs []types.Type
		for _, astr := range test.args {
			tv, err := types.Eval(fset, pkg, 0, astr)
			if err != nil {
				t.Fatalf("Eval(%s) failed: %v", astr, err)
			}
			targs = append(targs, tv.Type)
		}

		T := tv.Type.(*types.Named)

		subst := makeSubster(typeparams.NewContext(), nil, typeparams.ForNamed(T), targs, true)
		sub := subst.typ(T.Underlying())
		if got := sub.String(); got != test.want {
			t.Errorf("subst{%v->%v}.typ(%s) = %v, want %v", test.expr, test.args, T.Underlying(), got, test.want)
		}
	}
}

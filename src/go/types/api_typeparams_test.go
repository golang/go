// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build typeparams
// +build typeparams

package types_test

import (
	"fmt"
	"go/ast"
	"testing"

	. "go/types"
)

func TestInferredInfo(t *testing.T) {
	var tests = []struct {
		src   string
		fun   string
		targs []string
		sig   string
	}{
		{genericPkg + `p0; func f[T any](T); func _() { f(42) }`,
			`f`,
			[]string{`int`},
			`func(int)`,
		},
		{genericPkg + `p1; func f[T any](T) T; func _() { f('@') }`,
			`f`,
			[]string{`rune`},
			`func(rune) rune`,
		},
		{genericPkg + `p2; func f[T any](...T) T; func _() { f(0i) }`,
			`f`,
			[]string{`complex128`},
			`func(...complex128) complex128`,
		},
		{genericPkg + `p3; func f[A, B, C any](A, *B, []C); func _() { f(1.2, new(string), []byte{}) }`,
			`f`,
			[]string{`float64`, `string`, `byte`},
			`func(float64, *string, []byte)`,
		},
		{genericPkg + `p4; func f[A, B any](A, *B, ...[]B); func _() { f(1.2, new(byte)) }`,
			`f`,
			[]string{`float64`, `byte`},
			`func(float64, *byte, ...[]byte)`,
		},

		{genericPkg + `s1; func f[T any, P interface{type *T}](x T); func _(x string) { f(x) }`,
			`f`,
			[]string{`string`, `*string`},
			`func(x string)`,
		},
		{genericPkg + `s2; func f[T any, P interface{type *T}](x []T); func _(x []int) { f(x) }`,
			`f`,
			[]string{`int`, `*int`},
			`func(x []int)`,
		},
		{genericPkg + `s3; type C[T any] interface{type chan<- T}; func f[T any, P C[T]](x []T); func _(x []int) { f(x) }`,
			`f`,
			[]string{`int`, `chan<- int`},
			`func(x []int)`,
		},
		{genericPkg + `s4; type C[T any] interface{type chan<- T}; func f[T any, P C[T], Q C[[]*P]](x []T); func _(x []int) { f(x) }`,
			`f`,
			[]string{`int`, `chan<- int`, `chan<- []*chan<- int`},
			`func(x []int)`,
		},

		{genericPkg + `t1; func f[T any, P interface{type *T}]() T; func _() { _ = f[string] }`,
			`f`,
			[]string{`string`, `*string`},
			`func() string`,
		},
		{genericPkg + `t2; type C[T any] interface{type chan<- T}; func f[T any, P C[T]]() []T; func _() { _ = f[int] }`,
			`f`,
			[]string{`int`, `chan<- int`},
			`func() []int`,
		},
		{genericPkg + `t3; type C[T any] interface{type chan<- T}; func f[T any, P C[T], Q C[[]*P]]() []T; func _() { _ = f[int] }`,
			`f`,
			[]string{`int`, `chan<- int`, `chan<- []*chan<- int`},
			`func() []int`,
		},
	}

	for _, test := range tests {
		info := Info{}
		info.Inferred = make(map[ast.Expr]Inferred)
		name, err := mayTypecheck(t, "InferredInfo", test.src, &info)
		if err != nil {
			t.Errorf("package %s: %v", name, err)
			continue
		}

		// look for inferred type arguments and signature
		var targs []Type
		var sig *Signature
		for call, inf := range info.Inferred {
			var fun ast.Expr
			switch x := call.(type) {
			case *ast.CallExpr:
				fun = x.Fun
			case *ast.IndexExpr:
				fun = x.X
			default:
				panic(fmt.Sprintf("unexpected call expression type %T", call))
			}
			if ExprString(fun) == test.fun {
				targs = inf.Targs
				sig = inf.Sig
				break
			}
		}
		if targs == nil {
			t.Errorf("package %s: no inferred information found for %s", name, test.fun)
			continue
		}

		// check that type arguments are correct
		if len(targs) != len(test.targs) {
			t.Errorf("package %s: got %d type arguments; want %d", name, len(targs), len(test.targs))
			continue
		}
		for i, targ := range targs {
			if got := targ.String(); got != test.targs[i] {
				t.Errorf("package %s, %d. type argument: got %s; want %s", name, i, got, test.targs[i])
				continue
			}
		}

		// check that signature is correct
		if got := sig.String(); got != test.sig {
			t.Errorf("package %s: got %s; want %s", name, got, test.sig)
		}
	}
}

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/ast"
	"go/constant"
	"go/parser"
	"go/token"
	"go/types"
	"math/big"
	"strings"
	"testing"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/internal/typeparams"
)

func TestConstString(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestConstString requires type parameters.")
	}

	const source = `
	package P

	type Named string

	func fn() (int, bool, string) 
	func gen[T int]() {}
	`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}

	var conf types.Config
	pkg, err := conf.Check("P", fset, []*ast.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		expr     string      // type expression
		constant interface{} // constant value
		want     string      // expected String() value
	}{
		{"int", int64(0), "0:int"},
		{"int64", int64(0), "0:int64"},
		{"float32", int64(0), "0:float32"},
		{"float32", big.NewFloat(1.5), "1.5:float32"},
		{"bool", false, "false:bool"},
		{"string", "", `"":string`},
		{"Named", "", `"":P.Named`},
		{"struct{x string}", nil, "struct{x string}{}:struct{x string}"},
		{"[]int", nil, "nil:[]int"},
		{"[3]int", nil, "[3]int{}:[3]int"},
		{"*int", nil, "nil:*int"},
		{"interface{}", nil, "nil:interface{}"},
		{"interface{string}", nil, `"":interface{string}`},
		{"interface{int|int64}", nil, "0:interface{int|int64}"},
		{"interface{bool}", nil, "false:interface{bool}"},
		{"interface{bool|int}", nil, "nil:interface{bool|int}"},
		{"interface{int|string}", nil, "nil:interface{int|string}"},
		{"interface{bool|string}", nil, "nil:interface{bool|string}"},
		{"interface{struct{x string}}", nil, "nil:interface{struct{x string}}"},
		{"interface{int|int64}", int64(1), "1:interface{int|int64}"},
		{"interface{~bool}", true, "true:interface{~bool}"},
		{"interface{Named}", "lorem ipsum", `"lorem ipsum":interface{P.Named}`},
		{"func() (int, bool, string)", nil, "nil:func() (int, bool, string)"},
	} {
		// Eval() expr for its type.
		tv, err := types.Eval(fset, pkg, 0, test.expr)
		if err != nil {
			t.Fatalf("Eval(%s) failed: %v", test.expr, err)
		}
		var val constant.Value
		if test.constant != nil {
			val = constant.Make(test.constant)
		}
		c := ssa.NewConst(val, tv.Type)
		got := strings.ReplaceAll(c.String(), " | ", "|") // Accept both interface{a | b} and interface{a|b}.
		if got != test.want {
			t.Errorf("ssa.NewConst(%v, %s).String() = %v, want %v", val, tv.Type, got, test.want)
		}
	}

	// Test tuples
	fn := pkg.Scope().Lookup("fn")
	tup := fn.Type().(*types.Signature).Results()
	if got, want := ssa.NewConst(nil, tup).String(), `(0, false, ""):(int, bool, string)`; got != want {
		t.Errorf("ssa.NewConst(%v, %s).String() = %v, want %v", nil, tup, got, want)
	}

	// Test type-param
	gen := pkg.Scope().Lookup("gen")
	tp := typeparams.ForSignature(gen.Type().(*types.Signature)).At(0)
	if got, want := ssa.NewConst(nil, tp).String(), "0:T"; got != want {
		t.Errorf("ssa.NewConst(%v, %s).String() = %v, want %v", nil, tup, got, want)
	}
}

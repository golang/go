// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"strings"
	"testing"

	"simd/archsimd/_gen/specgen/specexpr"
)

func mustParseExpr(t *testing.T, s string) ast.Expr {
	t.Helper()
	expr, err := parser.ParseExpr(s)
	if err != nil {
		t.Fatalf("parser.ParseExpr(%q): %v", s, err)
	}
	return expr
}

func TestParseTypeExpr(t *testing.T) {
	tests := []struct {
		expr string
		want specexpr.Type
	}{
		{"int", specexpr.Basic{Base: "int", Bits: 0}},
		{"uint32", specexpr.Basic{Base: "uint", Bits: 32}},
		{"float64", specexpr.Basic{Base: "float", Bits: 64}},
		{"bool", specexpr.Basic{Base: "bool", Bits: 0}},
		{"Int32x4", specexpr.Vector{Elem: specexpr.Basic{Base: "int", Bits: 32}, Width: specexpr.Int(128)}},
		{"Float64s", specexpr.Vector{Elem: specexpr.Basic{Base: "float", Bits: 64}, Width: specexpr.VW()}},
		{"*int", specexpr.Pointer{Elem: specexpr.Basic{Base: "int", Bits: 0}}},
		{"*Int32x4", specexpr.Pointer{Elem: specexpr.Vector{Elem: specexpr.Basic{Base: "int", Bits: 32}, Width: specexpr.Int(128)}}},
		{"[]int8", specexpr.Slice{Elem: specexpr.Basic{Base: "int", Bits: 8}}},
		{"[4]int32", specexpr.Array{Elem: specexpr.Basic{Base: "int", Bits: 32}, Len: specexpr.Int(4)}},
		{"(int)", specexpr.Basic{Base: "int", Bits: 0}},
		{"(*Int32x4)", specexpr.Pointer{Elem: specexpr.Vector{Elem: specexpr.Basic{Base: "int", Bits: 32}, Width: specexpr.Int(128)}}},
	}

	for _, tc := range tests {
		t.Run(tc.expr, func(t *testing.T) {
			expr := mustParseExpr(t, tc.expr)
			got, err := parseTypeExpr(expr)
			if err != nil {
				t.Fatalf("parseTypeExpr(%q) unexpected error: %v", tc.expr, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("parseTypeExpr(%q) = %+v, want %+v", tc.expr, got, tc.want)
			}
		})
	}
}

func TestParseTypeExprErrors(t *testing.T) {
	bad := []struct {
		name string
		expr string
		want string
	}{
		{"unknown identifier", "unknown", "unknown type name"},
		{"any identifier", "any", "unknown type name"},
		{"Mask scalar", "Mask", "unknown type name"},
		{"w width form", "Int32w128", "unknown type name"},
		{"pointer to invalid", "*unknown", "unknown type name"},
		{"slice of invalid", "[]unknown", "unknown type name"},
		{"array non-int bound identifier", "[N]int", "array bound is not an integer literal"},
		{"array non-int bound expr", "[1+2]int", "array bound is not an integer literal"},
		{"array negative bound", "[-1]int", "array bound is not an integer literal"},
		{"array bound overflow", "[999999999999999999999999999999999999]int", "invalid array bound"},
		{"array of invalid", "[4]unknown", "unknown type name"},
		{"interface type", "interface{}", "unexpected type expression"},
		{"struct type", "struct{}", "unexpected type expression"},
		{"map type", "map[int]int", "unexpected type expression"},
		{"func type", "func()", "unexpected type expression"},
		{"channel type", "chan int", "unexpected type expression"},
	}

	for _, tc := range bad {
		t.Run(tc.name, func(t *testing.T) {
			expr := mustParseExpr(t, tc.expr)
			got, err := parseTypeExpr(expr)
			if err == nil {
				t.Fatalf("parseTypeExpr(%q) = %+v, want error containing %q", tc.expr, got, tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("parseTypeExpr(%q) error = %q, want containing %q", tc.expr, err.Error(), tc.want)
			}
		})
	}
}

func TestFieldListArgs(t *testing.T) {
	fset := token.NewFileSet()
	src := `package p
func f(a, b int, c *Int32x4) (Int32x4, int) {}
func g() (z Int32x4, err int) {}
`
	file, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("parsing test.go: %v", err)
	}
	fd := file.Decls[0].(*ast.FuncDecl)

	in, err := fieldListArgs(fd.Type.Params)
	if err != nil {
		t.Fatalf("fieldListArgs params unexpected error: %v", err)
	}
	if len(in) != 3 {
		t.Fatalf("expected 3 params, got %d", len(in))
	}
	if in[0].Name != "a" || in[1].Name != "b" || in[2].Name != "c" {
		t.Errorf("unexpected param names: %+v", in)
	}

	out, err := fieldListArgs(fd.Type.Results)
	if err != nil {
		t.Fatalf("fieldListArgs results unexpected error: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 results, got %d", len(out))
	}
	if out[0].Name != "" || out[1].Name != "" {
		t.Errorf("unexpected result names: %+v", out)
	}

	fdG := file.Decls[1].(*ast.FuncDecl)
	outG, err := fieldListArgs(fdG.Type.Results)
	if err != nil {
		t.Fatalf("fieldListArgs results unexpected error: %v", err)
	}
	if len(outG) != 2 {
		t.Fatalf("expected 2 results, got %d", len(outG))
	}
	if outG[0].Name != "z" || outG[1].Name != "err" {
		t.Errorf("unexpected result names: %+v", outG)
	}

	// Test nil FieldList
	nilArgs, err := fieldListArgs(nil)
	if err != nil || nilArgs != nil {
		t.Errorf("fieldListArgs(nil) = (%+v, %v), want (nil, nil)", nilArgs, err)
	}
}

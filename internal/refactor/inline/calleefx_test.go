// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/internal/refactor/inline"
)

// TestCalleeEffects is a unit test of the calleefx analysis.
func TestCalleeEffects(t *testing.T) {
	// Each callee must declare a function or method named f.
	const funcName = "f"

	var tests = []struct {
		descr  string
		callee string // Go source file (sans package decl) containing callee decl
		want   string // expected effects string (-1=R∞ -2=W∞)
	}{
		{
			"Assignments have unknown effects.",
			`func f(x, y int) { x = y }`,
			`[0 1 -2]`,
		},
		{
			"Reads from globals are impure.",
			`func f() { _ = g }; var g int`,
			`[-1]`,
		},
		{
			"Writes to globals have effects.",
			`func f() { g = 0 }; var g int`,
			`[-1 -2]`, // the -1 is spurious but benign
		},
		{
			"Blank assign has no effect.",
			`func f(x int) { _ = x }`,
			`[0]`,
		},
		{
			"Short decl of new var has has no effect.",
			`func f(x int) { y := x; _ = y }`,
			`[0]`,
		},
		{
			"Short decl of existing var (y) is an assignment.",
			`func f(x int) { y := x; y, z := 1, 2; _, _ = y, z }`,
			`[0 -2]`,
		},
		{
			"Unreferenced parameters are excluded.",
			`func f(x, y, z int) { _ = z + x }`,
			`[2 0]`,
		},
		{
			"Built-in len has no effect.",
			`func f(x, y string) { _ = len(y) + len(x) }`,
			`[1 0]`,
		},
		{
			"Built-in println has effects.",
			`func f(x, y int) { println(y, x) }`,
			`[1 0 -2]`,
		},
		{
			"Return has no effect, and no control successor.",
			`func f(x, y int) int { return x + y; panic(1) }`,
			`[0 1]`,
		},
		{
			"Loops (etc) have unknown effects.",
			`func f(x, y bool) { for x { _ = y } }`,
			`[0 -2 1]`,
		},
		{
			"Calls have unknown effects.",
			`func f(x, y int) { _, _, _ = x, g(), y }; func g() int`,
			`[0 -2 1]`,
		},
		{
			"Calls to some built-ins are pure.",
			`func f(x, y int) { _, _, _ = x, len("hi"), y }`,
			`[0 1]`,
		},
		{
			"Calls to some built-ins are pure (variant).",
			`func f(x, y int) { s := "hi"; _, _, _ = x, len(s), y; s = "bye" }`,
			`[0 1 -2]`,
		},
		{
			"Calls to some built-ins are pure (another variants).",
			`func f(x, y int) { s := "hi"; _, _, _ = x, len(s), y }`,
			`[0 1]`,
		},
		{
			"Reading a local var is impure but does not have effects.",
			`func f(x, y bool) { for x { _ = y } }`,
			`[0 -2 1]`,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.descr, func(t *testing.T) {
			fset := token.NewFileSet()
			mustParse := func(filename string, content any) *ast.File {
				f, err := parser.ParseFile(fset, filename, content, parser.ParseComments|parser.SkipObjectResolution)
				if err != nil {
					t.Fatalf("ParseFile: %v", err)
				}
				return f
			}

			// Parse callee file and find first func decl named f.
			calleeContent := "package p\n" + test.callee
			calleeFile := mustParse("callee.go", calleeContent)
			var decl *ast.FuncDecl
			for _, d := range calleeFile.Decls {
				if d, ok := d.(*ast.FuncDecl); ok && d.Name.Name == funcName {
					decl = d
					break
				}
			}
			if decl == nil {
				t.Fatalf("declaration of func %s not found: %s", funcName, test.callee)
			}

			info := &types.Info{
				Defs:       make(map[*ast.Ident]types.Object),
				Uses:       make(map[*ast.Ident]types.Object),
				Types:      make(map[ast.Expr]types.TypeAndValue),
				Implicits:  make(map[ast.Node]types.Object),
				Selections: make(map[*ast.SelectorExpr]*types.Selection),
				Scopes:     make(map[ast.Node]*types.Scope),
			}
			conf := &types.Config{Error: func(err error) { t.Error(err) }}
			pkg, err := conf.Check("p", fset, []*ast.File{calleeFile}, info)
			if err != nil {
				t.Fatal(err)
			}

			callee, err := inline.AnalyzeCallee(t.Logf, fset, pkg, info, decl, []byte(calleeContent))
			if err != nil {
				t.Fatal(err)
			}
			if got := fmt.Sprint(callee.Effects()); got != test.want {
				t.Errorf("for effects of %s, got %s want %s",
					test.callee, got, test.want)
			}
		})
	}
}

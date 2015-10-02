// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for Eval.

package types_test

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"internal/testenv"
	"strings"
	"testing"

	. "go/types"
)

func testEval(t *testing.T, fset *token.FileSet, pkg *Package, pos token.Pos, expr string, typ Type, typStr, valStr string) {
	gotTv, err := Eval(fset, pkg, pos, expr)
	if err != nil {
		t.Errorf("Eval(%q) failed: %s", expr, err)
		return
	}
	if gotTv.Type == nil {
		t.Errorf("Eval(%q) got nil type but no error", expr)
		return
	}

	// compare types
	if typ != nil {
		// we have a type, check identity
		if !Identical(gotTv.Type, typ) {
			t.Errorf("Eval(%q) got type %s, want %s", expr, gotTv.Type, typ)
			return
		}
	} else {
		// we have a string, compare type string
		gotStr := gotTv.Type.String()
		if gotStr != typStr {
			t.Errorf("Eval(%q) got type %s, want %s", expr, gotStr, typStr)
			return
		}
	}

	// compare values
	gotStr := ""
	if gotTv.Value != nil {
		gotStr = gotTv.Value.String()
	}
	if gotStr != valStr {
		t.Errorf("Eval(%q) got value %s, want %s", expr, gotStr, valStr)
	}
}

func TestEvalBasic(t *testing.T) {
	fset := token.NewFileSet()
	for _, typ := range Typ[Bool : String+1] {
		testEval(t, fset, nil, token.NoPos, typ.Name(), typ, "", "")
	}
}

func TestEvalComposite(t *testing.T) {
	fset := token.NewFileSet()
	for _, test := range independentTestTypes {
		testEval(t, fset, nil, token.NoPos, test.src, nil, test.str, "")
	}
}

func TestEvalArith(t *testing.T) {
	var tests = []string{
		`true`,
		`false == false`,
		`12345678 + 87654321 == 99999999`,
		`10 * 20 == 200`,
		`(1<<1000)*2 >> 100 == 2<<900`,
		`"foo" + "bar" == "foobar"`,
		`"abc" <= "bcd"`,
		`len([10]struct{}{}) == 2*5`,
	}
	fset := token.NewFileSet()
	for _, test := range tests {
		testEval(t, fset, nil, token.NoPos, test, Typ[UntypedBool], "", "true")
	}
}

func TestEvalPos(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// The contents of /*-style comments are of the form
	//	expr => value, type
	// where value may be the empty string.
	// Each expr is evaluated at the position of the comment
	// and the result is compared with the expected value
	// and type.
	var sources = []string{
		`
		package p
		import "fmt"
		import m "math"
		const c = 3.0
		type T []int
		func f(a int, s string) float64 {
			fmt.Println("calling f")
			_ = m.Pi // use package math
			const d int = c + 1
			var x int
			x = a + len(s)
			return float64(x)
			/* true => true, untyped bool */
			/* fmt.Println => , func(a ...interface{}) (n int, err error) */
			/* c => 3, untyped float */
			/* T => , p.T */
			/* a => , int */
			/* s => , string */
			/* d => 4, int */
			/* x => , int */
			/* d/c => 1, int */
			/* c/2 => 3/2, untyped float */
			/* m.Pi < m.E => false, untyped bool */
		}
		`,
		`
		package p
		/* c => 3, untyped float */
		type T1 /* T1 => , p.T1 */ struct {}
		var v1 /* v1 => , int */ = 42
		func /* f1 => , func(v1 float64) */ f1(v1 float64) {
			/* f1 => , func(v1 float64) */
			/* v1 => , float64 */
			var c /* c => 3, untyped float */ = "foo" /* c => , string */
			{
				var c struct {
					c /* c => , string */ int
				}
				/* c => , struct{c int} */
				_ = c
			}
			_ = func(a, b, c int) /* c => , string */ {
				/* c => , int */
			}
			_ = c
			type FT /* FT => , p.FT */ interface{}
		}
		`,
		`
		package p
		/* T => , p.T */
		`,
	}

	fset := token.NewFileSet()
	var files []*ast.File
	for i, src := range sources {
		file, err := parser.ParseFile(fset, "p", src, parser.ParseComments)
		if err != nil {
			t.Fatalf("could not parse file %d: %s", i, err)
		}
		files = append(files, file)
	}

	conf := Config{Importer: importer.Default()}
	pkg, err := conf.Check("p", fset, files, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, file := range files {
		for _, group := range file.Comments {
			for _, comment := range group.List {
				s := comment.Text
				if len(s) >= 4 && s[:2] == "/*" && s[len(s)-2:] == "*/" {
					str, typ := split(s[2:len(s)-2], ", ")
					str, val := split(str, "=>")
					testEval(t, fset, pkg, comment.Pos(), str, nil, typ, val)
				}
			}
		}
	}
}

// split splits string s at the first occurrence of s.
func split(s, sep string) (string, string) {
	i := strings.Index(s, sep)
	return strings.TrimSpace(s[:i]), strings.TrimSpace(s[i+len(sep):])
}

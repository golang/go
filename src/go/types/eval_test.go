// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for Eval.

package types_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"internal/godebug"
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
		gotStr = gotTv.Value.ExactString()
	}
	if gotStr != valStr {
		t.Errorf("Eval(%q) got value %s, want %s", expr, gotStr, valStr)
	}
}

func TestEvalBasic(t *testing.T) {
	fset := token.NewFileSet()
	for _, typ := range Typ[Bool : String+1] {
		testEval(t, fset, nil, nopos, typ.Name(), typ, "", "")
	}
}

func TestEvalComposite(t *testing.T) {
	fset := token.NewFileSet()
	for _, test := range independentTestTypes {
		testEval(t, fset, nil, nopos, test.src, nil, test.str, "")
	}
}

func TestEvalArith(t *testing.T) {
	var tests = []string{
		`true`,
		`false == false`,
		`12345678 + 87654321 == 99999999`,
		`10 * 20 == 200`,
		`(1<<500)*2 >> 100 == 2<<400`,
		`"foo" + "bar" == "foobar"`,
		`"abc" <= "bcd"`,
		`len([10]struct{}{}) == 2*5`,
	}
	fset := token.NewFileSet()
	for _, test := range tests {
		testEval(t, fset, nil, nopos, test, Typ[UntypedBool], "", "true")
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
			/* fmt.Println => , func(a ...any) (n int, err error) */
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
			_ = func(a, b, c int /* c => , string */) /* c => , int */ {
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
		`
		package p
		import "io"
		type R = io.Reader
		func _() {
			/* interface{R}.Read => , func(_ interface{io.Reader}, p []byte) (n int, err error) */
			_ = func() {
				/* interface{io.Writer}.Write => , func(_ interface{io.Writer}, p []byte) (n int, err error) */
				type io interface {} // must not shadow io in line above
			}
			type R interface {} // must not shadow R in first line of this function body
		}
		`,
	}

	fset := token.NewFileSet()
	var files []*ast.File
	for i, src := range sources {
		file, err := parser.ParseFile(fset, "p", src, parser.ParseComments)
		if err != nil {
			t.Fatalf("could not parse file %d: %s", i, err)
		}

		// Materialized aliases give a different (better)
		// result for the final test, so skip it for now.
		// TODO(adonovan): reenable when gotypesalias=1 is the default.
		switch gotypesalias.Value() {
		case "", "1":
			if strings.Contains(src, "interface{R}.Read") {
				continue
			}
		}

		files = append(files, file)
	}

	conf := Config{Importer: defaultImporter(fset)}
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

// gotypesalias controls the use of Alias types.
var gotypesalias = godebug.New("gotypesalias")

// split splits string s at the first occurrence of s, trimming spaces.
func split(s, sep string) (string, string) {
	before, after, _ := strings.Cut(s, sep)
	return strings.TrimSpace(before), strings.TrimSpace(after)
}

func TestCheckExpr(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// Each comment has the form /* expr => object */:
	// expr is an identifier or selector expression that is passed
	// to CheckExpr at the position of the comment, and object is
	// the string form of the object it denotes.
	const src = `
package p

import "fmt"

const c = 3.0
type T []int
type S struct{ X int }

func f(a int, s string) S {
	/* fmt.Println => func fmt.Println(a ...any) (n int, err error) */
	/* fmt.Stringer.String => func (fmt.Stringer).String() string */
	fmt.Println("calling f")

	var fmt struct{ Println int }
	/* fmt => var fmt struct{Println int} */
	/* fmt.Println => field Println int */
	/* f(1, "").X => field X int */
	fmt.Println = 1

	/* append => builtin append */

	/* new(S).X => field X int */

	return S{}
}`

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	conf := Config{Importer: defaultImporter(fset)}
	pkg, err := conf.Check("p", fset, []*ast.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}

	checkExpr := func(pos token.Pos, str string) (Object, error) {
		expr, err := parser.ParseExprFrom(fset, "eval", str, 0)
		if err != nil {
			return nil, err
		}

		info := &Info{
			Uses:       make(map[*ast.Ident]Object),
			Selections: make(map[*ast.SelectorExpr]*Selection),
		}
		if err := CheckExpr(fset, pkg, pos, expr, info); err != nil {
			return nil, fmt.Errorf("CheckExpr(%q) failed: %s", str, err)
		}
		switch expr := expr.(type) {
		case *ast.Ident:
			if obj, ok := info.Uses[expr]; ok {
				return obj, nil
			}
		case *ast.SelectorExpr:
			if sel, ok := info.Selections[expr]; ok {
				return sel.Obj(), nil
			}
			if obj, ok := info.Uses[expr.Sel]; ok {
				return obj, nil // qualified identifier
			}
		}
		return nil, fmt.Errorf("no object for %s", str)
	}

	for _, group := range f.Comments {
		for _, comment := range group.List {
			s := comment.Text
			if len(s) >= 4 && strings.HasPrefix(s, "/*") && strings.HasSuffix(s, "*/") {
				pos := comment.Pos()
				expr, wantObj := split(s[2:len(s)-2], "=>")
				obj, err := checkExpr(pos, expr)
				if err != nil {
					t.Errorf("%s: %s", fset.Position(pos), err)
					continue
				}
				if obj.String() != wantObj {
					t.Errorf("%s: checkExpr(%s) = %s, want %v",
						fset.Position(pos), expr, obj, wantObj)
				}
			}
		}
	}
}

func TestIssue65898(t *testing.T) {
	const src = `
package p
func _[A any](A) {}
`

	fset := token.NewFileSet()
	f := mustParse(fset, src)

	var conf types.Config
	pkg, err := conf.Check(pkgName(src), fset, []*ast.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, d := range f.Decls {
		if fun, _ := d.(*ast.FuncDecl); fun != nil {
			// type parameter A is not found at the start of the function type
			if err := types.CheckExpr(fset, pkg, fun.Type.Pos(), fun.Type, nil); err == nil || !strings.Contains(err.Error(), "undefined") {
				t.Fatalf("got %s, want undefined error", err)
			}
			// type parameter A must be found at the end of the function type
			if err := types.CheckExpr(fset, pkg, fun.Type.End(), fun.Type, nil); err != nil {
				t.Fatal(err)
			}
		}
	}
}

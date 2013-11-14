// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for Eval.

package types_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	_ "code.google.com/p/go.tools/go/gcimporter"
	. "code.google.com/p/go.tools/go/types"
)

func testEval(t *testing.T, pkg *Package, scope *Scope, str string, typ Type, typStr, valStr string) {
	gotTyp, gotVal, err := Eval(str, pkg, scope)
	if err != nil {
		t.Errorf("Eval(%q) failed: %s", str, err)
		return
	}
	if gotTyp == nil {
		t.Errorf("Eval(%q) got nil type but no error", str)
		return
	}

	// compare types
	if typ != nil {
		// we have a type, check identity
		if !IsIdentical(gotTyp, typ) {
			t.Errorf("Eval(%q) got type %s, want %s", str, gotTyp, typ)
			return
		}
	} else {
		// we have a string, compare type string
		gotStr := gotTyp.String()
		if gotStr != typStr {
			t.Errorf("Eval(%q) got type %s, want %s", str, gotStr, typStr)
			return
		}
	}

	// compare values
	gotStr := ""
	if gotVal != nil {
		gotStr = gotVal.String()
	}
	if gotStr != valStr {
		t.Errorf("Eval(%q) got value %s, want %s", str, gotStr, valStr)
	}
}

func TestEvalBasic(t *testing.T) {
	for _, typ := range Typ[Bool : String+1] {
		testEval(t, nil, nil, typ.Name(), typ, "", "")
	}
}

func TestEvalComposite(t *testing.T) {
	for _, test := range independentTestTypes {
		testEval(t, nil, nil, test.src, nil, test.str, "")
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
	for _, test := range tests {
		testEval(t, nil, nil, test, Typ[UntypedBool], "", "true")
	}
}

func TestEvalContext(t *testing.T) {
	src := `
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
}
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "p", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	pkg, err := Check("p", fset, []*ast.File{file})
	if err != nil {
		t.Fatal(err)
	}

	pkgScope := pkg.Scope()
	if n := pkgScope.NumChildren(); n != 1 {
		t.Fatalf("got %d file scopes, want 1", n)
	}

	fileScope := pkgScope.Child(0)
	if n := fileScope.NumChildren(); n != 1 {
		t.Fatalf("got %d functions scopes, want 1", n)
	}

	funcScope := fileScope.Child(0)

	var tests = []string{
		`true => true, untyped boolean`,
		`fmt.Println => , func(a·3 ...interface{}) (n·1 int, err·2 error)`,
		`c => 3, untyped float`,
		`T => , p.T`,
		`a => , int`,
		`s => , string`,
		`d => 4, int`,
		`x => , int`,
		`d/c => 1, int`,
		`c/2 => 3/2, untyped float`,
		`m.Pi < m.E => false, untyped boolean`,
	}
	for _, test := range tests {
		str, typ := split(test, ", ")
		str, val := split(str, "=>")
		testEval(t, pkg, funcScope, str, nil, typ, val)
	}
}

// split splits string s at the first occurrence of s.
func split(s, sep string) (string, string) {
	i := strings.Index(s, sep)
	return strings.TrimSpace(s[:i]), strings.TrimSpace(s[i+len(sep):])
}

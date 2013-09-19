// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(gri) This file needs to be expanded significantly.

package types

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

func pkgFor(path, source string, info *Info) (*Package, error) {
	fset = token.NewFileSet()
	f, err := parser.ParseFile(fset, path, source, 0)
	if err != nil {
		return nil, err
	}

	var conf Config
	pkg, err := conf.Check(path, fset, []*ast.File{f}, info)
	if err != nil {
		return nil, err
	}

	return pkg, nil
}

func mustTypecheck(t *testing.T, path, source string, info *Info) *Package {
	pkg, err := pkgFor(path, source, info)
	if err != nil {
		t.Fatalf("%s: didn't type-check (%s)", path, err)
	}
	return pkg
}

func TestCommaOkTypes(t *testing.T) {
	var tests = []struct {
		src  string
		expr string // comma-ok expression string
		typ  string // typestring of comma-ok value
	}{
		{`package p; var x interface{}; var _, _ = x.(int)`,
			`x.(int)`,
			`(int, bool)`,
		},
		{`package p; var x interface{}; func _() { _, _ = x.(int) }`,
			`x.(int)`,
			`(int, bool)`,
		},
		{`package p; type mybool bool; var m map[string]complex128; var b mybool; func _() { _, b = m["foo"] }`,
			`m["foo"]`,
			`(complex128, p.mybool)`,
		},
		{`package p; var c chan string; var _, _ = <-c`,
			`<-c`,
			`(string, bool)`,
		},
	}

	for i, test := range tests {
		path := fmt.Sprintf("CommaOkTypes%d", i)
		info := Info{Types: make(map[ast.Expr]Type)}
		mustTypecheck(t, path, test.src, &info)

		// look for comma-ok expression type
		var typ Type
		for e, t := range info.Types {
			if exprString(e) == test.expr {
				typ = t
				break
			}
		}
		if typ == nil {
			t.Errorf("%s: no type found for %s", path, test.expr)
			continue
		}

		// check that type is correct
		if got := typ.String(); got != test.typ {
			t.Errorf("%s: got %s; want %s", path, got, test.typ)
		}
	}
}

func TestScopesInfo(t *testing.T) {
	var tests = []struct {
		src    string
		scopes []string // list of scope descriptors of the form kind:varlist
	}{
		{`package p`, []string{
			"file:",
		}},
		{`package p; import ( "fmt"; m "math"; _ "os" ); var ( _ = fmt.Println; _ = m.Pi )`, []string{
			"file:fmt m",
		}},
		{`package p; func _() {}`, []string{
			"file:", "func:",
		}},
		{`package p; func _(x, y int) {}`, []string{
			"file:", "func:x y",
		}},
		{`package p; func _(x, y int) { x, z := 1, 2; _ = z }`, []string{
			"file:", "func:x y z", // redeclaration of x
		}},
		{`package p; func _(x, y int) (u, _ int) { return }`, []string{
			"file:", "func:u x y",
		}},
		{`package p; func _() { { var x int; _ = x } }`, []string{
			"file:", "func:", "block:x",
		}},
		{`package p; func _() { if true {} }`, []string{
			"file:", "func:", "if:", "block:",
		}},
		{`package p; func _() { if x := 0; x < 0 { y := x; _ = y } }`, []string{
			"file:", "func:", "if:x", "block:y",
		}},
		{`package p; func _() { switch x := 0; x {} }`, []string{
			"file:", "func:", "switch:x",
		}},
		{`package p; func _() { switch x := 0; x { case 1: y := x; _ = y; default: }}`, []string{
			"file:", "func:", "switch:x", "case:y", "case:",
		}},
		{`package p; func _(t interface{}) { switch t.(type) {} }`, []string{
			"file:", "func:t", "type switch:",
		}},
		{`package p; func _(t interface{}) { switch t := t; t.(type) {} }`, []string{
			"file:", "func:t", "type switch:t",
		}},
		{`package p; func _(t interface{}) { switch x := t.(type) { case int: _ = x } }`, []string{
			"file:", "func:t", "type switch:", "case:x", // x implicitly declared
		}},
		{`package p; func _() { select{} }`, []string{
			"file:", "func:",
		}},
		{`package p; func _(c chan int) { select{ case <-c: } }`, []string{
			"file:", "func:c", "comm:",
		}},
		{`package p; func _(c chan int) { select{ case i := <-c: x := i; _ = x} }`, []string{
			"file:", "func:c", "comm:i x",
		}},
		{`package p; func _() { for{} }`, []string{
			"file:", "func:", "for:", "block:",
		}},
		{`package p; func _(n int) { for i := 0; i < n; i++ { _ = i } }`, []string{
			"file:", "func:n", "for:i", "block:",
		}},
		{`package p; func _(a []int) { for i := range a { _ = i} }`, []string{
			"file:", "func:a", "range:i", "block:",
		}},
		{`package p; var s int; func _(a []int) { for i, x := range a { s += x; _ = i } }`, []string{
			"file:", "func:a", "range:i x", "block:",
		}},
	}

	for i, test := range tests {
		path := fmt.Sprintf("ScopesInfo%d", i)
		info := Info{Scopes: make(map[ast.Node]*Scope)}
		mustTypecheck(t, path, test.src, &info)

		// number of scopes must match
		if len(info.Scopes) != len(test.scopes) {
			t.Errorf("%s: got %d scopes; want %d", path, len(info.Scopes), len(test.scopes))
		}

		// scope descriptions must match
		for node, scope := range info.Scopes {
			kind := "<unknown node kind>"
			switch node.(type) {
			case *ast.File:
				kind = "file"
			case *ast.FuncType:
				kind = "func"
			case *ast.BlockStmt:
				kind = "block"
			case *ast.IfStmt:
				kind = "if"
			case *ast.SwitchStmt:
				kind = "switch"
			case *ast.TypeSwitchStmt:
				kind = "type switch"
			case *ast.CaseClause:
				kind = "case"
			case *ast.CommClause:
				kind = "comm"
			case *ast.ForStmt:
				kind = "for"
			case *ast.RangeStmt:
				kind = "range"
			}

			// look for matching scope description
			desc := kind + ":" + strings.Join(scope.Names(), " ")
			found := false
			for _, d := range test.scopes {
				if desc == d {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("%s: no matching scope found for %s", path, desc)
			}
		}
	}
}

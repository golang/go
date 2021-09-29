// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil_test

// This file defines tests of PathEnclosingInterval.

// TODO(adonovan): exhaustive tests that run over the whole input
// tree, not just handcrafted examples.

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/typeparams"
)

// pathToString returns a string containing the concrete types of the
// nodes in path.
func pathToString(path []ast.Node) string {
	var buf bytes.Buffer
	fmt.Fprint(&buf, "[")
	for i, n := range path {
		if i > 0 {
			fmt.Fprint(&buf, " ")
		}
		fmt.Fprint(&buf, strings.TrimPrefix(fmt.Sprintf("%T", n), "*ast."))
	}
	fmt.Fprint(&buf, "]")
	return buf.String()
}

// findInterval parses input and returns the [start, end) positions of
// the first occurrence of substr in input.  f==nil indicates failure;
// an error has already been reported in that case.
//
func findInterval(t *testing.T, fset *token.FileSet, input, substr string) (f *ast.File, start, end token.Pos) {
	f, err := parser.ParseFile(fset, "<input>", input, 0)
	if err != nil {
		t.Errorf("parse error: %s", err)
		return
	}

	i := strings.Index(input, substr)
	if i < 0 {
		t.Errorf("%q is not a substring of input", substr)
		f = nil
		return
	}

	filePos := fset.File(f.Package)
	return f, filePos.Pos(i), filePos.Pos(i + len(substr))
}

// Common input for following tests.
var input = makeInput()

func makeInput() string {
	src := `
// Hello.
package main
import "fmt"
func f() {}
func main() {
	z := (x + y) // add them
        f() // NB: ExprStmt and its CallExpr have same Pos/End
}
`

	if typeparams.Enabled {
		src += `
func g[A any, P interface{ctype1| ~ctype2}](a1 A, p1 P) {}

type PT[T constraint] struct{ t T }

var v GT[targ1]

var h = g[ targ2, targ3]
`
	}
	return src
}

func TestPathEnclosingInterval_Exact(t *testing.T) {
	type testCase struct {
		substr string // first occurrence of this string indicates interval
		node   string // complete text of expected containing node
	}

	dup := func(s string) testCase { return testCase{s, s} }
	// For the exact tests, we check that a substring is mapped to
	// the canonical string for the node it denotes.
	tests := []testCase{
		{"package",
			input[11 : len(input)-1]},
		{"\npack",
			input[11 : len(input)-1]},
		dup("main"),
		{"import",
			"import \"fmt\""},
		dup("\"fmt\""),
		{"\nfunc f() {}\n",
			"func f() {}"},
		{"x ",
			"x"},
		{" y",
			"y"},
		dup("z"),
		{" + ",
			"x + y"},
		{" :=",
			"z := (x + y)"},
		dup("x + y"),
		dup("(x + y)"),
		{" (x + y) ",
			"(x + y)"},
		{" (x + y) // add",
			"(x + y)"},
		{"func",
			"func f() {}"},
		dup("func f() {}"),
		{"\nfun",
			"func f() {}"},
		{" f",
			"f"},
	}
	if typeparams.Enabled {
		tests = append(tests, []testCase{
			dup("[A any, P interface{ctype1| ~ctype2}]"),
			{"[", "[A any, P interface{ctype1| ~ctype2}]"},
			dup("A"),
			{" any", "any"},
			dup("ctype1"),
			{"|", "ctype1| ~ctype2"},
			dup("ctype2"),
			{"~", "~ctype2"},
			dup("~ctype2"),
			{" ~ctype2", "~ctype2"},
			{"]", "[A any, P interface{ctype1| ~ctype2}]"},
			dup("a1"),
			dup("a1 A"),
			dup("(a1 A, p1 P)"),
			dup("type PT[T constraint] struct{ t T }"),
			dup("PT"),
			dup("[T constraint]"),
			dup("constraint"),
			dup("targ1"),
			{" targ2", "targ2"},
			dup("g[ targ2, targ3]"),
		}...)
	}
	for _, test := range tests {
		f, start, end := findInterval(t, new(token.FileSet), input, test.substr)
		if f == nil {
			continue
		}

		path, exact := astutil.PathEnclosingInterval(f, start, end)
		if !exact {
			t.Errorf("PathEnclosingInterval(%q) not exact", test.substr)
			continue
		}

		if len(path) == 0 {
			if test.node != "" {
				t.Errorf("PathEnclosingInterval(%q).path: got [], want %q",
					test.substr, test.node)
			}
			continue
		}

		if got := input[path[0].Pos():path[0].End()]; got != test.node {
			t.Errorf("PathEnclosingInterval(%q): got %q, want %q (path was %s)",
				test.substr, got, test.node, pathToString(path))
			continue
		}
	}
}

func TestPathEnclosingInterval_Paths(t *testing.T) {
	type testCase struct {
		substr string // first occurrence of this string indicates interval
		path   string // the pathToString(),exact of the expected path
	}
	// For these tests, we check only the path of the enclosing
	// node, but not its complete text because it's often quite
	// large when !exact.
	tests := []testCase{
		{"// add",
			"[BlockStmt FuncDecl File],false"},
		{"(x + y",
			"[ParenExpr AssignStmt BlockStmt FuncDecl File],false"},
		{"x +",
			"[BinaryExpr ParenExpr AssignStmt BlockStmt FuncDecl File],false"},
		{"z := (x",
			"[AssignStmt BlockStmt FuncDecl File],false"},
		{"func f",
			"[FuncDecl File],false"},
		{"func f()",
			"[FuncDecl File],false"},
		{" f()",
			"[FuncDecl File],false"},
		{"() {}",
			"[FuncDecl File],false"},
		{"// Hello",
			"[File],false"},
		{" f",
			"[Ident FuncDecl File],true"},
		{"func ",
			"[FuncDecl File],true"},
		{"mai",
			"[Ident File],true"},
		{"f() // NB",
			"[CallExpr ExprStmt BlockStmt FuncDecl File],true"},
	}
	if typeparams.Enabled {
		tests = append(tests, []testCase{
			{" any", "[Ident Field FieldList FuncDecl File],true"},
			{"|", "[BinaryExpr Field FieldList InterfaceType Field FieldList FuncDecl File],true"},
			{"ctype2",
				"[Ident UnaryExpr BinaryExpr Field FieldList InterfaceType Field FieldList FuncDecl File],true"},
			{"a1", "[Ident Field FieldList FuncDecl File],true"},
			{"PT[T constraint]", "[TypeSpec GenDecl File],false"},
			{"[T constraint]", "[FieldList TypeSpec GenDecl File],true"},
			{"targ2", "[Ident IndexListExpr ValueSpec GenDecl File],true"},
		}...)
	}
	for _, test := range tests {
		f, start, end := findInterval(t, new(token.FileSet), input, test.substr)
		if f == nil {
			continue
		}

		path, exact := astutil.PathEnclosingInterval(f, start, end)
		if got := fmt.Sprintf("%s,%v", pathToString(path), exact); got != test.path {
			t.Errorf("PathEnclosingInterval(%q): got %q, want %q",
				test.substr, got, test.path)
			continue
		}
	}
}

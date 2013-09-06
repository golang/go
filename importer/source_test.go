// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer_test

// This file defines tests of source utilities.

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

	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
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
	f, err := parser.ParseFile(fset, "<input>", input, parser.DeclarationErrors)
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
const input = `
// Hello.
package main
import "fmt"
func f() {}
func main() {
	z := (x + y) // add them
        f() // NB: ExprStmt and its CallExpr have same Pos/End
}
`

func TestPathEnclosingInterval_Exact(t *testing.T) {
	// For the exact tests, we check that a substring is mapped to
	// the canonical string for the node it denotes.
	tests := []struct {
		substr string // first occurrence of this string indicates interval
		node   string // complete text of expected containing node
	}{
		{"package",
			input[11 : len(input)-1]},
		{"\npack",
			input[11 : len(input)-1]},
		{"main",
			"main"},
		{"import",
			"import \"fmt\""},
		{"\"fmt\"",
			"\"fmt\""},
		{"\nfunc f() {}\n",
			"func f() {}"},
		{"x ",
			"x"},
		{" y",
			"y"},
		{"z",
			"z"},
		{" + ",
			"x + y"},
		{" :=",
			"z := (x + y)"},
		{"x + y",
			"x + y"},
		{"(x + y)",
			"(x + y)"},
		{" (x + y) ",
			"(x + y)"},
		{" (x + y) // add",
			"(x + y)"},
		{"func",
			"func f() {}"},
		{"func f() {}",
			"func f() {}"},
		{"\nfun",
			"func f() {}"},
		{" f",
			"f"},
	}
	for _, test := range tests {
		f, start, end := findInterval(t, new(token.FileSet), input, test.substr)
		if f == nil {
			continue
		}

		path, exact := importer.PathEnclosingInterval(f, start, end)
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
	// For these tests, we check only the path of the enclosing
	// node, but not its complete text because it's often quite
	// large when !exact.
	tests := []struct {
		substr string // first occurrence of this string indicates interval
		path   string // the pathToString(),exact of the expected path
	}{
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
	for _, test := range tests {
		f, start, end := findInterval(t, new(token.FileSet), input, test.substr)
		if f == nil {
			continue
		}

		path, exact := importer.PathEnclosingInterval(f, start, end)
		if got := fmt.Sprintf("%s,%v", pathToString(path), exact); got != test.path {
			t.Errorf("PathEnclosingInterval(%q): got %q, want %q",
				test.substr, got, test.path)
			continue
		}
	}
}

// -------- Tests of source.go -----------------------------------------

func TestEnclosingFunction(t *testing.T) {
	tests := []struct {
		input  string // the input file
		substr string // first occurrence of this string denotes interval
		fn     string // name of expected containing function
	}{
		// We use distinctive numbers as syntactic landmarks.

		// Ordinary function:
		{`package main
		  func f() { println(1003) }`,
			"100", "main.f"},
		// Methods:
		{`package main
                  type T int
		  func (t T) f() { println(200) }`,
			"200", "(main.T).f"},
		// Function literal:
		{`package main
		  func f() { println(func() { print(300) }) }`,
			"300", "func@2.24"},
		// Doubly nested
		{`package main
		  func f() { println(func() { print(func() { print(350) })})}`,
			"350", "func@2.39"},
		// Implicit init for package-level var initializer.
		{"package main; var a = 400", "400", "main.init"},
		// No code for constants:
		{"package main; const a = 500", "500", "(none)"},
		// Explicit init()
		{"package main; func init() { println(600) }", "600", "main.init"},
		// Multiple explicit init functions:
		{`package main
		  func init() { println("foo") }
		  func init() { println(800) }`,
			"800", "main.init"},
		// init() containing FuncLit.
		{`package main
		  func init() { println(func(){print(900)}) }`,
			"900", "func@2.27"},
	}
	for _, test := range tests {
		imp := importer.New(new(importer.Config)) // (NB: no go/build.Config)
		f, start, end := findInterval(t, imp.Fset, test.input, test.substr)
		if f == nil {
			continue
		}
		path, exact := importer.PathEnclosingInterval(f, start, end)
		if !exact {
			t.Errorf("EnclosingFunction(%q) not exact", test.substr)
			continue
		}
		mainInfo := imp.LoadMainPackage(f)
		prog := ssa.NewProgram(imp.Fset, 0)
		if err := prog.CreatePackages(imp); err != nil {
			t.Error(err)
			continue
		}
		pkg := prog.Package(mainInfo.Pkg)
		pkg.Build()

		name := "(none)"
		fn := ssa.EnclosingFunction(pkg, path)
		if fn != nil {
			name = fn.String()
		}

		if name != test.fn {
			t.Errorf("EnclosingFunction(%q in %q) got %s, want %s",
				test.substr, test.input, name, test.fn)
			continue
		}

		// While we're here: test HasEnclosingFunction.
		if has := ssa.HasEnclosingFunction(pkg, path); has != (fn != nil) {
			t.Errorf("HasEnclosingFunction(%q in %q) got %v, want %v",
				test.substr, test.input, has, fn != nil)
			continue
		}
	}
}

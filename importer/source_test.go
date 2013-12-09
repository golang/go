// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer_test

// This file defines tests of source utilities.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	"code.google.com/p/go.tools/astutil"
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
)

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
		{"package main; func init() { println(600) }", "600", "main.init$1"},
		// Multiple explicit init functions:
		{`package main
		  func init() { println("foo") }
		  func init() { println(800) }`,
			"800", "main.init$2"},
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
		path, exact := astutil.PathEnclosingInterval(f, start, end)
		if !exact {
			t.Errorf("EnclosingFunction(%q) not exact", test.substr)
			continue
		}
		mainInfo := imp.CreatePackage("main", f)
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

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func TestSortImportsUpdatesFileImportsField(t *testing.T) {
	t.Run("one import statement", func(t *testing.T) {
		const src = `package test

import (
	"test"
	"test" // test comment
)
`

		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments|parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}

		ast.SortImports(fset, f)

		// Check that the duplicate import spec is eliminated.
		importDeclSpecCount := len(f.Decls[0].(*ast.GenDecl).Specs)
		if importDeclSpecCount != 1 {
			t.Fatalf("len(f.Decls[0].(*ast.GenDecl).Specs) = %v; want = 1", importDeclSpecCount)
		}

		// Check that File.Imports is consistent.
		if len(f.Imports) != 1 {
			t.Fatalf("len(f.Imports) = %v; want = 1", len(f.Imports))
		}
	})

	t.Run("multiple import statements", func(t *testing.T) {
		const src = `package test

import "unsafe"

import (
	"package"
	"package"
)

import (
	"test"
	"test"
)
`

		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments|parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}

		ast.SortImports(fset, f)

		// Check that three single-spec import decls remain.
		for i := range 3 {
			importDeclSpecCount := len(f.Decls[i].(*ast.GenDecl).Specs)
			if importDeclSpecCount != 1 {
				t.Fatalf("len(f.Decls[%v].(*ast.GenDecl).Specs) = %v; want = 1", i, importDeclSpecCount)
			}
		}

		// Check that File.Imports is consistent.
		if len(f.Imports) != 3 {
			t.Fatalf("len(f.Imports) = %v; want = 3", len(f.Imports))
		}
	})
}

func TestIssue69183(t *testing.T) {
	const src = `package A
import (
"a"//a
"a")
`
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, "test.go", src, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	ast.SortImports(fs, f) // should not panic
}

func TestSortImportsSameLastLine(t *testing.T) {
	const src = `package A
import (
"a"//a
"a")
func a() {}
`

	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, "test.go", src, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	ast.SortImports(fs, f)
	fd := f.Decls[1].(*ast.FuncDecl)
	fdPos := fs.Position(fd.Pos())
	if fdPos.Column != 1 {
		t.Errorf("invalid fdPos.Column = %v; want = 1", fdPos.Column)
	}
	if fdPos.Line != 5 {
		t.Errorf("invalid fdPos.Line = %v; want = 5", fdPos.Line)
	}
}

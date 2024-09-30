// Tests is a helper package to avoid cyclic dependency between go/ast and go/parser.
package tests

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

		importDeclSpecCount := len(f.Decls[0].(*ast.GenDecl).Specs)
		if importDeclSpecCount != 1 {
			t.Fatalf("len(f.Decls[0].(*ast.GenDecl).Specs) = %v; want = 1", importDeclSpecCount)
		}

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

		for i := range 3 {
			importDeclSpecCount := len(f.Decls[i].(*ast.GenDecl).Specs)
			if importDeclSpecCount != 1 {
				t.Fatalf("len(f.Decls[%v].(*ast.GenDecl).Specs) = %v; want = 1", i, importDeclSpecCount)
			}
		}

		if len(f.Imports) != 3 {
			t.Fatalf("len(f.Imports) = %v; want = 3", len(f.Imports))
		}
	})
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

var sources = []string{
	`package p
	import "fmt"
	import "math"
	const pi = math.Pi
	func sin(x float64) float64 {
		return math.Sin(x)
	}
	var Println = fmt.Println
	`,
	`package p
	import "fmt"
	func f() string {
		return fmt.Sprintf("%d", g())
	}
	func g() (x int) { return }
	`,
	`package p
	import . "go/parser"
	func g() Mode { return ImportsOnly }`,
}

var pkgnames = []string{
	"fmt",
	"math",
}

func TestResolveQualifiedIdents(t *testing.T) {
	// parse package files
	fset := token.NewFileSet()
	var files []*ast.File
	for _, src := range sources {
		f, err := parser.ParseFile(fset, "", src, parser.DeclarationErrors)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
	}

	// resolve and type-check package AST
	idents := make(map[*ast.Ident]Object)
	ctxt := Default
	ctxt.Ident = func(id *ast.Ident, obj Object) { idents[id] = obj }
	pkg, err := ctxt.Check(fset, files)
	if err != nil {
		t.Fatal(err)
	}

	// check that all packages were imported
	for _, name := range pkgnames {
		if pkg.Imports[name] == nil {
			t.Errorf("package %s not imported", name)
		}
	}

	// check that there are no top-level unresolved identifiers
	for _, f := range files {
		for _, x := range f.Unresolved {
			t.Errorf("%s: unresolved global identifier %s", fset.Position(x.Pos()), x.Name)
		}
	}

	// check that qualified identifiers are resolved
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			if s, ok := n.(*ast.SelectorExpr); ok {
				if x, ok := s.X.(*ast.Ident); ok {
					obj := idents[x]
					if obj == nil {
						t.Errorf("%s: unresolved qualified identifier %s", fset.Position(x.Pos()), x.Name)
						return false
					}
					if _, ok := obj.(*Package); ok && idents[s.Sel] == nil {
						t.Errorf("%s: unresolved selector %s", fset.Position(s.Sel.Pos()), s.Sel.Name)
						return false
					}
					return false
				}
				return false
			}
			return true
		})
	}
}

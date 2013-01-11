// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
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
	// TODO(gri) fix this
	// cannot handle dot-import at the moment
	/*
		`package p
		import . "go/parser"
		func g() Mode { return ImportsOnly }`,
	*/
}

var pkgnames = []string{
	"fmt",
	"math",
}

// ResolveQualifiedIdents resolves the selectors of qualified
// identifiers by associating the correct ast.Object with them.
// TODO(gri): Eventually, this functionality should be subsumed
//            by Check.
//
func ResolveQualifiedIdents(fset *token.FileSet, pkg *ast.Package) error {
	var errors scanner.ErrorList

	findObj := func(pkg *ast.Object, name *ast.Ident) *ast.Object {
		scope := pkg.Data.(*ast.Scope)
		obj := scope.Lookup(name.Name)
		if obj == nil {
			errors.Add(fset.Position(name.Pos()), fmt.Sprintf("no %s in package %s", name.Name, pkg.Name))
		}
		return obj
	}

	ast.Inspect(pkg, func(n ast.Node) bool {
		if s, ok := n.(*ast.SelectorExpr); ok {
			if x, ok := s.X.(*ast.Ident); ok && x.Obj != nil && x.Obj.Kind == ast.Pkg {
				// find selector in respective package
				s.Sel.Obj = findObj(x.Obj, s.Sel)
			}
			return false
		}
		return true
	})

	return errors.Err()
}

func TestResolveQualifiedIdents(t *testing.T) {
	// parse package files
	fset := token.NewFileSet()
	files := make([]*ast.File, len(sources))
	for i, src := range sources {
		f, err := parser.ParseFile(fset, "", src, parser.DeclarationErrors)
		if err != nil {
			t.Fatal(err)
		}
		files[i] = f
	}

	// resolve package AST
	astpkg, pkg, err := Check(fset, files)
	if err != nil {
		t.Fatal(err)
	}

	// check that all packages were imported
	for _, name := range pkgnames {
		if pkg.Imports[name] == nil {
			t.Errorf("package %s not imported", name)
		}
	}

	// TODO(gri) fix this
	// unresolved identifiers are not collected at the moment
	// check that there are no top-level unresolved identifiers
	for _, f := range astpkg.Files {
		for _, x := range f.Unresolved {
			t.Errorf("%s: unresolved global identifier %s", fset.Position(x.Pos()), x.Name)
		}
	}

	// resolve qualified identifiers
	if err := ResolveQualifiedIdents(fset, astpkg); err != nil {
		t.Error(err)
	}

	// check that qualified identifiers are resolved
	ast.Inspect(astpkg, func(n ast.Node) bool {
		if s, ok := n.(*ast.SelectorExpr); ok {
			if x, ok := s.X.(*ast.Ident); ok {
				if x.Obj == nil {
					t.Errorf("%s: unresolved qualified identifier %s", fset.Position(x.Pos()), x.Name)
					return false
				}
				if x.Obj.Kind == ast.Pkg && s.Sel != nil && s.Sel.Obj == nil {
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

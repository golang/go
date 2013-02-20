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
	`
	package p
	import "fmt"
	import "math"
	const pi = math.Pi
	func sin(x float64) float64 {
		return math.Sin(x)
	}
	var Println = fmt.Println
	`,
	`
	package p
	import "fmt"
	func f() string {
		_ = "foo"
		return fmt.Sprintf("%d", g())
	}
	func g() (x int) { return }
	`,
	`
	package p
	import . "go/parser"
	import "sync"
	func g() Mode { return ImportsOnly }
	var _, x int = 1, 2
	func init() {}
	type T struct{ sync.Mutex; a, b, c int}
	type I interface{ m() }
	var _ = T{a: 1, b: 2, c: 3}
	func (_ T) m() {}
	`,
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
	var ctxt Context
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

	// Currently, the Check API doesn't call Ident for fields, methods, and composite literal keys.
	// Introduce them artifically so that we can run the check below.
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.StructType:
				for _, list := range x.Fields.List {
					for _, f := range list.Names {
						assert(idents[f] == nil)
						idents[f] = &Var{Pkg: pkg, Name: f.Name}
					}
				}
			case *ast.InterfaceType:
				for _, list := range x.Methods.List {
					for _, f := range list.Names {
						assert(idents[f] == nil)
						idents[f] = &Func{Pkg: pkg, Name: f.Name}
					}
				}
			case *ast.CompositeLit:
				for _, e := range x.Elts {
					if kv, ok := e.(*ast.KeyValueExpr); ok {
						if k, ok := kv.Key.(*ast.Ident); ok {
							assert(idents[k] == nil)
							idents[k] = &Var{Pkg: pkg, Name: k.Name}
						}
					}
				}
			}
			return true
		})
	}

	// check that each identifier in the source is enumerated by the Context.Ident callback
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			if x, ok := n.(*ast.Ident); ok && x.Name != "_" && x.Name != "." {
				obj := idents[x]
				if obj == nil {
					t.Errorf("%s: unresolved identifier %s", fset.Position(x.Pos()), x.Name)
				} else {
					delete(idents, x)
				}
				return false
			}
			return true
		})
	}

	// TODO(gri) enable code below
	// At the moment, the type checker introduces artifical identifiers which are not
	// present in the source. Once it doesn't do that anymore, enable the checks below.
	/*
		for x := range idents {
			t.Errorf("%s: identifier %s not present in source", fset.Position(x.Pos()), x.Name)
		}
	*/
}

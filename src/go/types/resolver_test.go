// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"sort"
	"testing"

	. "go/types"
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
	type errorStringer struct { fmt.Stringer; error }
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
	func h() Mode { return ImportsOnly }
	var _, x int = 1, 2
	func init() {}
	type T struct{ *sync.Mutex; a, b, c int}
	type I interface{ m() }
	var _ = T{a: 1, b: 2, c: 3}
	func (_ T) m() {}
	func (T) _() {}
	var i I
	var _ = i.m
	func _(s []int) { for i, x := range s { _, _ = i, x } }
	func _(x interface{}) {
		switch x := x.(type) {
		case int:
			_ = x
		}
		switch {} // implicit 'true' tag
	}
	`,
	`
	package p
	type S struct{}
	func (T) _() {}
	func (T) _() {}
	`,
	`
	package p
	func _() {
	L0:
	L1:
		goto L0
		for {
			goto L1
		}
		if true {
			goto L2
		}
	L2:
	}
	`,
}

var pkgnames = []string{
	"fmt",
	"math",
}

type resolveTestImporter struct {
	importer Importer
	imported map[string]bool
}

func (imp *resolveTestImporter) Import(path string) (*Package, error) {
	if imp.importer == nil {
		imp.importer = importer.Default()
		imp.imported = make(map[string]bool)
	}
	pkg, err := imp.importer.Import(path)
	if err != nil {
		return nil, err
	}
	imp.imported[path] = true
	return pkg, nil
}

func TestResolveIdents(t *testing.T) {
	skipSpecialPlatforms(t)

	// parse package files
	fset := token.NewFileSet()
	var files []*ast.File
	for i, src := range sources {
		f, err := parser.ParseFile(fset, fmt.Sprintf("sources[%d]", i), src, parser.DeclarationErrors)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
	}

	// resolve and type-check package AST
	importer := new(resolveTestImporter)
	conf := Config{Importer: importer}
	uses := make(map[*ast.Ident]Object)
	defs := make(map[*ast.Ident]Object)
	_, err := conf.Check("testResolveIdents", fset, files, &Info{Defs: defs, Uses: uses})
	if err != nil {
		t.Fatal(err)
	}

	// check that all packages were imported
	for _, name := range pkgnames {
		if !importer.imported[name] {
			t.Errorf("package %s not imported", name)
		}
	}

	// check that qualified identifiers are resolved
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			if s, ok := n.(*ast.SelectorExpr); ok {
				if x, ok := s.X.(*ast.Ident); ok {
					obj := uses[x]
					if obj == nil {
						t.Errorf("%s: unresolved qualified identifier %s", fset.Position(x.Pos()), x.Name)
						return false
					}
					if _, ok := obj.(*PkgName); ok && uses[s.Sel] == nil {
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

	for id, obj := range uses {
		if obj == nil {
			t.Errorf("%s: Uses[%s] == nil", fset.Position(id.Pos()), id.Name)
		}
	}

	// check that each identifier in the source is found in uses or defs or both
	var both []string
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			if x, ok := n.(*ast.Ident); ok {
				var objects int
				if _, found := uses[x]; found {
					objects |= 1
					delete(uses, x)
				}
				if _, found := defs[x]; found {
					objects |= 2
					delete(defs, x)
				}
				if objects == 0 {
					t.Errorf("%s: unresolved identifier %s", fset.Position(x.Pos()), x.Name)
				} else if objects == 3 {
					both = append(both, x.Name)
				}
				return false
			}
			return true
		})
	}

	// check the expected set of idents that are simultaneously uses and defs
	sort.Strings(both)
	if got, want := fmt.Sprint(both), "[Mutex Stringer error]"; got != want {
		t.Errorf("simultaneous uses/defs = %s, want %s", got, want)
	}

	// any left-over identifiers didn't exist in the source
	for x := range uses {
		t.Errorf("%s: identifier %s not present in source", fset.Position(x.Pos()), x.Name)
	}
	for x := range defs {
		t.Errorf("%s: identifier %s not present in source", fset.Position(x.Pos()), x.Name)
	}

	// TODO(gri) add tests to check ImplicitObj callbacks
}

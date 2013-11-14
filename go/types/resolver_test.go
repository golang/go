// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"

	_ "code.google.com/p/go.tools/go/gcimporter"
	. "code.google.com/p/go.tools/go/types"
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
	func h() Mode { return ImportsOnly }
	var _, x int = 1, 2
	func init() {}
	type T struct{ sync.Mutex; a, b, c int}
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

func TestResolveIdents(t *testing.T) {
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
	var conf Config
	idents := make(map[*ast.Ident]Object)
	_, err := conf.Check("testResolveIdents", fset, files, &Info{Objects: idents})
	if err != nil {
		t.Fatal(err)
	}

	// check that all packages were imported
	for _, name := range pkgnames {
		if conf.Packages[name] == nil {
			t.Errorf("package %s not imported", name)
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
					if _, ok := obj.(*PkgName); ok && idents[s.Sel] == nil {
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

	// check that each identifier in the source is found in the idents map
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			if x, ok := n.(*ast.Ident); ok {
				if _, found := idents[x]; found {
					delete(idents, x)
				} else {
					t.Errorf("%s: unresolved identifier %s", fset.Position(x.Pos()), x.Name)
				}
				return false
			}
			return true
		})
	}

	// any left-over identifiers didn't exist in the source
	for x := range idents {
		t.Errorf("%s: identifier %s not present in source", fset.Position(x.Pos()), x.Name)
	}

	// TODO(gri) add tests to check ImplicitObj callbacks
}

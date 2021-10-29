// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeutil_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"testing"

	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/typeparams"
)

func TestStaticCallee(t *testing.T) {
	testStaticCallee(t, []string{
		`package q;
		func Abs(x int) int {
			if x < 0 {
				return -x
			}
			return x
		}`,
		`package p
		import "q"

		type T int

		func g(int)

		var f = g

		var x int

		type s struct{ f func(int) }
		func (s) g(int)

		type I interface{ f(int) }

		var a struct{b struct{c s}}

		var n map[int]func()
		var m []func()

		func calls() {
			g(x)           // a declared func
			s{}.g(x)       // a concrete method
			a.b.c.g(x)     // same
			_ = q.Abs(x)   // declared func, qualified identifier
		}

		func noncalls() {
			_ = T(x)    // a type
			f(x)        // a var
			panic(x)    // a built-in
			s{}.f(x)    // a field
			I(nil).f(x) // interface method
			m[0]()      // a map
			n[0]()      // a slice
		}
		`})
}

func TestTypeParamStaticCallee(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("type parameters are not enabled")
	}
	testStaticCallee(t, []string{
		`package q
		func R[T any]() {}
		`,
		`package p
		import "q"
		type I interface{
			i()
		}

		type G[T any] func() T
		func F[T any]() T { var x T; return x }

		type M[T I] struct{ t T }
		func (m M[T]) noncalls() {
			m.t.i()   // method on a type parameter
		}

		func (m M[T]) calls() {
			m.calls() // method on a generic type
		}

		type Chain[T I] struct{ r struct { s M[T] } }

		type S int
		func (S) i() {}

		func Multi[TP0, TP1 any](){}

		func calls() {
			_ = F[int]()            // instantiated function
			_ = (F[int])()          // go through parens
			M[S]{}.calls()          // instantiated method
			Chain[S]{}.r.s.calls()  // same as above
			Multi[int,string]()     // multiple type parameters
			q.R[int]()              // different package
		}

		func noncalls() {
			_ = G[int](nil)()  // instantiated function
		}
		`})
}

// testStaticCallee parses and type checks each file content in contents
// as a single file package in order. Within functions that have the suffix
// "calls" it checks that the CallExprs within have a static callee.
// If the function's name == "calls" all calls must have static callees,
// and if the name != "calls", the calls must not have static callees.
// Failures are reported on t.
func testStaticCallee(t *testing.T, contents []string) {
	fset := token.NewFileSet()
	packages := make(map[string]*types.Package)
	cfg := &types.Config{Importer: closure(packages)}
	info := &types.Info{
		Uses:       make(map[*ast.Ident]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	typeparams.InitInstanceInfo(info)

	var files []*ast.File
	for i, content := range contents {
		// parse
		f, err := parser.ParseFile(fset, fmt.Sprintf("%d.go", i), content, 0)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)

		// type-check
		pkg, err := cfg.Check(f.Name.Name, fset, []*ast.File{f}, info)
		if err != nil {
			t.Fatal(err)
		}
		packages[pkg.Path()] = pkg
	}

	// check
	for _, f := range files {
		for _, decl := range f.Decls {
			if decl, ok := decl.(*ast.FuncDecl); ok && strings.HasSuffix(decl.Name.Name, "calls") {
				wantCallee := decl.Name.Name == "calls" // false within func noncalls()
				ast.Inspect(decl.Body, func(n ast.Node) bool {
					if call, ok := n.(*ast.CallExpr); ok {
						fn := typeutil.StaticCallee(info, call)
						if fn == nil && wantCallee {
							t.Errorf("%s: StaticCallee returned nil",
								fset.Position(call.Lparen))
						} else if fn != nil && !wantCallee {
							t.Errorf("%s: StaticCallee returned %s, want nil",
								fset.Position(call.Lparen), fn)
						}
					}
					return true
				})
			}
		}
	}
}

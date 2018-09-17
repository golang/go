// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeutil_test

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"testing"

	"golang.org/x/tools/go/types/typeutil"
)

func TestStaticCallee(t *testing.T) {
	const src = `package p

import "fmt"

type T int

func g(int)

var f = g

var x int

type s struct{ f func(int) }
func (s) g(int)

type I interface{ f(int) }

var a struct{b struct{c s}}

func calls() {
	g(x)           // a declared func
	s{}.g(x)       // a concrete method
	a.b.c.g(x)     // same
	fmt.Println(x) // declared func, qualified identifier
}

func noncalls() {
	_ = T(x)    // a type
	f(x)        // a var
	panic(x)    // a built-in
	s{}.f(x)    // a field
	I(nil).f(x) // interface method
}
`
	// parse
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	// type-check
	info := &types.Info{
		Uses:       make(map[*ast.Ident]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	cfg := &types.Config{Importer: importer.For("source", nil)}
	if _, err := cfg.Check("p", fset, []*ast.File{f}, info); err != nil {
		t.Fatal(err)
	}

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

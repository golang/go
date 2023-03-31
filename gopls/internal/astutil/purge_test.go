// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"reflect"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/astutil"
	"golang.org/x/tools/internal/testenv"
)

// TestPurgeFuncBodies tests PurgeFuncBodies by comparing it against a
// (less efficient) reference implementation that purges after parsing.
func TestPurgeFuncBodies(t *testing.T) {
	testenv.NeedsGoBuild(t) // we need the source code for std

	// Load a few standard packages.
	config := packages.Config{Mode: packages.NeedCompiledGoFiles}
	pkgs, err := packages.Load(&config, "encoding/...")
	if err != nil {
		t.Fatal(err)
	}

	// preorder returns the nodes of tree f in preorder.
	preorder := func(f *ast.File) (nodes []ast.Node) {
		ast.Inspect(f, func(n ast.Node) bool {
			if n != nil {
				nodes = append(nodes, n)
			}
			return true
		})
		return nodes
	}

	packages.Visit(pkgs, nil, func(p *packages.Package) {
		for _, filename := range p.CompiledGoFiles {
			content, err := os.ReadFile(filename)
			if err != nil {
				t.Fatal(err)
			}

			fset := token.NewFileSet()

			// Parse then purge (reference implementation).
			f1, _ := parser.ParseFile(fset, filename, content, 0)
			ast.Inspect(f1, func(n ast.Node) bool {
				switch n := n.(type) {
				case *ast.FuncDecl:
					if n.Body != nil {
						n.Body.List = nil
					}
				case *ast.FuncLit:
					n.Body.List = nil
				case *ast.CompositeLit:
					n.Elts = nil
				}
				return true
			})

			// Purge before parse (logic under test).
			f2, _ := parser.ParseFile(fset, filename, astutil.PurgeFuncBodies(content), 0)

			// Compare sequence of node types.
			nodes1 := preorder(f1)
			nodes2 := preorder(f2)
			if len(nodes2) < len(nodes1) {
				t.Errorf("purged file has fewer nodes: %d vs  %d",
					len(nodes2), len(nodes1))
				nodes1 = nodes1[:len(nodes2)] // truncate
			}
			for i := range nodes1 {
				x, y := nodes1[i], nodes2[i]
				if reflect.TypeOf(x) != reflect.TypeOf(y) {
					t.Errorf("%s: got %T, want %T",
						fset.Position(x.Pos()), y, x)
					break
				}
			}
		}
	})
}

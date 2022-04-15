// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/internal/typeparams"
)

// Tests that MethodValue returns the expected method.
func TestMethodValue(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestMethodValue requires type parameters")
	}
	input := `
package p

type I interface{ M() }

type S int
func (S) M() {}
type R[T any] struct{ S }

var i I
var s S
var r R[string]

func selections[T any]() {
	_ = i.M
	_ = s.M
	_ = r.M

	var v R[T]
	_ = v.M
}
`

	// Parse the file.
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", input, 0)
	if err != nil {
		t.Error(err)
		return
	}

	// Build an SSA program from the parsed file.
	p, info, err := ssautil.BuildPackage(&types.Config{}, fset,
		types.NewPackage("p", ""), []*ast.File{f}, ssa.SanityCheckFunctions)
	if err != nil {
		t.Error(err)
		return
	}

	// Collect all of the *types.Selection in the function "selections".
	var selections []*types.Selection
	for _, decl := range f.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "selections" {
			for _, stmt := range fn.Body.List {
				if assign, ok := stmt.(*ast.AssignStmt); ok {
					sel := assign.Rhs[0].(*ast.SelectorExpr)
					selections = append(selections, info.Selections[sel])
				}
			}
		}
	}

	wants := map[string]string{
		"method (p.S) M()":         "(p.S).M",
		"method (p.R[string]) M()": "(p.R[string]).M",
		"method (p.I) M()":         "nil", // interface
		"method (p.R[T]) M()":      "nil", // parameterized
	}
	if len(wants) != len(selections) {
		t.Fatalf("Wanted %d selections. got %d", len(wants), len(selections))
	}
	for _, selection := range selections {
		var got string
		if m := p.Prog.MethodValue(selection); m != nil {
			got = m.String()
		} else {
			got = "nil"
		}
		if want := wants[selection.String()]; want != got {
			t.Errorf("p.Prog.MethodValue(%s) expected %q. got %q", selection, want, got)
		}
	}
}

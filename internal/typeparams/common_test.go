// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/internal/testenv"
	. "golang.org/x/tools/internal/typeparams"
)

func TestGetIndexExprData(t *testing.T) {
	x := &ast.Ident{}
	i := &ast.Ident{}

	want := &IndexListExpr{X: x, Lbrack: 1, Indices: []ast.Expr{i}, Rbrack: 2}
	tests := map[ast.Node]bool{
		&ast.IndexExpr{X: x, Lbrack: 1, Index: i, Rbrack: 2}: true,
		want:         true,
		&ast.Ident{}: false,
	}

	for n, isIndexExpr := range tests {
		X, lbrack, indices, rbrack := UnpackIndexExpr(n)
		if got := X != nil; got != isIndexExpr {
			t.Errorf("UnpackIndexExpr(%v) = %v, _, _, _; want nil: %t", n, x, !isIndexExpr)
		}
		if X == nil {
			continue
		}
		if X != x || lbrack != 1 || indices[0] != i || rbrack != 2 {
			t.Errorf("UnpackIndexExprData(%v) = %v, %v, %v, %v; want %+v", n, x, lbrack, indices, rbrack, want)
		}
	}
}

func TestOriginMethodRecursive(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)
	src := `package p

type N[A any] int

func (r N[B]) m() { r.m(); r.n() }

func (r *N[C]) n() { }
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	info := types.Info{
		Defs: make(map[*ast.Ident]types.Object),
		Uses: make(map[*ast.Ident]types.Object),
	}
	var conf types.Config
	if _, err := conf.Check("p", fset, []*ast.File{f}, &info); err != nil {
		t.Fatal(err)
	}

	// Collect objects from types.Info.
	var m, n *types.Func   // the 'origin' methods in Info.Defs
	var mm, mn *types.Func // the methods used in the body of m

	for _, decl := range f.Decls {
		fdecl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		def := info.Defs[fdecl.Name].(*types.Func)
		switch fdecl.Name.Name {
		case "m":
			m = def
			ast.Inspect(fdecl.Body, func(n ast.Node) bool {
				if call, ok := n.(*ast.CallExpr); ok {
					sel := call.Fun.(*ast.SelectorExpr)
					use := info.Uses[sel.Sel].(*types.Func)
					switch sel.Sel.Name {
					case "m":
						mm = use
					case "n":
						mn = use
					}
				}
				return true
			})
		case "n":
			n = def
		}
	}

	tests := []struct {
		name        string
		input, want *types.Func
	}{
		{"declared m", m, m},
		{"declared n", n, n},
		{"used m", mm, m},
		{"used n", mn, n},
	}

	for _, test := range tests {
		if got := OriginMethod(test.input); got != test.want {
			t.Errorf("OriginMethod(%q) = %v, want %v", test.name, test.input, test.want)
		}
	}
}

func TestOriginMethodUses(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)

	tests := []string{
		`type T interface { m() }; func _(t T) { t.m() }`,
		`type T[P any] interface { m() P }; func _[A any](t T[A]) { t.m() }`,
		`type T[P any] interface { m() P }; func _(t T[int]) { t.m() }`,
		`type T[P any] int; func (r T[A]) m() { r.m() }`,
		`type T[P any] int; func (r *T[A]) m() { r.m() }`,
		`type T[P any] int; func (r *T[A]) m() {}; func _(t T[int]) { t.m() }`,
		`type T[P any] int; func (r *T[A]) m() {}; func _[A any](t T[A]) { t.m() }`,
	}

	for _, src := range tests {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "p.go", "package p; "+src, 0)
		if err != nil {
			t.Fatal(err)
		}
		info := types.Info{
			Uses: make(map[*ast.Ident]types.Object),
		}
		var conf types.Config
		pkg, err := conf.Check("p", fset, []*ast.File{f}, &info)
		if err != nil {
			t.Fatal(err)
		}

		// Look up func T.m.
		T := pkg.Scope().Lookup("T").Type()
		obj, _, _ := types.LookupFieldOrMethod(T, true, pkg, "m")
		m := obj.(*types.Func)

		// Assert that the origin of each t.m() call is p.T.m.
		ast.Inspect(f, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				sel := call.Fun.(*ast.SelectorExpr)
				use := info.Uses[sel.Sel].(*types.Func)
				orig := OriginMethod(use)
				if orig != m {
					t.Errorf("%s:\nUses[%v] = %v, want %v", src, types.ExprString(sel), use, m)
				}
			}
			return true
		})
	}
}

// Issue #60628 was a crash in gopls caused by inconsistency (#60634) between
// LookupFieldOrMethod and NewFileSet for methods with an illegal
// *T receiver type, where T itself is a pointer.
// This is a regression test for the workaround in OriginMethod.
func TestOriginMethod60628(t *testing.T) {
	const src = `package p; type T[P any] *int; func (r *T[A]) f() {}`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	// Expect type error: "invalid receiver type T[A] (pointer or interface type)".
	info := types.Info{
		Uses: make(map[*ast.Ident]types.Object),
	}
	var conf types.Config
	pkg, _ := conf.Check("p", fset, []*ast.File{f}, &info) // error expected
	if pkg == nil {
		t.Fatal("no package")
	}

	// Look up methodset of *T.
	T := pkg.Scope().Lookup("T").Type()
	mset := types.NewMethodSet(types.NewPointer(T))
	if mset.Len() == 0 {
		t.Errorf("NewMethodSet(*T) is empty")
	}
	for i := 0; i < mset.Len(); i++ {
		sel := mset.At(i)
		m := sel.Obj().(*types.Func)

		// TODO(adonovan): check the consistency property required to fix #60634.
		if false {
			m2, _, _ := types.LookupFieldOrMethod(T, true, m.Pkg(), m.Name())
			if m2 != m {
				t.Errorf("LookupFieldOrMethod(%v, indirect=true, %v) = %v, want %v",
					T, m, m2, m)
			}
		}

		// Check the workaround.
		if OriginMethod(m) == nil {
			t.Errorf("OriginMethod(%v) = nil", m)
		}
	}
}

func TestGenericAssignableTo(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)

	tests := []struct {
		src  string
		want bool
	}{
		// The inciting issue: golang/go#50887.
		{`
			type T[P any] interface {
			        Accept(P)
			}

			type V[Q any] struct {
			        Element Q
			}

			func (c V[Q]) Accept(q Q) { c.Element = q }
			`, true},

		// Various permutations on constraints and signatures.
		{`type T[P ~int] interface{ A(P) }; type V[Q int] int; func (V[Q]) A(Q) {}`, true},
		{`type T[P int] interface{ A(P) }; type V[Q ~int] int; func (V[Q]) A(Q) {}`, false},
		{`type T[P int|string] interface{ A(P) }; type V[Q int] int; func (V[Q]) A(Q) {}`, true},
		{`type T[P any] interface{ A(P) }; type V[Q any] int; func (V[Q]) A(Q, Q) {}`, false},
		{`type T[P any] interface{ int; A(P) }; type V[Q any] int; func (V[Q]) A(Q) {}`, false},

		// Various structural restrictions on T.
		{`type T[P any] interface{ ~int; A(P) }; type V[Q any] int; func (V[Q]) A(Q) {}`, true},
		{`type T[P any] interface{ ~int|string; A(P) }; type V[Q any] int; func (V[Q]) A(Q) {}`, true},
		{`type T[P any] interface{ int; A(P) }; type V[Q int] int; func (V[Q]) A(Q) {}`, false},

		// Various recursive constraints.
		{`type T[P ~struct{ f *P }] interface{ A(P) }; type V[Q ~struct{ f *Q }] int; func (V[Q]) A(Q) {}`, true},
		{`type T[P ~struct{ f *P }] interface{ A(P) }; type V[Q ~struct{ g *Q }] int; func (V[Q]) A(Q) {}`, false},
		{`type T[P ~*X, X any] interface{ A(P) X }; type V[Q ~*Y, Y any] int; func (V[Q, Y]) A(Q) (y Y) { return }`, true},
		{`type T[P ~*X, X any] interface{ A(P) X }; type V[Q ~**Y, Y any] int; func (V[Q, Y]) A(Q) (y Y) { return }`, false},
		{`type T[P, X any] interface{ A(P) X }; type V[Q ~*Y, Y any] int; func (V[Q, Y]) A(Q) (y Y) { return }`, true},
		{`type T[P ~*X, X any] interface{ A(P) X }; type V[Q, Y any] int; func (V[Q, Y]) A(Q) (y Y) { return }`, false},
		{`type T[P, X any] interface{ A(P) X }; type V[Q, Y any] int; func (V[Q, Y]) A(Q) (y Y) { return }`, true},

		// In this test case, we reverse the type parameters in the signature of V.A
		{`type T[P, X any] interface{ A(P) X }; type V[Q, Y any] int; func (V[Q, Y]) A(Y) (y Q) { return }`, false},
		// It would be nice to return true here: V can only be instantiated with
		// [int, int], so the identity of the type parameters should not matter.
		{`type T[P, X any] interface{ A(P) X }; type V[Q, Y int] int; func (V[Q, Y]) A(Y) (y Q) { return }`, false},
	}

	for _, test := range tests {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "p.go", "package p; "+test.src, 0)
		if err != nil {
			t.Fatalf("%s:\n%v", test.src, err)
		}
		var conf types.Config
		pkg, err := conf.Check("p", fset, []*ast.File{f}, nil)
		if err != nil {
			t.Fatalf("%s:\n%v", test.src, err)
		}

		V := pkg.Scope().Lookup("V").Type()
		T := pkg.Scope().Lookup("T").Type()

		if types.AssignableTo(V, T) {
			t.Fatal("AssignableTo")
		}

		if got := GenericAssignableTo(nil, V, T); got != test.want {
			t.Fatalf("%s:\nGenericAssignableTo(%v, %v) = %v, want %v", test.src, V, T, got, test.want)
		}
	}
}

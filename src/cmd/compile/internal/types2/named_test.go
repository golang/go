// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"testing"

	"cmd/compile/internal/syntax"
	. "cmd/compile/internal/types2"
)

func BenchmarkNamed(b *testing.B) {
	const src = `
package p

type T struct {
	P int
}

func (T) M(int) {}
func (T) N() (i int) { return }

type G[P any] struct {
	F P
}

func (G[P]) M(P) {}
func (G[P]) N() (p P) { return }

type Inst = G[int]
	`
	pkg := mustTypecheck(src, nil, nil)

	var (
		T        = pkg.Scope().Lookup("T").Type()
		G        = pkg.Scope().Lookup("G").Type()
		SrcInst  = pkg.Scope().Lookup("Inst").Type()
		UserInst = mustInstantiate(b, G, Typ[Int])
	)

	tests := []struct {
		name string
		typ  Type
	}{
		{"nongeneric", T},
		{"generic", G},
		{"src instance", SrcInst},
		{"user instance", UserInst},
	}

	b.Run("Underlying", func(b *testing.B) {
		for _, test := range tests {
			b.Run(test.name, func(b *testing.B) {
				// Access underlying once, to trigger any lazy calculation.
				_ = test.typ.Underlying()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = test.typ.Underlying()
				}
			})
		}
	})
}

func mustInstantiate(tb testing.TB, orig Type, targs ...Type) Type {
	inst, err := Instantiate(nil, orig, targs, true)
	if err != nil {
		tb.Fatal(err)
	}
	return inst
}

// Test that types do not expand infinitely, as in go.dev/issue/52715.
func TestFiniteTypeExpansion(t *testing.T) {
	const src = `
package p

type Tree[T any] struct {
	*Node[T]
}

func (*Tree[R]) N(r R) R { return r }

type Node[T any] struct {
	*Tree[T]
}

func (Node[Q]) M(Q) {}

type Inst = *Tree[int]
`

	f := mustParse(src)
	pkg := NewPackage("p", f.PkgName.Value)
	if err := NewChecker(nil, pkg, nil).Files([]*syntax.File{f}); err != nil {
		t.Fatal(err)
	}

	firstFieldType := func(n *Named) *Named {
		return n.Underlying().(*Struct).Field(0).Type().(*Pointer).Elem().(*Named)
	}

	Inst := pkg.Scope().Lookup("Inst").Type().(*Pointer).Elem().(*Named)
	Node := firstFieldType(Inst)
	Tree := firstFieldType(Node)
	if !Identical(Inst, Tree) {
		t.Fatalf("Not a cycle: got %v, want %v", Tree, Inst)
	}
	if Inst != Tree {
		t.Errorf("Duplicate instances in cycle: %s (%p) -> %s (%p) -> %s (%p)", Inst, Inst, Node, Node, Tree, Tree)
	}
}

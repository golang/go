// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"testing"

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
	pkg, err := pkgFor("p", src, nil)
	if err != nil {
		b.Fatal(err)
	}

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

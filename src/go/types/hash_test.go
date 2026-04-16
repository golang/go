// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

// This file defines a test of using [types.Hasher].

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"hash/maphash"
	"testing"
)

func TestHasher(t *testing.T) {
	const src = `
package p

// Basic defined types.
type T1 int
type T2 int

// Identical methods.
func (T1) M(int) {}
func (T2) M(int) {}

// A constraint interface.
type C interface {
	~int | string
}

type I interface {
}

// A generic type.
type G[P C] int

// Generic functions with identical signature.
func Fa1[P C](p P) {}
func Fa2[Q C](q Q) {}

// Fb1 and Fb2 are identical and should be mapped to the same entry, even if we
// map their arguments first.
func Fb1[P any](x *P) {
	var y *P // Map this first.
	_ = y
}
func Fb2[Q any](x *Q) {
}

// G1 and G2 are mutally recursive, and have identical methods.
type G1[P any] struct{
	Field *G2[P]
}
func (G1[P]) M(G1[P], G2[P]) {}
type G2[Q any] struct{
	Field *G1[Q]
}
func (G2[P]) M(G1[P], G2[P]) {}

// Method type expressions on different generic types are different.
var ME1 = G1[int].M
var ME2 = G2[int].M

// ME1Type should have identical type as ME1.
var ME1Type func(G1[int], G1[int], G2[int])

// Examples from issue #51314
type Constraint[T any] any
func Foo[T Constraint[T]]() {}
func Fn[T1 ~*T2, T2 ~*T1](t1 T1, t2 T2) {}

// Bar and Baz are identical to Foo.
func Bar[P Constraint[P]]() {}
func Baz[Q any]() {} // The underlying type of Constraint[P] is any.
// But Quux is not.
func Quux[Q interface{ quux() }]() {}

type Issue56048_I interface{ m() interface { Issue56048_I } }
var Issue56048 = Issue56048_I.m

type Issue56048_Ib interface{ m() chan []*interface { Issue56048_Ib } }
var Issue56048b = Issue56048_Ib.m

// Non-generic alias
type NonAlias int
type Alias1 = NonAlias
type Alias2 = NonAlias

type Tagged1 struct { F int "tag1" }
type Tagged2 struct { F int "tag2" }
`

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "p.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	var conf types.Config
	pkg, err := conf.Check("", fset, []*ast.File{file}, nil)
	if err != nil {
		t.Fatal(err)
	}

	instantiate := func(origin types.Type, targs ...types.Type) types.Type {
		inst, err := types.Instantiate(nil, origin, targs, true)
		if err != nil {
			t.Fatal(err)
		}
		return inst
	}

	scope := pkg.Scope()
	var (
		tInt    = types.Typ[types.Int]
		tString = types.Typ[types.String]

		T1      = scope.Lookup("T1").Type().(*types.Named)
		T2      = scope.Lookup("T2").Type().(*types.Named)
		T1M     = T1.Method(0).Type()
		T2M     = T2.Method(0).Type()
		G       = scope.Lookup("G").Type()
		GInt1   = instantiate(G, tInt)
		GInt2   = instantiate(G, tInt)
		GStr    = instantiate(G, tString)
		C       = scope.Lookup("C").Type()
		CI      = C.Underlying().(*types.Interface)
		I       = scope.Lookup("I").Type()
		II      = I.Underlying().(*types.Interface)
		U       = CI.EmbeddedType(0).(*types.Union)
		Fa1     = scope.Lookup("Fa1").Type().(*types.Signature)
		Fa2     = scope.Lookup("Fa2").Type().(*types.Signature)
		Fa1P    = Fa1.TypeParams().At(0)
		Fa2Q    = Fa2.TypeParams().At(0)
		Fb1     = scope.Lookup("Fb1").Type().(*types.Signature)
		Fb1x    = Fb1.Params().At(0).Type()
		Fb1y    = scope.Lookup("Fb1").(*types.Func).Scope().Lookup("y").Type()
		Fb2     = scope.Lookup("Fb2").Type().(*types.Signature)
		Fb2x    = Fb2.Params().At(0).Type()
		G1      = scope.Lookup("G1").Type().(*types.Named)
		G1M     = G1.Method(0).Type()
		G1IntM1 = instantiate(G1, tInt).(*types.Named).Method(0).Type()
		G1IntM2 = instantiate(G1, tInt).(*types.Named).Method(0).Type()
		G1StrM  = instantiate(G1, tString).(*types.Named).Method(0).Type()
		G2      = scope.Lookup("G2").Type()
		G2IntM  = instantiate(G2, tInt).(*types.Named).Method(0).Type()
		ME1     = scope.Lookup("ME1").Type()
		ME1Type = scope.Lookup("ME1Type").Type()
		ME2     = scope.Lookup("ME2").Type()

		Constraint  = scope.Lookup("Constraint").Type()
		Foo         = scope.Lookup("Foo").Type()
		Fn          = scope.Lookup("Fn").Type()
		Bar         = scope.Lookup("Bar").Type()
		Baz         = scope.Lookup("Baz").Type()
		Quux        = scope.Lookup("Quux").Type()
		Issue56048  = scope.Lookup("Issue56048").Type()
		Issue56048b = scope.Lookup("Issue56048b").Type()

		NonAlias = scope.Lookup("NonAlias").Type()
		Alias1   = scope.Lookup("Alias1").Type()
		Alias2   = scope.Lookup("Alias2").Type()

		Tagged1 = scope.Lookup("Tagged1").Type().Underlying().(*types.Struct)
		Tagged2 = scope.Lookup("Tagged2").Type().Underlying().(*types.Struct)
	)

	// eqclasses groups the above types into types.Identical equivalence classes.
	eqclasses := [][]types.Type{
		{T1},
		{T2},
		{G},
		{C},
		{CI},
		{U},
		{I},
		{II}, // should not be identical to CI
		{T1M, T2M},
		{GInt1, GInt2},
		{GStr},
		{Fa1, Fa2},
		{Fa1P},
		{Fa2Q},
		{Fb1y, Fb1x},
		{Fb2x},
		{Fb1, Fb2},
		{G1},
		{G1M},
		{G2},
		{G1IntM1, G1IntM2, G2IntM},
		{G1StrM},
		{ME1, ME1Type},
		{ME2},
		{Constraint},
		{Foo, Bar},
		{Baz},
		{Fn},
		{Quux},
		{Issue56048},
		{Issue56048b},
		{NonAlias, Alias1, Alias2},
	}

	run := func(t *testing.T, hasher maphash.Hasher[types.Type], eq func(x, y types.Type) bool, classes [][]types.Type) {
		seed := maphash.MakeSeed()

		hash := func(t types.Type) uint64 {
			var h maphash.Hash
			h.SetSeed(seed)
			hasher.Hash(&h, t)
			return h.Sum64()
		}

		for xi, class := range classes {
			tx := class[0] // arbitrary representative of class

			for yi := range classes {
				if xi == yi {
					// Within a class, each element is equivalent to first
					// and has the same hash.
					for i, ty := range class {
						hx, hy := hash(tx), hash(ty)
						if !eq(tx, ty) || hx != hy {
							t.Fatalf("class[%d][%d] (%v, hash %x) is not equivalent to class[%d][%d] (%v, hash %x)",
								xi, 0, tx, hx,
								yi, i, ty, hy)
						}
					}
				} else {
					// Across classes, no element is equivalent to first.
					// (We can't say for sure that the hashes are unequal.)
					for k, typ := range classes[yi] {
						if eq(tx, typ) {
							t.Fatalf("class[%d][%d] (%v) is equivalent to class[%d][%d] (%v)",
								xi, 0, tx,
								yi, k, typ)
						}
					}
				}
			}
		}
	}

	// Hasher considers the two Tagged{1,2} types distinct.
	t.Run("Hasher", func(t *testing.T) {
		run(t, types.Hasher{}, types.Identical, append(
			eqclasses,
			[]types.Type{Tagged1},
			[]types.Type{Tagged2},
		))
	})

	// HasherIgnoreTags considers the two Tagged{1,2} types equal.
	t.Run("HasherIgnoreTags", func(t *testing.T) {
		run(t, types.HasherIgnoreTags{}, types.IdenticalIgnoreTags, append(
			eqclasses,
			[]types.Type{Tagged1, Tagged2},
		))
	})
}

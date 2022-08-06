// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test case stress tests a number of subtle cases involving
// nested type-parameterized declarations. At a high-level, it
// declares a generic function that contains a generic type
// declaration:
//
//	func F[A intish]() {
//		type T[B intish] struct{}
//
//		// store reflect.Type tuple (A, B, F[A].T[B]) in tests
//	}
//
// It then instantiates this function with a variety of type arguments
// for A and B. Particularly tricky things like shadowed types.
//
// From this data it tests two things:
//
// 1. Given tuples (A, B, F[A].T[B]) and (A', B', F[A'].T[B']),
//    F[A].T[B] should be identical to F[A'].T[B'] iff (A, B) is
//    identical to (A', B').
//
// 2. A few of the instantiations are constructed to be identical, and
//    it tests that exactly these pairs are duplicated (by golden
//    output comparison to nested.out).
//
// In both cases, we're effectively using the compiler's existing
// runtime.Type handling (which is well tested) of type identity of A
// and B as a way to help bootstrap testing and validate its new
// runtime.Type handling of F[A].T[B].
//
// This isn't perfect, but it smoked out a handful of issues in
// gotypes2 and unified IR.

package main

import (
	"fmt"
	"reflect"
)

type test struct {
	TArgs    [2]reflect.Type
	Instance reflect.Type
}

var tests []test

type intish interface{ ~int }

type Int int
type GlobalInt = Int // allow access to global Int, even when shadowed

func F[A intish]() {
	add := func(B, T interface{}) {
		tests = append(tests, test{
			TArgs: [2]reflect.Type{
				reflect.TypeOf(A(0)),
				reflect.TypeOf(B),
			},
			Instance: reflect.TypeOf(T),
		})
	}

	type Int int

	type T[B intish] struct{}

	add(int(0), T[int]{})
	add(Int(0), T[Int]{})
	add(GlobalInt(0), T[GlobalInt]{})
	add(A(0), T[A]{}) // NOTE: intentionally dups with int and GlobalInt

	type U[_ any] int
	type V U[int]
	type W V

	add(U[int](0), T[U[int]]{})
	add(U[Int](0), T[U[Int]]{})
	add(U[GlobalInt](0), T[U[GlobalInt]]{})
	add(U[A](0), T[U[A]]{}) // NOTE: intentionally dups with U[int] and U[GlobalInt]
	add(V(0), T[V]{})
	add(W(0), T[W]{})
}

func main() {
	type Int int

	F[int]()
	F[Int]()
	F[GlobalInt]()

	type U[_ any] int
	type V U[int]
	type W V

	F[U[int]]()
	F[U[Int]]()
	F[U[GlobalInt]]()
	F[V]()
	F[W]()

	// TODO(go.dev/issue/54512): Restore these tests. They currently
	// cause problems for shaping with unified IR.
	//
	// For example, instantiating X[int] requires instantiating shape
	// type X[shapify(int)] == X[go.shape.int]. In turn, this requires
	// instantiating U[shapify(X[go.shape.int])]. But we're still in the
	// process of constructing X[go.shape.int], so we don't yet know its
	// underlying type.
	//
	// Notably, this is a consequence of unified IR writing out type
	// declarations with a reference to the full RHS expression (i.e.,
	// U[X[A]]) rather than its underlying type (i.e., int), which is
	// necessary to support //go:notinheap. Once go.dev/issue/46731 is
	// implemented and unified IR is updated, I expect this will just
	// work.
	//
	// type X[A any] U[X[A]]
	//
	// F[X[int]]()
	// F[X[Int]]()
	// F[X[GlobalInt]]()

	for j, tj := range tests {
		for i, ti := range tests[:j+1] {
			if (ti.TArgs == tj.TArgs) != (ti.Instance == tj.Instance) {
				fmt.Printf("FAIL: %d,%d: %s, but %s\n", i, j, eq(ti.TArgs, tj.TArgs), eq(ti.Instance, tj.Instance))
			}

			// The test is constructed so we should see a few identical types.
			// See "NOTE" comments above.
			if i != j && ti.Instance == tj.Instance {
				fmt.Printf("%d,%d: %v\n", i, j, ti.Instance)
			}
		}
	}
}

func eq(a, b interface{}) string {
	op := "=="
	if a != b {
		op = "!="
	}
	return fmt.Sprintf("%v %s %v", a, op, b)
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package test_helpers

import (
	"fmt"
	"math/rand/v2"
	"reflect"
	"slices"
	"testing"
)

// == and != on vectors compare them as if they were structs with one field per
// element, and on masks as if they were structs with one bool per element.

// EqVector is a vector type T that can be compared element-wise with another T
// to produce a mask of type M.
type EqVector[T any, M EqMask] interface {
	comparable
	Equal(T) M
	Len() int
}

// EqMask is a mask type, returned by Vec.Equal(Vec).
type EqMask interface {
	comparable
	All() bool
}

// CheckVectorEqual checks == and != on vectors of type T against comparing
// their elements one at a time, for many pairs of vectors loaded from values.
//
// eq and ne must be x == y and x != y, written for the concrete type T, so that
// they test the compiler's direct lowering of ==. CheckVectorEqual also tests
// == in generic code, on structs and arrays containing vectors, on interfaces,
// through reflect, and on map keys.
func CheckVectorEqual[T EqVector[T, M], M EqMask, E number](t *testing.T, load func([]E) T, eq, ne func(x, y T) bool, values []E) {
	t.Helper()
	values = distinct(values)
	var zero T
	a, b := make([]E, zero.Len()), make([]E, zero.Len())

	check := func(a, b []E) {
		t.Helper()
		x, y := load(a), load(b)
		want := slices.Equal(a, b)
		if got := eq(x, y); got != want {
			t.Fatalf("%v == %v is %v, want %v", a, b, got, want)
		}
		if got := ne(x, y); got == want {
			t.Fatalf("%v != %v is %v, want %v", a, b, got, !want)
		}
		if got := x.Equal(y).All(); got != want {
			t.Fatalf("%v.Equal(%v).All() is %v, want %v", a, b, got, want)
		}
		checkEqualForms(t, x, y, want, func() string { return fmt.Sprintf("%v and %v", a, b) })
	}

	// Starting from several random vectors a, compare a to itself, and to
	// each vector that differs from it in one element, setting that element
	// to each of the values in turn.
	r := rand.New(rand.NewPCG(1, 2))
	for range 10 {
		pickN(r, a, values)
		copy(b, a)
		check(a, b)
		for i := range b {
			for _, v := range values {
				b[i] = v
				check(a, b)
			}
			b[i] = a[i]
		}
	}

	// Check that random vectors work as map keys.
	for range 20 {
		pickN(r, a, values)
		x := load(a)
		m := map[T]int{x: 1}
		m[load(a)]++
		// x == x is false exactly when x has a NaN element. Like a NaN key
		// in a map of floats, such a vector is a new key every time it's
		// stored, and can never be found.
		if x == x {
			if len(m) != 1 || m[x] != 2 {
				t.Fatalf("map keyed by %v twice is %v, want one key with value 2", a, m)
			}
		} else {
			if len(m) != 2 {
				t.Fatalf("map keyed by %v (which has a NaN) twice has %d keys, want 2", a, len(m))
			}
			if _, ok := m[x]; ok {
				t.Fatalf("found key %v, which has a NaN, in map", a)
			}
		}
	}

	if !reflect.TypeFor[T]().Comparable() {
		t.Errorf("reflect reports %T as not comparable", zero)
	}
}

// CheckMaskEqual checks == and != on masks of type M against comparing their
// elements one at a time. It makes each mask by using load to make a vector of
// type T whose elements are zero where the mask should be true, and comparing
// that vector to zero.
//
// eq and ne must be x == y and x != y, written for the concrete type M, so that
// they test the compiler's direct lowering of ==. CheckMaskEqual also tests ==
// in generic code, on structs and arrays containing masks, on interfaces,
// through reflect, and on map keys.
func CheckMaskEqual[T EqVector[T, M], M EqMask, E number](t *testing.T, load func([]E) T, eq, ne func(x, y M) bool) {
	t.Helper()
	var zeroVec T
	n := zeroVec.Len()
	a, b, zeros := make([]bool, n), make([]bool, n), load(make([]E, n))
	toMask := func(s []bool) M {
		v := make([]E, n)
		for i, b := range s {
			if !b {
				v[i] = 1
			}
		}
		return load(v).Equal(zeros)
	}

	check := func(a, b []bool) {
		t.Helper()
		x, y := toMask(a), toMask(b)
		want := slices.Equal(a, b)
		if got := eq(x, y); got != want {
			t.Fatalf("mask %v == %v is %v, want %v", a, b, got, want)
		}
		if got := ne(x, y); got == want {
			t.Fatalf("mask %v != %v is %v, want %v", a, b, got, !want)
		}
		checkEqualForms(t, x, y, want, func() string { return fmt.Sprintf("masks %v and %v", a, b) })
	}

	// Starting from several random masks a, compare a to itself, and to
	// every mask that differs from it in one element.
	r := rand.New(rand.NewPCG(1, 2))
	for range 10 {
		pickN(r, a, []bool{false, true})
		copy(b, a)
		check(a, b)
		for i := range b {
			b[i] = !b[i]
			check(a, b)
			b[i] = a[i]
		}
	}

	// Check that random masks work as map keys.
	for range 20 {
		pickN(r, a, []bool{false, true})
		m := map[M]int{toMask(a): 1}
		m[toMask(a)]++
		if len(m) != 1 || m[toMask(a)] != 2 {
			t.Fatalf("map keyed by mask %v twice is %v, want one key with value 2", a, m)
		}
	}

	if !reflect.TypeFor[M]().Comparable() {
		t.Errorf("reflect reports %T as not comparable", zeroVec.Equal(zeroVec))
	}
}

// checkEqualForms checks that the forms of == other than a direct comparison of
// two values of a concrete vector or mask type report want for x and y. desc
// describes x and y for failure messages.
//
// Since V is a type parameter, x == y here tests == in generic code. Which of
// these comparisons the compiler lowers directly and which call V's generated
// equality function depends on how it stencils generic code and on how it
// compares structs and arrays. Currently, generic code is stenciled for V's
// shape, which is V's SIMD struct type, so x == y and the struct comparison are
// lowered directly, while the array comparison, interfaces and reflect call an
// equality function.
func checkEqualForms[V comparable](t *testing.T, x, y V, want bool, desc func() string) {
	t.Helper()
	if got := x == y; got != want {
		t.Fatalf("generic == on %s is %v, want %v", desc(), got, want)
	}
	if got := any(x) == any(y); got != want {
		t.Fatalf("== on %s as interfaces is %v, want %v", desc(), got, want)
	}
	if got := (struct {
		v V
		n int
	}{x, 1} == struct {
		v V
		n int
	}{y, 1}); got != want {
		t.Fatalf("== on structs containing %s is %v, want %v", desc(), got, want)
	}
	if got := [2]V{x, x} == [2]V{x, y}; got != want {
		t.Fatalf("== on arrays containing %s is %v, want %v", desc(), got, want)
	}
	if got := reflect.ValueOf(x).Equal(reflect.ValueOf(y)); got != want {
		t.Fatalf("reflect.Value.Equal on %s is %v, want %v", desc(), got, want)
	}
	if !reflect.ValueOf(x).Comparable() {
		t.Fatalf("reflect.Value.Comparable on %s is false", desc())
	}
}

// pickN sets each element of out to a random element of options.
func pickN[E any](r *rand.Rand, out, options []E) {
	for i := range out {
		out[i] = options[r.IntN(len(options))]
	}
}

// distinct returns values without repeats, in order. It treats all NaNs as the
// same, since they all compare the same way, but keeps +0 and -0 apart.
func distinct[E number](values []E) []E {
	same := func(x, y E) bool {
		// The same test CheckSlicesLogInput uses with no flakiness.
		if x != y {
			return isSameFloat(x, y, 0)
		}
		return !isDifferentSignZero(x, y)
	}
	var out []E
	for _, v := range values {
		if !slices.ContainsFunc(out, func(w E) bool { return same(v, w) }) {
			out = append(out, v)
		}
	}
	return out
}

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file shows some examples of constraint literals with elided interfaces.
// These examples are permitted if proposal issue #48424 is accepted.

package p

// Constraint type sets of the form T, ~T, or A|B may omit the interface.
type (
	_[T int]            struct{}
	_[T ~int]           struct{}
	_[T int | string]   struct{}
	_[T ~int | ~string] struct{}
)

func min[T int | string](x, y T) T {
	if x < y {
		return x
	}
	return y
}

func lookup[M ~map[K]V, K comparable, V any](m M, k K) V {
	return m[k]
}

func deref[P ~*E, E any](p P) E {
	return *p
}

func _() int {
	p := new(int)
	return deref(p)
}

func addrOfCopy[V any, P *V](v V) P {
	return &v
}

func _() *int {
	return addrOfCopy(0)
}

// A type parameter may not be embedded in an interface;
// so it can also not be used as a constraint.
func _[A any, B A /* ERROR "cannot use a type parameter as constraint" */]()    {}
func _[A any, B, C A /* ERROR "cannot use a type parameter as constraint" */]() {}

// Error messages refer to the type constraint as it appears in the source.
// (No implicit interface should be exposed.)
func _[T string](x T) T {
	return x /* ERROR "constrained by string" */ * x
}

func _[T int | string](x T) T {
	return x /* ERROR "constrained by int | string" */ * x
}

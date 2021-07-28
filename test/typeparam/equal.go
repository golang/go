// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// comparisons of type parameters to interfaces

package main

func f[T comparable](t, u T) bool {
	// Comparing two type parameters directly.
	// (Not really testing comparisons to interfaces, but just 'cause we're here.)
	return t == u
}

func g[T comparable](t T, i interface{}) bool {
	// Compare type parameter value to empty interface.
	return t == i
}

type I interface {
	foo()
}

type C interface {
	comparable
	I
}

func h[T C](t T, i I) bool {
	// Compare type parameter value to nonempty interface.
	return t == i
}

type myint int

func (x myint) foo() {
}

func k[T comparable](t T, i interface{}) bool {
	// Compare derived type value to interface.
	return struct{ a, b T }{t, t} == i
}

func main() {
	assert(f(3, 3))
	assert(!f(3, 5))
	assert(g(3, 3))
	assert(!g(3, 5))
	assert(h(myint(3), myint(3)))
	assert(!h(myint(3), myint(5)))

	type S struct{ a, b float64 }

	assert(f(S{3, 5}, S{3, 5}))
	assert(!f(S{3, 5}, S{4, 6}))
	assert(g(S{3, 5}, S{3, 5}))
	assert(!g(S{3, 5}, S{4, 6}))

	assert(k(3, struct{ a, b int }{3, 3}))
	assert(!k(3, struct{ a, b int }{3, 4}))
}

func assert(b bool) {
	if !b {
		panic("assertion failed")
	}
}

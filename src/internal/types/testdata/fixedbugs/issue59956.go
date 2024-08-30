// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f1(func(int))
func f2(func(int), func(string))
func f3(func(int), func(string), func(float32))

func g1[P any](P) {}

func _() {
	f1(g1)
	f2(g1, g1)
	f3(g1, g1, g1)
}

// More complex examples

func g2[P any](P, P)                                         {}
func h3[P any](func(P), func(P), func() P)                   {}
func h4[P, Q any](func(P), func(P, Q), func() Q, func(P, Q)) {}

func r1() int { return 0 }

func _() {
	h3(g1, g1, r1)
	h4(g1, g2, r1, g2)
}

// Variadic cases

func f(func(int))
func g[P any](P) {}

func d[P any](...func(P)) {}

func _() {
	d /* ERROR "cannot infer P" */ ()
	d(f)
	d(f, g)
	d(f, g, g)
	d /* ERROR "cannot infer P" */ (g, g, g)
	d(g, g, f)
	d(g, f, g, f)
}

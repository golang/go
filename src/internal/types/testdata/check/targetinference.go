// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file checks error messages for cases where function types were inferred from context.
//
// The function doc strings refer to the type checker functions that contain the relevant
// newTarget/Hint calls (there's a chance that they may change, so this needs to be kept
// in mind when looking for the calls).
//
// Note that most descriptor strings provided to newTarget/Hint never make it into an error
// message (because the descriptors are not used). This test ensures that we will notice if
// that changes.

package p

// This is a map with an invalid key, but we can still test type inference against the key.
// (The type checker reports an error but keeps working with that key type.)
type M map[F /* ERROR "invalid map key type" */ ]int
type F func(int, string)

func f[T any](T, T) {}

// Checker.assignment
func _() {
	var m M
	delete(m, f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in argument to delete" */ [int])
}

// Checker.assignVar
func _() {
	var v1 F
	v1 = f /* ERROR "inferred type func(int, int) for func(T, T) does not match type F of v1" */
	_ = v1

	var v2 func(int, string)
	v2 = f /* ERROR "inferred type func(int, int) for func(T, T) does not match type func(int, string) of v2" */
	_ = v2

	type A = func(int, string)
	var v3 A
	v3 = f /* ERROR "inferred type func(int, int) for func(T, T) does not match type A of v3" */
	_ = v3

	var a []F
	var i, j int
	a[i+j] = f /* ERROR "inferred type func(int, int) for func(T, T) does not match type F of a[i + j]" */
}

// Checker.initVars
func _() F {
	return f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in return statement" */ [int]
}

// Checker.callExpr
var _ = F(f /* ERROR "cannot convert f[int] (value of type func(int, int)) to type F" */ [int])

// Checker.callExpr
func g(F) {}
func _() {
	g(f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in argument to g" */ [int])
}

// Checker.callExpr
func h(int, F) {}
func _() {
	h(1, f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in argument to h" */ [int])
}

// Checker.varDecl
var _ F = f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in variable declaration" */ [int]

// Checker.indexExpr
func _() {
	var m M
	_ = m[f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in map key" */ [int]]
}

// Checker.indexExpr
func _() {
	var m M
	m[f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in map key" */ [int]] = 1
}

// Checker.compositeLit
type S struct{ f F }

var _ = S{f: f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in struct literal" */ [int]}

// Checker.compositeLit
var _ = S{f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in struct literal" */ [int]}

// Checker.compositeLit
var _ = M{f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in map literal" */ [int]: 1}

// Checker.compositeLit
var _ = map[int]F{1: f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in map literal" */ [int]}

// Checker.indexElts
var _ = []F{f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in array or slice literal" */ [int]}

// Checker.stmt
func _() {
	var c chan F
	c <- f /* ERROR "cannot use f[int] (value of type func(int, int)) as F value in send" */ [int]
}

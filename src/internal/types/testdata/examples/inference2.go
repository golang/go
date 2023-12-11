// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file shows some examples of "reverse" type inference
// where the type arguments for generic functions are determined
// from assigning the functions.

package p

func f1[P any](P)        {}
func f2[P any]() P       { var x P; return x }
func f3[P, Q any](P) Q   { var x Q; return x }
func f4[P any](P, P)     {}
func f5[P any](P) []P    { return nil }
func f6[P any](int) P    { var x P; return x }
func f7[P any](P) string { return "" }

// initialization expressions
var (
	v1           = f1 // ERROR "cannot use generic function f1 without instantiation"
	v2 func(int) = f2 // ERROR "cannot infer P"

	v3 func(int)     = f1
	v4 func() int    = f2
	v5 func(int) int = f3
	_  func(int) int = f3[int]

	v6 func(int, int)     = f4
	v7 func(int, string)  = f4 // ERROR "type func(int, string) of v7 does not match inferred type func(int, int) for func(P, P)"
	v8 func(int) []int    = f5
	v9 func(string) []int = f5 // ERROR "type func(string) []int of v9 does not match inferred type func(string) []string for func(P) []P"

	_, _ func(int) = f1, f1
	_, _ func(int) = f1, f2 // ERROR "cannot infer P"
)

// Regular assignments
func _() {
	v1 = f1 // no error here because v1 is invalid (we don't know its type) due to the error above
	var v1_ func() int
	_ = v1_
	v1_ = f1 // ERROR "cannot infer P"
	v2 = f2  // ERROR "cannot infer P"

	v3 = f1
	v4 = f2
	v5 = f3
	v5 = f3[int]

	v6 = f4
	v7 = f4 // ERROR "type func(int, string) of v7 does not match inferred type func(int, int) for func(P, P)"
	v8 = f5
	v9 = f5 // ERROR "type func(string) []int of v9 does not match inferred type func(string) []string for func(P) []P"

	// non-trivial LHS
	var a [2]func(string) []int
	a[0] = f5 // ERROR "type func(string) []int of a[0] does not match inferred type func(string) []string for func(P) []P"
}

// Return statements
func _() func(int)     { return f1 }
func _() func() int    { return f2 }
func _() func(int) int { return f3 }
func _() func(int) int { return f3[int] }

func _() func(int, int) { return f4 }
func _() func(int, string) {
	return f4 /* ERROR "type func(int, string) of result variable does not match inferred type func(int, int) for func(P, P)" */
}
func _() func(int) []int { return f5 }
func _() func(string) []int {
	return f5 /* ERROR "type func(string) []int of result variable does not match inferred type func(string) []string for func(P) []P" */
}

func _() (_, _ func(int)) { return f1, f1 }
func _() (_, _ func(int)) { return f1, f2 /* ERROR "cannot infer P" */ }

// Argument passing
func g1(func(int))                           {}
func g2(func(int, int))                      {}
func g3(func(int) string)                    {}
func g4[P any](func(P) string)               {}
func g5[P, Q any](func(P) string, func(P) Q) {}
func g6(func(int), func(string))             {}

func _() {
	g1(f1)
	g1(f2 /* ERROR "cannot infer P" */)
	g2(f4)
	g4(f6)
	g5(f6, f7)
	g6(f1, f1)
}

// Argument passing of partially instantiated functions
func h(func(int, string), func(string, int)) {}

func p[P, Q any](P, Q) {}

func _() {
	h(p, p)
	h(p[int], p[string])
}

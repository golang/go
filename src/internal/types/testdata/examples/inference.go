// -lang=go1.20

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file shows some examples of type inference.

package p

type Ordered interface {
	~int | ~float64 | ~string
}

func min[T Ordered](x, y T) T { panic(0) }

func _() {
	// min can be called with explicit instantiation.
	_ = min[int](1, 2)

	// Alternatively, the type argument can be inferred from
	// one of the arguments. Untyped arguments will be considered
	// last.
	var x int
	_ = min(x, x)
	_ = min(x, 1)
	_ = min(x, 1.0)
	_ = min(1, 2)
	_ = min(1, 2.3)

	var y float64
	_ = min(1, y)
	_ = min(1.2, y)
	_ = min(1.2, 3.4)
	_ = min(1.2, 3)

	var s string
	_ = min(s, "foo")
	_ = min("foo", "bar")
}

func mixed[T1, T2, T3 any](T1, T2, T3) {}

func _() {
	// mixed can be called with explicit instantiation.
	mixed[int, string, bool](0, "", false)

	// Alternatively, partial type arguments may be provided
	// (from left to right), and the other may be inferred.
	mixed[int, string](0, "", false)
	mixed[int](0, "", false)
	mixed(0, "", false)

	// Provided type arguments always take precedence over
	// inferred types.
	mixed[int, string](1.1 /* ERROR "cannot use 1.1" */, "", false)
}

func related1[Slice interface{ ~[]Elem }, Elem any](s Slice, e Elem) {}

func _() {
	// related1 can be called with explicit instantiation.
	var si []int
	related1[[]int, int](si, 0)

	// Alternatively, the 2nd type argument can be inferred
	// from the first one through constraint type inference.
	var ss []string
	_ = related1[[]string]
	related1[[]string](ss, "foo")

	// A type argument inferred from another explicitly provided
	// type argument overrides whatever value argument type is given.
	related1[[]string](ss, 0 /* ERROR "cannot use 0" */)

	// A type argument may be inferred from a value argument
	// and then help infer another type argument via constraint
	// type inference.
	related1(si, 0)
	related1(si, "foo" /* ERROR `cannot use "foo"` */)
}

func related2[Elem any, Slice interface{ []Elem }](e Elem, s Slice) {}

func _() {
	// related2 can be called with explicit instantiation.
	var si []int
	related2[int, []int](0, si)

	// Alternatively, the 2nd type argument can be inferred
	// from the first one through constraint type inference.
	var ss []string
	_ = related2[string]
	related2[string]("foo", ss)

	// A type argument may be inferred from a value argument
	// and then help infer another type argument via constraint
	// type inference. Untyped arguments are always considered
	// last.
	related2(1.2, []float64{})
	related2(1.0, []int{})
	related2 /* ERROR "Slice (type []int) does not satisfy interface{[]Elem}" */ (float64(1.0), []int{}) // TODO(gri) better error message
}

type List[P any] []P

func related3[Elem any, Slice []Elem | List[Elem]]() Slice { return nil }

func _() {
	// related3 can be instantiated explicitly
	related3[int, []int]()
	related3[byte, List[byte]]()

	// The 2nd type argument cannot be inferred from the first
	// one because there's two possible choices: []Elem and
	// List[Elem].
	related3 /* ERROR "cannot infer Slice" */ [int]()
}

func wantsMethods[P interface {
	m1(Q)
	m2() R
}, Q, R any](P) {
}

type hasMethods1 struct{}

func (hasMethods1) m1(int)
func (hasMethods1) m2() string

type hasMethods2 struct{}

func (*hasMethods2) m1(int)
func (*hasMethods2) m2() string

type hasMethods3 interface {
	m1(float64)
	m2() complex128
}

type hasMethods4 interface {
	m1()
}

func _() {
	// wantsMethod can be called with arguments that have the relevant methods
	// and wantsMethod's type arguments are inferred from those types' method
	// signatures.
	wantsMethods(hasMethods1{})
	wantsMethods(&hasMethods1{})
	wantsMethods /* ERROR "P (type hasMethods2) does not satisfy interface{m1(Q); m2() R} (method m1 has pointer receiver)" */ (hasMethods2{})
	wantsMethods(&hasMethods2{})
	wantsMethods(hasMethods3(nil))
	wantsMethods /* ERROR "P (type any) does not satisfy interface{m1(Q); m2() R} (missing method m1)" */ (any(nil))
	wantsMethods /* ERROR "P (type hasMethods4) does not satisfy interface{m1(Q); m2() R} (wrong type for method m1)" */ (hasMethods4(nil))
}

// "Reverse" type inference is not yet permitted.

func f[P any](P) {}

// This must not crash.
var _ func(int) = f // ERROR "implicitly instantiated function in assignment requires go1.21 or later"

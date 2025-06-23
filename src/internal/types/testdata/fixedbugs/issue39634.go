// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Examples from the issue adjusted to match new [T any] syntax for type parameters.
// Also, previously permitted empty type parameter lists and instantiations
// are now syntax errors.
//
// The primary concern here is that these tests shouldn't crash the type checker.
// The quality of the error messages is secondary as these are all pretty esoteric
// or artificial test cases.

package p

// crash 1
type nt1[_ any]interface{g /* ERROR "undefined" */ }
type ph1[e nt1[e],g(d /* ERROR "undefined" */ )]s /* ERROR "undefined" */
func(*ph1[e,e /* ERROR "redeclared" */ ])h(d /* ERROR "undefined" */ )

// crash 2
// Disabled: empty []'s are now syntax errors. This example leads to too many follow-on errors.
// type Numeric2 interface{t2 /* ERROR "not a type" */ }
// func t2[T Numeric2](s[]T){0 /* ERROR "not a type */ []{s /* ERROR cannot index" */ [0][0]}}

// crash 3
type t3 *interface{ t3.p /* ERROR "t3.p is not a type" */ }

// crash 4
type Numeric4 interface{t4 /* ERROR "not a type" */ }
func t4[T Numeric4](s[]T){if( /* ERROR "non-boolean" */ 0){*s /* ERROR "cannot indirect" */ [0]}}

// crash 7
type foo7 interface { bar() }
type x7[A any] struct{ foo7 }
func main7() { var _ foo7 = x7[int]{} }

// crash 8
type foo8[A any] interface { ~A /* ERROR "cannot be a type parameter" */ }
func bar8[A foo8[A]](a A) {}

// crash 9
type foo9[A any] interface { foo9 /* ERROR "invalid recursive type" */ [A] }
func _() { var _ = new(foo9[int]) }

// crash 12
var u, i [func /* ERROR "used as value" */ /* ERROR "used as value" */ (u /* ERROR "u is not a type" */ /* ERROR "u is not a type" */ , c /* ERROR "undefined" */ /* ERROR "undefined" */ ) {}(0, len /* ERROR "must be called" */ /* ERROR "must be called" */ )]c /* ERROR "undefined" */ /* ERROR "undefined" */

// crash 15
func y15() { var a /* ERROR "declared and not used" */ interface{ p() } = G15[string]{} }
type G15[X any] s /* ERROR "undefined" */
func (G15 /* ERRORx `generic type .* without instantiation` */ ) p()

// crash 16
type Foo16[T any] r16 /* ERROR "not a type" */
func r16[T any]() Foo16[Foo16[T]] { panic(0) }

// crash 17
type Y17 interface{ c() }
type Z17 interface {
	c() Y17
	Y17 /* ERROR "duplicate method" */
}
func F17[T Z17](T) {}

// crash 18
type o18[T any] []func(_ o18[[]_ /* ERROR "cannot use _" */ ])

// crash 19
type Z19 [][[]Z19{}[0][0]]c19 /* ERROR "undefined" */

// crash 20
type Z20 /* ERROR "invalid recursive type" */ interface{ Z20 }
func F20[t Z20]() { F20(t /* ERROR "invalid composite literal type" */ /* ERROR "too many arguments in call to F20\n\thave (unknown type)\n\twant ()" */ {}) }

// crash 21
type Z21 /* ERROR "invalid recursive type" */ interface{ Z21 }
func F21[T Z21]() { ( /* ERROR "not used" */ F21[Z21]) }

// crash 24
type T24[P any] P // ERROR "cannot use a type parameter as RHS in type declaration"
func (r T24[P]) m() { T24 /* ERROR "without instantiation" */ .m() }

// crash 25
type T25[A any] int
func (t T25[A]) m1() {}
var x T25 /* ERROR "without instantiation" */ .m1

// crash 26
type T26 = interface{ F26[ /* ERROR "interface method must have no type parameters" */ Z any]() }
func F26[Z any]() T26 { return F26[] /* ERROR "operand" */ }

// crash 27
func e27[T any]() interface{ x27 /* ERROR "not a type" */ } { panic(0) }
func x27() { e27 /* ERROR "cannot infer T" */ () }

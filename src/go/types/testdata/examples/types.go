// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file shows some examples of generic types.

package p

// List is just what it says - a slice of E elements.
type List[E any] []E

// A generic (parameterized) type must always be instantiated
// before it can be used to designate the type of a variable
// (including a struct field, or function parameter); though
// for the latter cases, the provided type may be another type
// parameter. So:
var _ List[byte] = []byte{}

// A generic binary tree might be declared as follows.
type Tree[E any] struct {
	left, right *Tree[E]
	payload E
}

// A simple instantiation of Tree:
var root1 Tree[int]

// The actual type parameter provided may be a generic type itself:
var root2 Tree[List[int]]

// A couple of more complex examples.
// We don't need extra parentheses around the element type of the slices on
// the right (unlike when we use ()'s rather than []'s for type parameters).
var _ List[List[int]] = []List[int]{}
var _ List[List[List[Tree[int]]]] = []List[List[Tree[int]]]{}

// Type parameters act like type aliases when used in generic types
// in the sense that we can "emulate" a specific type instantiation
// with type aliases.
type T1[P any] struct {
	f P
}

type T2[P any] struct {
	f struct {
		g P
	}
}

var x1 T1[struct{ g int }]
var x2 T2[int]

func _() {
	// This assignment is invalid because the types of x1, x2 are T1(...)
	// and T2(...) respectively, which are two different defined types.
	x1 = x2 // ERROR assignment

	// This assignment is valid because the types of x1.f and x2.f are
	// both struct { g int }; the type parameters act like type aliases
	// and their actual names don't come into play here.
	x1.f = x2.f
}

// We can verify this behavior using type aliases instead:
type T1a struct {
	f A1
}
type A1 = struct { g int }

type T2a struct {
	f struct {
		g A2
	}
}
type A2 = int

var x1a T1a
var x2a T2a

func _() {
	x1a = x2a // ERROR assignment
	x1a.f = x2a.f
}

// Another interesting corner case are generic types that don't use
// their type arguments. For instance:
type T[P any] struct{}

var xint T[int]
var xbool T[bool]

// Are these two variables of the same type? After all, their underlying
// types are identical. We consider them to be different because each type
// instantiation creates a new named type, in this case T<int> and T<bool>
// even if their underlying types are identical. This is sensible because
// we might still have methods that have different signatures or behave
// differently depending on the type arguments, and thus we can't possibly
// consider such types identical. Consequently:
func _() {
	xint = xbool // ERROR assignment
}

// Generic types cannot be used without instantiation.
var _ T // ERROR cannot use generic type T
var _ = T /* ERROR cannot use generic type T */ (0)

// In type context, generic (parameterized) types cannot be parenthesized before
// being instantiated. See also NOTES entry from 12/4/2019.
var _ (T /* ERROR cannot use generic type T */ )[ /* ERROR unexpected \[|expected ';' */ int]

// All types may be parameterized, including interfaces.
type I1[T any] interface{
	m1(T)
}

// There is no such thing as a variadic generic type.
type _[T ... /* ERROR invalid use of ... */ any] struct{}

// Generic interfaces may be embedded as one would expect.
type I2 interface {
	I1(int)     // method!
	I1[string]  // embedded I1
}

func _() {
	var x I2
	x.I1(0)
	x.m1("foo")
}

type I0 interface {
	m0()
}

type I3 interface {
	I0
	I1[bool]
	m(string)
}

func _() {
	var x I3
	x.m0()
	x.m1(true)
	x.m("foo")
}

type _ struct {
	( /* ERROR cannot parenthesize */ int8)
	( /* ERROR cannot parenthesize */ *int16)
	*( /* ERROR cannot parenthesize */ int32)
	List[int]

	int8 /* ERROR int8 redeclared */
	* /* ERROR int16 redeclared */ int16
	List /* ERROR List redeclared */ [int]
}

// Issue #45639: We don't allow this anymore. Keep this code
//               in case we decide to revisit this decision.
//
// It's possible to declare local types whose underlying types
// are type parameters. As with ordinary type definitions, the
// types underlying properties are "inherited" but the methods
// are not.
// func _[T interface{ m(); ~int }]() {
// 	type L T
// 	var x L
// 
// 	// m is not defined on L (it is not "inherited" from
// 	// its underlying type).
// 	x.m /* ERROR x.m undefined */ ()
// 
// 	// But the properties of T, such that as that it supports
// 	// the operations of the types given by its type bound,
// 	// are also the properties of L.
// 	x++
// 	_ = x - x
// 
// 	// On the other hand, if we define a local alias for T,
// 	// that alias stands for T as expected.
// 	type A = T
// 	var y A
// 	y.m()
// 	_ = y < 0
// }

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// // It is not permitted to declare a local type whose underlying
// // type is a type parameter not declared by that type declaration.
// func _[T any]() {
// 	type _ T         // ERROR cannot use function type parameter T as RHS in type declaration
// 	type _ [_ any] T // ERROR cannot use function type parameter T as RHS in type declaration
// }

// As a special case, an explicit type argument may be omitted
// from a type parameter bound if the type bound expects exactly
// one type argument. In that case, the type argument is the
// respective type parameter to which the type bound applies.
// Note: We may not permit this syntactic sugar at first.
// Note: This is now disabled. All examples below are adjusted.
type Adder[T any] interface {
	Add(T) T
}

// We don't need to explicitly instantiate the Adder bound
// if we have exactly one type parameter.
func Sum[T Adder[T]](list []T) T {
	var sum T
	for _, x := range list {
		sum = sum.Add(x)
	}
	return sum
}

// Valid and invalid variations.
type B0 any
type B1[_ any] any
type B2[_, _ any] any

func _[T1 B0]() {}
func _[T1 B1[T1]]() {}
func _[T1 B2 /* ERROR cannot use generic type .* without instantiation */ ]() {}

func _[T1, T2 B0]() {}
func _[T1 B1[T1], T2 B1[T2]]() {}
func _[T1, T2 B2 /* ERROR cannot use generic type .* without instantiation */ ]() {}

func _[T1 B0, T2 B1[T2]]() {} // here B1 applies to T2

// When the type argument is left away, the type bound is
// instantiated for each type parameter with that type
// parameter.
// Note: We may not permit this syntactic sugar at first.
func _[A Adder[A], B Adder[B], C Adder[A]]() {
	var a A // A's type bound is Adder[A]
	a = a.Add(a)
	var b B // B's type bound is Adder[B]
	b = b.Add(b)
	var c C // C's type bound is Adder[A]
	a = c.Add(a)
}

// The type of variables (incl. parameters and return values) cannot
// be an interface with type constraints or be/embed comparable.
type I interface {
	~int
}

var (
	_ interface /* ERROR contains type constraints */ {~int}
	_ I /* ERROR contains type constraints */
)

func _(I /* ERROR contains type constraints */ )
func _(x, y, z I /* ERROR contains type constraints */ )
func _() I /* ERROR contains type constraints */

func _() {
	var _ I /* ERROR contains type constraints */
}

type C interface {
	comparable
}

var _ comparable /* ERROR comparable */
var _ C /* ERROR comparable */

func _(_ comparable /* ERROR comparable */ , _ C /* ERROR comparable */ )

func _() {
	var _ comparable /* ERROR comparable */
	var _ C /* ERROR comparable */
}

// Type parameters are never const types, i.e., it's
// not possible to declare a constant of type parameter type.
// (If a type set contains just a single const type, we could
// allow it, but such type sets don't make much sense in the
// first place.)
func _[T interface{~int|~float64}]() {
	// not valid
	const _ = T /* ERROR not constant */ (0)
	const _ T /* ERROR invalid constant type T */ = 1

	// valid
	var _ = T(0)
	var _ T = 1
	_ = T(0)
}

// It is possible to create composite literals of type parameter
// type as long as it's possible to create a composite literal
// of the core type of the type parameter's constraint.
func _[P interface{ ~[]int }]() P {
	return P{}
	return P{1, 2, 3}
}

func _[P interface{ ~[]E }, E interface{ map[string]P } ]() P {
	x := P{}
	return P{{}}
	return P{E{}}
	return P{E{"foo": x}}
	return P{{"foo": x}, {}}
}

// This is a degenerate case with a singleton type set, but we can create
// composite literals even if the core type is a defined type.
type MyInts []int

func _[P MyInts]() P {
	return P{}
}

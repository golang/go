// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file shows some examples of type-parameterized functions.

package p

// Reverse is a generic function that takes a []T argument and
// reverses that slice in place.
func Reverse[T any](list []T) {
	i := 0
	j := len(list)-1
	for i < j {
		list[i], list[j] = list[j], list[i]
		i++
		j--
	}
}

func _() {
	// Reverse can be called with an explicit type argument.
	Reverse[int](nil)
	Reverse[string]([]string{"foo", "bar"})
	Reverse[struct{x, y int}]([]struct{x, y int}{{1, 2}, {2, 3}, {3, 4}})

	// Since the type parameter is used for an incoming argument,
	// it can be inferred from the provided argument's type.
	Reverse([]string{"foo", "bar"})
	Reverse([]struct{x, y int}{{1, 2}, {2, 3}, {3, 4}})

	// But the incoming argument must have a type, even if it's a
	// default type. An untyped nil won't work.
	// Reverse(nil) // this won't type-check

	// A typed nil will work, though.
	Reverse([]int(nil))
}

// Certain functions, such as the built-in `new` could be written using
// type parameters.
func new[T any]() *T {
	var x T
	return &x
}

// When calling our own `new`, we need to pass the type parameter
// explicitly since there is no (value) argument from which the
// result type could be inferred. We don't try to infer the
// result type from the assignment to keep things simple and
// easy to understand.
var _ = new[int]()
var _ *float64 = new[float64]() // the result type is indeed *float64

// A function may have multiple type parameters, of course.
func foo[A, B, C any](a A, b []B, c *C) B {
	// do something here
	return b[0]
}

// As before, we can pass type parameters explicitly.
var s = foo[int, string, float64](1, []string{"first"}, new[float64]())

// Or we can use type inference.
var _ float64 = foo(42, []float64{1.0}, &s)

// Type inference works in a straight-forward manner even
// for variadic functions.
func variadic[A, B any](A, B, ...B) int { panic(0) }

// var _ = variadic(1) // ERROR not enough arguments
var _ = variadic(1, 2.3)
var _ = variadic(1, 2.3, 3.4, 4.5)
var _ = variadic[int, float64](1, 2.3, 3.4, 4)

// Type inference also works in recursive function calls where
// the inferred type is the type parameter of the caller.
func f1[T any](x T) {
	f1(x)
}

func f2a[T any](x, y T) {
	f2a(x, y)
}

func f2b[T any](x, y T) {
	f2b(y, x)
}

func g2a[P, Q any](x P, y Q) {
	g2a(x, y)
}

func g2b[P, Q any](x P, y Q) {
	g2b(y, x)
}

// Here's an example of a recursive function call with variadic
// arguments and type inference inferring the type parameter of
// the caller (i.e., itself).
func max[T interface{ ~int }](x ...T) T {
	var x0 T
	if len(x) > 0 {
		x0 = x[0]
	}
	if len(x) > 1 {
		x1 := max(x[1:]...)
		if x1 > x0 {
			return x1
		}
	}
	return x0
}

// When inferring channel types, the channel direction is ignored
// for the purpose of type inference. Once the type has been in-
// fered, the usual parameter passing rules are applied.
// Thus even if a type can be inferred successfully, the function
// call may not be valid.

func fboth[T any](chan T) {}
func frecv[T any](<-chan T) {}
func fsend[T any](chan<- T) {}

func _() {
	var both chan int
	var recv <-chan int
	var send chan<-int

	fboth(both)
	fboth(recv /* ERROR cannot use */ )
	fboth(send /* ERROR cannot use */ )

	frecv(both)
	frecv(recv)
	frecv(send /* ERROR cannot use */ )

	fsend(both)
	fsend(recv /* ERROR cannot use */)
	fsend(send)
}

func ffboth[T any](func(chan T)) {}
func ffrecv[T any](func(<-chan T)) {}
func ffsend[T any](func(chan<- T)) {}

func _() {
	var both func(chan int)
	var recv func(<-chan int)
	var send func(chan<- int)

	ffboth(both)
	ffboth(recv /* ERROR cannot use */ )
	ffboth(send /* ERROR cannot use */ )

	ffrecv(both /* ERROR cannot use */ )
	ffrecv(recv)
	ffrecv(send /* ERROR cannot use */ )

	ffsend(both /* ERROR cannot use */ )
	ffsend(recv /* ERROR cannot use */ )
	ffsend(send)
}

// When inferring elements of unnamed composite parameter types,
// if the arguments are defined types, use their underlying types.
// Even though the matching types are not exactly structurally the
// same (one is a type literal, the other a named type), because
// assignment is permitted, parameter passing is permitted as well,
// so type inference should be able to handle these cases well.

func g1[T any]([]T) {}
func g2[T any]([]T, T) {}
func g3[T any](*T, ...T) {}

func _() {
	type intSlize []int
	g1([]int{})
	g1(intSlize{})
	g2(nil, 0)

	type myString string
	var s1 string
	g3(nil, "1", myString("2"), "3")
	g3(& /* ERROR does not match */ s1, "1", myString("2"), "3")
	_ = s1

	type myStruct struct{x int}
	var s2 myStruct
	g3(nil, struct{x int}{}, myStruct{})
	g3(&s2, struct{x int}{}, myStruct{})
	g3(nil, myStruct{}, struct{x int}{})
	g3(&s2, myStruct{}, struct{x int}{})
}

// Here's a realistic example.

func append[T any](s []T, t ...T) []T { panic(0) }

func _() {
	var f func()
	type Funcs []func()
	var funcs Funcs
	_ = append(funcs, f)
}

// Generic type declarations cannot have empty type parameter lists
// (that would indicate a slice type). Thus, generic functions cannot
// have empty type parameter lists, either. This is a syntax error.

func h[] /* ERROR empty type parameter list */ () {}

func _() {
	h /* ERROR cannot index */ [] /* ERROR operand */ ()
}

// Parameterized functions must have a function body.

func _ /* ERROR missing function body */ [P any]()

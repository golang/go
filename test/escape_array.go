// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for arrays and some large things

package foo

var Ssink *string

type U [2]*string

func bar(a, b *string) U { // ERROR "leaking param: a to result ~r2 level=0$" "leaking param: b to result ~r2 level=0$"
	return U{a, b}
}

func foo(x U) U { // ERROR "leaking param: x to result ~r1 level=0$"
	return U{x[1], x[0]}
}

func bff(a, b *string) U { // ERROR "leaking param: a to result ~r2 level=0$" "leaking param: b to result ~r2 level=0$"
	return foo(foo(bar(a, b)))
}

func tbff1() *string {
	a := "cat"
	b := "dog"       // ERROR "moved to heap: b$"
	u := bff(&a, &b) // ERROR "tbff1 &a does not escape$" "tbff1 &b does not escape$"
	_ = u[0]
	return &b // ERROR "&b escapes to heap$"
}

// BAD: need fine-grained analysis to track u[0] and u[1] differently.
func tbff2() *string {
	a := "cat"       // ERROR "moved to heap: a$"
	b := "dog"       // ERROR "moved to heap: b$"
	u := bff(&a, &b) // ERROR "&a escapes to heap$" "&b escapes to heap$"
	_ = u[0]
	return u[1]
}

func car(x U) *string { // ERROR "leaking param: x to result ~r1 level=0$"
	return x[0]
}

// BAD: need fine-grained analysis to track x[0] and x[1] differently.
func fun(x U, y *string) *string { // ERROR "leaking param: x to result ~r2 level=0$" "leaking param: y to result ~r2 level=0$"
	x[0] = y
	return x[1]
}

func fup(x *U, y *string) *string { // ERROR "leaking param: x to result ~r2 level=1$" "leaking param: y$"
	x[0] = y // leaking y to heap is intended
	return x[1]
}

func fum(x *U, y **string) *string { // ERROR "leaking param: x to result ~r2 level=1$" "leaking param content: y$"
	x[0] = *y
	return x[1]
}

func fuo(x *U, y *U) *string { // ERROR "leaking param: x to result ~r2 level=1$" "leaking param content: y$"
	x[0] = y[0]
	return x[1]
}

// These two tests verify that:
// small array literals are stack allocated;
// pointers stored in small array literals do not escape;
// large array literals are heap allocated;
// pointers stored in large array literals escape.
func hugeLeaks1(x **string, y **string) { // ERROR "leaking param content: x" "hugeLeaks1 y does not escape" "mark escaped content: x"
	a := [10]*string{*y}
	_ = a
	// 4 x 4,000,000 exceeds MaxStackVarSize, therefore it must be heap allocated if pointers are 4 bytes or larger.
	b := [4000000]*string{*x} // ERROR "moved to heap: b"
	_ = b
}

func hugeLeaks2(x *string, y *string) { // ERROR "leaking param: x" "hugeLeaks2 y does not escape"
	a := [10]*string{y}
	_ = a
	// 4 x 4,000,000 exceeds MaxStackVarSize, therefore it must be heap allocated if pointers are 4 bytes or larger.
	b := [4000000]*string{x} // ERROR "moved to heap: b"
	_ = b
}

// BAD: x need not leak.
func doesNew1(x *string, y *string) { // ERROR "leaking param: x" "leaking param: y"
	a := new([10]*string) // ERROR "new\(\[10\]\*string\) does not escape"
	a[0] = x
	b := new([65537]*string) // ERROR "new\(\[65537\]\*string\) escapes to heap"
	b[0] = y
}

type a10 struct {
	s *string
	i [10]int32
}

type a65537 struct {
	s *string
	i [65537]int32
}

// BAD: x need not leak.
func doesNew2(x *string, y *string) { // ERROR "leaking param: x" "leaking param: y"
	a := new(a10) // ERROR "new\(a10\) does not escape"
	a.s = x
	b := new(a65537) // ERROR "new\(a65537\) escapes to heap"
	b.s = y
}

// BAD: x need not leak.
func doesMakeSlice(x *string, y *string) { // ERROR "leaking param: x" "leaking param: y"
	a := make([]*string, 10) // ERROR "make\(\[\]\*string, 10\) does not escape"
	a[0] = x
	b := make([]*string, 65537) // ERROR "make\(\[\]\*string, 65537\) escapes to heap"
	b[0] = y
}

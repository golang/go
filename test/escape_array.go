// errorcheck -0 -m -l

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for function parameters.

// In this test almost everything is BAD except the simplest cases
// where input directly flows to output.

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

// BAD: would be nice to record that *y (content) is what leaks, not y itself
func fum(x *U, y **string) *string { // ERROR "leaking param: x to result ~r2 level=1$" "leaking param content: y$"
	x[0] = *y
	return x[1]
}

// BAD: would be nice to record that y[0] (content) is what leaks, not y itself
func fuo(x *U, y *U) *string { // ERROR "leaking param: x to result ~r2 level=1$" "leaking param content: y$"
	x[0] = y[0]
	return x[1]
}

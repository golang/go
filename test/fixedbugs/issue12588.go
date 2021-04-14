// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests escape analysis for range of arrays.
// Compiles but need not run.  Inlining is disabled.

package main

type A struct {
	b [3]uint64
}

type B struct {
	b [3]*uint64
}

func f(a A) int {
	for i, x := range &a.b {
		if x != 0 {
			return 64*i + int(x)
		}
	}
	return 0
}

func g(a *A) int { // ERROR "a does not escape"
	for i, x := range &a.b {
		if x != 0 {
			return 64*i + int(x)
		}
	}
	return 0
}

func h(a *B) *uint64 { // ERROR "leaking param: a to result ~r1 level=1"
	for i, x := range &a.b {
		if i == 0 {
			return x
		}
	}
	return nil
}

func h2(a *B) *uint64 { // ERROR "leaking param: a to result ~r1 level=1"
	p := &a.b
	for i, x := range p {
		if i == 0 {
			return x
		}
	}
	return nil
}

// Seems like below should be level=1, not 0.
func k(a B) *uint64 { // ERROR "leaking param: a to result ~r1 level=0"
	for i, x := range &a.b {
		if i == 0 {
			return x
		}
	}
	return nil
}

var sink *uint64

func main() {
	var a1, a2 A
	var b1, b2, b3, b4 B
	var x1, x2, x3, x4 uint64 // ERROR "moved to heap: x1" "moved to heap: x3"
	b1.b[0] = &x1
	b2.b[0] = &x2
	b3.b[0] = &x3
	b4.b[0] = &x4
	f(a1)
	g(&a2)
	sink = h(&b1)
	h(&b2)
	sink = h2(&b1)
	h2(&b4)
	x1 = 17
	println("*sink=", *sink) // Verify that sink addresses x1
	x3 = 42
	sink = k(b3)
	println("*sink=", *sink) // Verify that sink addresses x3
}

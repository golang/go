// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that heavy recursion works. Simple torture test for
// segmented stacks: do math in unary by recursion.

package main

import "runtime"

type Number *Number

// -------------------------------------
// Peano primitives

func zero() *Number {
	return nil
}

func is_zero(x *Number) bool {
	return x == nil
}

func add1(x *Number) *Number {
	e := new(Number)
	*e = x
	return e
}

func sub1(x *Number) *Number {
	return *x
}

func add(x, y *Number) *Number {
	if is_zero(y) {
		return x
	}

	return add(add1(x), sub1(y))
}

func mul(x, y *Number) *Number {
	if is_zero(x) || is_zero(y) {
		return zero()
	}

	return add(mul(x, sub1(y)), x)
}

func fact(n *Number) *Number {
	if is_zero(n) {
		return add1(zero())
	}

	return mul(fact(sub1(n)), n)
}

// -------------------------------------
// Helpers to generate/count Peano integers

func gen(n int) *Number {
	if n > 0 {
		return add1(gen(n - 1))
	}

	return zero()
}

func count(x *Number) int {
	if is_zero(x) {
		return 0
	}

	return count(sub1(x)) + 1
}

func check(x *Number, expected int) {
	var c = count(x)
	if c != expected {
		print("error: found ", c, "; expected ", expected, "\n")
		panic("fail")
	}
}

// -------------------------------------
// Test basic functionality

func init() {
	check(zero(), 0)
	check(add1(zero()), 1)
	check(gen(10), 10)

	check(add(gen(3), zero()), 3)
	check(add(zero(), gen(4)), 4)
	check(add(gen(3), gen(4)), 7)

	check(mul(zero(), zero()), 0)
	check(mul(gen(3), zero()), 0)
	check(mul(zero(), gen(4)), 0)
	check(mul(gen(3), add1(zero())), 3)
	check(mul(add1(zero()), gen(4)), 4)
	check(mul(gen(3), gen(4)), 12)

	check(fact(zero()), 1)
	check(fact(add1(zero())), 1)
	check(fact(gen(5)), 120)
}

// -------------------------------------
// Factorial

var results = [...]int{
	1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800,
	39916800, 479001600,
}

func main() {
	max := 9
	if runtime.GOARCH == "wasm" {
		max = 7 // stack size is limited
	}
	for i := 0; i <= max; i++ {
		if f := count(fact(gen(i))); f != results[i] {
			println("FAIL:", i, "!:", f, "!=", results[i])
			panic(0)
		}
	}
}

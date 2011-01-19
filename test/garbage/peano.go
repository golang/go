// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
	"time"
)


type Number struct {
	next *Number
}


// -------------------------------------
// Peano primitives

func zero() *Number { return nil }


func is_zero(x *Number) bool { return x == nil }


func add1(x *Number) *Number {
	e := new(Number)
	e.next = x
	return e
}


func sub1(x *Number) *Number { return x.next }


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
		panic(fmt.Sprintf("error: found %d; expected %d", c, expected))
	}
}


// -------------------------------------
// Test basic functionality

func verify() {
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


func main() {
	t0 := time.Nanoseconds()
	verify()
	for i := 0; i <= 9; i++ {
		print(i, "! = ", count(fact(gen(i))), "\n")
	}
	runtime.GC()
	t1 := time.Nanoseconds()

	gcstats("BenchmarkPeano", 1, t1-t0)
}

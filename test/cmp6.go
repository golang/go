// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that incorrect comparisons are detected.
// Does not compile.

package main

func use(bool) {}

type T1 *int
type T2 *int

type T3 struct{ z []int }

var t3 T3

type T4 struct {
	_ []int
	a float64
}

var t4 T4

func main() {
	// Arguments to comparison must be
	// assignable one to the other (or vice versa)
	// so chan int can be compared against
	// directional channels but channel of different
	// direction cannot be compared against each other.
	var c1 chan<- int
	var c2 <-chan int
	var c3 chan int

	use(c1 == c2) // ERROR "invalid operation|incompatible"
	use(c2 == c1) // ERROR "invalid operation|incompatible"
	use(c1 == c3)
	use(c2 == c2)
	use(c3 == c1)
	use(c3 == c2)

	// Same applies to named types.
	var p1 T1
	var p2 T2
	var p3 *int

	use(p1 == p2) // ERROR "invalid operation|incompatible"
	use(p2 == p1) // ERROR "invalid operation|incompatible"
	use(p1 == p3)
	use(p2 == p2)
	use(p3 == p1)
	use(p3 == p2)

	// Arrays are comparable if and only if their element type is comparable.
	var a1 [1]int
	var a2 [1]func()
	var a3 [0]func()
	use(a1 == a1)
	use(a2 == a2) // ERROR "invalid operation|invalid comparison"
	use(a3 == a3) // ERROR "invalid operation|invalid comparison"

	// Comparison of structs should have a good message
	use(t3 == t3) // ERROR "struct|expected|cannot compare"
	use(t4 == t4) // ERROR "cannot be compared|non-comparable|cannot compare"

	// Slices, functions, and maps too.
	var x []int
	var f func()
	var m map[int]int
	use(x == x) // ERROR "slice can only be compared to nil|cannot compare"
	use(f == f) // ERROR "func can only be compared to nil|cannot compare"
	use(m == m) // ERROR "map can only be compared to nil|cannot compare"

	// Comparison with interface that cannot return true
	// (would panic).
	var i interface{}
	use(i == x) // ERROR "invalid operation"
	use(x == i) // ERROR "invalid operation"
	use(i == f) // ERROR "invalid operation"
	use(f == i) // ERROR "invalid operation"
	use(i == m) // ERROR "invalid operation"
	use(m == i) // ERROR "invalid operation"
}

// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

var trace []int

func f(x int) int {
	trace = append(trace, x)
	return x
}

func wantTrace(what string, want ...int) {
	if len(trace) != len(want) {
		fmt.Printf("%s: evaluation order %v, want %v\n", what, trace, want)
		panic(what)
	}
	for i, x := range want {
		if trace[i] != x {
			fmt.Printf("%s: evaluation order %v, want %v\n", what, trace, want)
			panic(what)
		}
	}
	trace = trace[:0]
}

func wantShape(what string, s []int, length, capacity int) {
	if len(s) != length || cap(s) != capacity {
		fmt.Printf("%s: len %d cap %d, want len %d cap %d\n", what, len(s), cap(s), length, capacity)
		panic(what)
	}
}

func main() {
	// The length is evaluated before the capacity, even though the
	// capacity is what gets rewritten.
	s := make([]int, f(1), f(3))
	wantTrace("make", 1, 3)
	wantShape("make", s, 1, 3)

	// Same, for a capacity whose value comes from the length of a slice
	// literal rather than from a constant.
	s = make([]int, f(2), len([]int{0, 0, 0, 0}))
	wantTrace("make with len of literal", 2)
	wantShape("make with len of literal", s, 2, 4)

	// A two-operand make rewrites the length itself, which stays first.
	s = make([]int, f(4))
	wantTrace("make without cap", 4)
	wantShape("make without cap", s, 4, 4)
}

// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure that zeroing range loops have the requisite side-effects.

package main

import (
	"fmt"
	"os"
)

func check(n int) {
	// When n == 0, i is untouched by the range loop.
	// Picking an initial value of -1 for i makes the
	// "want" calculation below correct in all cases.
	i := -1
	s := make([]byte, n)
	for i = range s {
		s[i] = 0
	}
	if want := n - 1; i != want {
		fmt.Printf("index after range with side-effect = %d want %d\n", i, want)
		os.Exit(1)
	}

	i = n + 1
	// i is shadowed here, so its value should be unchanged.
	for i := range s {
		s[i] = 0
	}
	if want := n + 1; i != want {
		fmt.Printf("index after range without side-effect = %d want %d\n", i, want)
		os.Exit(1)
	}

	// Index variable whose evaluation has side-effects
	var x int
	f := func() int {
		x++
		return 0
	}
	var a [1]int
	for a[f()] = range s {
		s[a[f()]] = 0
	}
	if want := n * 2; x != want {
		fmt.Printf("index function calls = %d want %d\n", x, want)
		os.Exit(1)
	}

	// Range expression whose evaluation has side-effects
	x = 0
	b := [1][]byte{s}
	for i := range b[f()] {
		b[f()][i] = 0
	}
	if want := n + 1; x != n+1 {
		fmt.Printf("range expr function calls = %d want %d\n", x, want)
		os.Exit(1)
	}
}

func main() {
	check(0)
	check(1)
	check(15)
}

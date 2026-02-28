// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure that range loops over a string have the requisite side-effects.

package main

import (
	"fmt"
	"os"
)

func check(n int) {
	var i int
	var r rune

	b := make([]byte, n)
	for i = range b {
		b[i] = byte(i + 1)
	}
	s := string(b)

	// When n == 0, i is untouched by the range loop.
	// Picking an initial value of -1 for i makes the
	// "want" calculation below correct in all cases.
	i = -1
	for i = range s {
		b[i] = s[i]
	}
	if want := n - 1; i != want {
		fmt.Printf("index after range with side-effect = %d want %d\n", i, want)
		os.Exit(1)
	}

	i = -1
	r = '\x00'
	for i, r = range s {
		b[i] = byte(r)
	}
	if want := n - 1; i != want {
		fmt.Printf("index after range with side-effect = %d want %d\n", i, want)
		os.Exit(1)
	}
	if want := rune(n); r != want {
		fmt.Printf("rune after range with side-effect = %q want %q\n", r, want)
		os.Exit(1)
	}

	i = -1
	// i is shadowed here, so its value should be unchanged.
	for i := range s {
		b[i] = s[i]
	}
	if want := -1; i != want {
		fmt.Printf("index after range without side-effect = %d want %d\n", i, want)
		os.Exit(1)
	}

	i = -1
	r = -1
	// i and r are shadowed here, so their values should be unchanged.
	for i, r := range s {
		b[i] = byte(r)
	}
	if want := -1; i != want {
		fmt.Printf("index after range without side-effect = %d want %d\n", i, want)
		os.Exit(1)
	}
	if want := rune(-1); r != want {
		fmt.Printf("rune after range without side-effect = %q want %q\n", r, want)
		os.Exit(1)
	}
}

func main() {
	check(0)
	check(1)
	check(15)
}

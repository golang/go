// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure that range loops over maps with delete statements
// have the requisite side-effects.

package main

import (
	"fmt"
	"os"
)

func checkcleared() {
	m := make(map[byte]int)
	m[1] = 1
	m[2] = 2
	for k := range m {
		delete(m, k)
	}
	l := len(m)
	if want := 0; l != want {
		fmt.Printf("len after map clear = %d want %d\n", l, want)
		os.Exit(1)
	}

	m[0] = 0 // To have non empty map and avoid internal map code fast paths.
	n := 0
	for range m {
		n++
	}
	if want := 1; n != want {
		fmt.Printf("number of keys found = %d want %d\n", n, want)
		os.Exit(1)
	}
}

func checkloopvars() {
	k := 0
	m := make(map[int]int)
	m[42] = 0
	for k = range m {
		delete(m, k)
	}
	if want := 42; k != want {
		fmt.Printf("var after range with side-effect = %d want %d\n", k, want)
		os.Exit(1)
	}
}

func checksideeffects() {
	var x int
	f := func() int {
		x++
		return 0
	}
	m := make(map[int]int)
	m[0] = 0
	m[1] = 1
	for k := range m {
		delete(m, k+f())
	}
	if want := 2; x != want {
		fmt.Printf("var after range with side-effect = %d want %d\n", x, want)
		os.Exit(1)
	}

	var n int
	m = make(map[int]int)
	m[0] = 0
	m[1] = 1
	for k := range m {
		delete(m, k)
		n++
	}
	if want := 2; n != want {
		fmt.Printf("counter for range with side-effect = %d want %d\n", n, want)
		os.Exit(1)
	}
}

func main() {
	checkcleared()
	checkloopvars()
	checksideeffects()
}

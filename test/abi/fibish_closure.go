// run

//go:build !wasm

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

// Test that register results are correctly returned (and passed)

type MagicLastTypeNameForTestingRegisterABI func(int, MagicLastTypeNameForTestingRegisterABI) (int, int)

//go:noinline
func f(x int, unused MagicLastTypeNameForTestingRegisterABI) (int, int) {

	if x < 3 {
		return 0, x
	}

	a, b := f(x-2, unused)
	c, d := f(x-1, unused)
	return a + d, b + c
}

func main() {
	x := 40
	a, b := f(x, f)
	fmt.Printf("f(%d)=%d,%d\n", x, a, b)
}

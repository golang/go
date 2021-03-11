// run

//go:build !wasm
// +build !wasm

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

// Test that register results are correctly returned (and passed)

type MagicLastTypeNameForTestingRegisterABI func(int, MagicLastTypeNameForTestingRegisterABI) int

//go:registerparams
//go:noinline
func minus(decrement int) MagicLastTypeNameForTestingRegisterABI {
	return MagicLastTypeNameForTestingRegisterABI(func(x int, _ MagicLastTypeNameForTestingRegisterABI) int { return x - decrement })
}

//go:noinline
func f(x int, sub1 MagicLastTypeNameForTestingRegisterABI) (int, int) {

	if x < 3 {
		return 0, x
	}

	a, b := f(sub1(sub1(x, sub1), sub1), sub1)
	c, d := f(sub1(x, sub1), sub1)
	return a + d, b + c
}

func main() {
	x := 40
	a, b := f(x, minus(1))
	fmt.Printf("f(%d)=%d,%d\n", x, a, b)
}

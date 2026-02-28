// run

//go:build !wasm

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// wasm is excluded because the compiler chatter about register abi pragma ends up
// on stdout, and causes the expected output to not match.

package main

import "fmt"

type Z struct {
}

type NZ struct {
	x, y int
}

//go:noinline
func f(x, y int) (Z, NZ, Z) {
	var z Z
	return z, NZ{x, y}, z
}

//go:noinline
func g() (Z, NZ, Z) {
	a, b, c := f(3, 4)
	return c, b, a
}

func main() {
	_, b, _ := g()
	fmt.Println(b.x + b.y)
}

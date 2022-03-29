// run

//go:build !wasm
// +build !wasm

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// wasm is excluded because the compiler chatter about register abi pragma ends up
// on stdout, and causes the expected output to not match.

package main

import "fmt"

type i5f5 struct {
	a, b          int16
	c, d, e       int32
	r, s, t, u, v float32
}

//go:noinline
func spills(_ *float32) {

}

//go:registerparams
//go:noinline
func F(x i5f5) i5f5 {
	y := x.v
	spills(&y)
	x.r = y
	return x
}

func main() {
	x := i5f5{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	y := x
	z := F(x)
	if (i5f5{1, 2, 3, 4, 5, 10, 7, 8, 9, 10}) != z {
		fmt.Printf("y=%v, z=%v\n", y, z)
	}
}

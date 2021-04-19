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

type i4 struct {
	a, b, c, d int
}

//go:registerparams
//go:noinline
func F(x i4) i4 {
	ab := x.a + x.b
	bc := x.b + x.c
	cd := x.c + x.d
	ad := x.a + x.d
	ba := x.a - x.b
	cb := x.b - x.c
	dc := x.c - x.d
	da := x.a - x.d

	return i4{ab*bc + da, cd*ad + cb, ba*cb + ad, dc*da + bc}
}

func main() {
	x := i4{1, 2, 3, 4}
	y := x
	z := F(x)
	if (i4{12, 34, 6, 8}) != z {
		fmt.Printf("y=%v, z=%v\n", y, z)
	}
}

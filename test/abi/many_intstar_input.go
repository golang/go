// run

//go:build !wasm
// +build !wasm

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// wasm is excluded because the compiler chatter about register abi pragma ends up
// on stdout, and causes the expected output to not match.

package main

import (
	"fmt"
)

var sink int = 3

//go:registerparams
//go:noinline
func F(a, b, c, d, e, f *int) {
	G(f, e, d, c, b, a)
	sink += *a // *a == 6 after swapping in G
}

//go:registerparams
//go:noinline
func G(a, b, c, d, e, f *int) {
	var scratch [1000 * 100]int
	scratch[*a] = *f                    // scratch[6] = 1
	fmt.Println(*a, *b, *c, *d, *e, *f) // Forces it to spill b
	sink = scratch[*b+1]                // scratch[5+1] == 1
	*f, *a = *a, *f
	*e, *b = *b, *e
	*d, *c = *c, *d
}

func main() {
	a, b, c, d, e, f := 1, 2, 3, 4, 5, 6
	F(&a, &b, &c, &d, &e, &f)
	fmt.Println(a, b, c, d, e, f)
	fmt.Println(sink)
}

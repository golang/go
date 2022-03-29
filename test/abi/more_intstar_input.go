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

var sink int

//go:registerparams
//go:noinline
func F(a, b, c, d, e, f, g, h, i, j, k, l, m *int) {
	G(m, l, k, j, i, h, g, f, e, d, c, b, a)
	// did the pointers get properly updated?
	sink = *a + *m
}

//go:registerparams
//go:noinline
func G(a, b, c, d, e, f, g, h, i, j, k, l, m *int) {
	// Do not reference the parameters
	var scratch [1000 * 100]int
	I := *c - *e - *l // zero.
	scratch[I] = *d
	fmt.Println("Got this far!")
	sink += scratch[0]
}

func main() {
	a, b, c, d, e, f, g, h, i, j, k, l, m := 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
	F(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l, &m)
	fmt.Printf("Sink = %d\n", sink-7)
}

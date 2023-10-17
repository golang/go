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

// This test is designed to provoke a stack growth
// in a way that very likely leaves junk in the
// parameter save area if they aren't saved or spilled
// there, as appropriate.

//go:registerparams
//go:noinline
func f(x int, xm1, xm2, p *int) {
	var y = [2]int{x - 4, 0}
	if x < 2 {
		*p += x
		return
	}
	x -= 3
	g(*xm1, xm2, &x, p)   // xm1 is no longer live.
	h(*xm2, &x, &y[0], p) // xm2 is no longer live, but was spilled.
}

//go:registerparams
//go:noinline
func g(x int, xm1, xm2, p *int) {
	var y = [3]int{x - 4, 0, 0}
	if x < 2 {
		*p += x
		return
	}
	x -= 3
	k(*xm2, &x, &y[0], p)
	h(*xm1, xm2, &x, p)
}

//go:registerparams
//go:noinline
func h(x int, xm1, xm2, p *int) {
	var y = [4]int{x - 4, 0, 0, 0}
	if x < 2 {
		*p += x
		return
	}
	x -= 3
	k(*xm1, xm2, &x, p)
	f(*xm2, &x, &y[0], p)
}

//go:registerparams
//go:noinline
func k(x int, xm1, xm2, p *int) {
	var y = [5]int{x - 4, 0, 0, 0, 0}
	if x < 2 {
		*p += x
		return
	}
	x -= 3
	f(*xm2, &x, &y[0], p)
	g(*xm1, xm2, &x, p)
}

func main() {
	x := 40
	var y int
	xm1 := x - 1
	xm2 := x - 2
	f(x, &xm1, &xm2, &y)

	fmt.Printf("Fib(%d)=%d\n", x, y)
}

// run

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 16515: spilled Duff-adjusted address may be invalid

package main

import "runtime"

type T [62]int // DUFFZERO with non-zero adjustment on AMD64

var sink interface{}

//go:noinline
func zero(x *T) {
	// Two DUFFZEROs on the same address with a function call in between.
	// Duff-adjusted address will be spilled and loaded

	*x = T{} // DUFFZERO
	runtime.GC()
	(*x)[0] = 1
	g()      // call a function with large frame, trigger a stack move
	*x = T{} // DUFFZERO again
}

//go:noinline
// a function with large frame
func g() {
	var x [1000]int
	_ = x
}

func main() {
	var s struct { a T; b [8192-62]int } // allocate 64K, hopefully it's in a new span and a few bytes before it is garbage
	sink = &s // force heap allocation
	s.a[0] = 2
	zero(&s.a)
	if s.a[0] != 0 {
		println("s.a[0] =", s.a[0])
		panic("zeroing failed")
	}

	var a T // on stack
	a[0] = 2
	zero(&a)
	if a[0] != 0 {
		println("a[0] =", a[0])
		panic("zeroing failed")
	}
}

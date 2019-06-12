// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Values smaller than 64-bits were mistakenly always proven to be
// non-negative.
//
// The tests here are marked go:noinline to ensure they're
// independently optimized by SSA.

package main

var x int32 = -1

//go:noinline
func a() {
	if x != -1 {
		panic(1)
	}
	if x > 0 || x != -1 {
		panic(2)
	}
}

//go:noinline
func b() {
	if x != -1 {
		panic(3)
	}
	if x > 0 {
		panic(4)
	}
}

//go:noinline
func c() {
	if x > 0 || x != -1 {
		panic(5)
	}
	if x > 0 || x != -1 {
		panic(6)
	}
}

func main() {
	a()
	b()
	c()
}

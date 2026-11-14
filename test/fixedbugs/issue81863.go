// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func f(c bool) int {
	var x, y int
	p := &x
	if c {
		p = &y
	}
	*p = 1
	return x
}

//go:noinline
func g(c bool) int {
	var x, y int
	p := &y
	if c {
		p = &x
	}
	*p = 1
	return x
}

func main() {
	if got := f(false); got != 1 {
		panic(got)
	}
	if got := f(true); got != 0 {
		panic(got)
	}
	if got := g(false); got != 0 {
		panic(got)
	}
	if got := g(true); got != 1 {
		panic(got)
	}
}

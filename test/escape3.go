// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the run-time behavior of escape analysis-related optimizations.

package main

func main() {
	test1()
}

func test1() {
	check1(0)
	check1(1)
	check1(2)
}

type T1 struct {
	X, Y, Z int
}

func f() int {
	return 1
}

func check1(pass int) T1 {
	v := []T1{{X: f(), Z: f()}}
	if v[0].Y != 0 {
		panic("nonzero init")
	}
	v[0].Y = pass
	return v[0]
}

// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func g() {}

func f1() (a, b int) {
	a, b = 2, 1
	g() // defeat optimizer
	return a, b
}

func f2() (a, b int) {
	a, b = 1, 2
	g() // defeat optimizer
	return b, a
}

func main() {
	x, y := f1()
	if x != 2 || y != 1 {
		println("f1", x, y)
		panic("fail")
	}

	x, y = f2()
	if x != 2 || y != 1 {
		println("f2", x, y)
		panic("fail")
	}
}

// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test simultaneous assignment.

package main

var a, b, c, d, e, f, g, h, i int

func printit() {
	println(a, b, c, d, e, f, g, h, i)
}

func testit(permuteok bool) bool {
	if a+b+c+d+e+f+g+h+i != 45 {
		print("sum does not add to 45\n")
		printit()
		return false
	}
	return permuteok ||
		a == 1 &&
			b == 2 &&
			c == 3 &&
			d == 4 &&
			e == 5 &&
			f == 6 &&
			g == 7 &&
			h == 8 &&
			i == 9
}

func swap(x, y int) (u, v int) {
	return y, x
}

func main() {
	a = 1
	b = 2
	c = 3
	d = 4
	e = 5
	f = 6
	g = 7
	h = 8
	i = 9

	if !testit(false) {
		panic("init val\n")
	}

	for z := 0; z < 100; z++ {
		a, b, c, d, e, f, g, h, i = b, c, d, a, i, e, f, g, h

		if !testit(z%20 != 19) {
			print("on ", z, "th iteration\n")
			printit()
			panic("fail")
		}
	}

	if !testit(false) {
		print("final val\n")
		printit()
		panic("fail")
	}

	a, b = swap(1, 2)
	if a != 2 || b != 1 {
		panic("bad swap")
	}

	a, b = swap(swap(a, b))
	if a != 2 || b != 1 {
		panic("bad swap")
	}
}

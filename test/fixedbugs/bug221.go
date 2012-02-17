// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// function call arg reordering was picking out 1 call that
// didn't need to be in a temporary, but it was picking
// out the first call instead of the last call.
// http://code.google.com/p/go/issues/detail?id=370

package main

var gen = 'a'

func f(n int) string {
	s := string(gen) + string(n+'A'-1)
	gen++
	return s
}

func g(x, y string) string {
	return x + y
}

func main() {
	s := f(1) + f(2)
	if s != "aAbB" {
		println("BUG: bug221a: ", s)
		panic("fail")
	}
	s = g(f(3), f(4))
	if s != "cCdD" {
		println("BUG: bug221b: ", s)
		panic("fail")
	}
	s = f(5) + f(6) + f(7) + f(8) + f(9)
	if s != "eEfFgGhHiI" {
		println("BUG: bug221c: ", s)
		panic("fail")
	}
}

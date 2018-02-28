// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	v1 := [2]int32{-1, 88}
	v2 := [2]int32{-1, 99}
	if v1 == v2 {
		panic("bad comparison")
	}

	w1 := [2]int16{-1, 88}
	w2 := [2]int16{-1, 99}
	if w1 == w2 {
		panic("bad comparison")
	}
	x1 := [4]int16{-1, 88, 88, 88}
	x2 := [4]int16{-1, 99, 99, 99}
	if x1 == x2 {
		panic("bad comparison")
	}

	a1 := [2]int8{-1, 88}
	a2 := [2]int8{-1, 99}
	if a1 == a2 {
		panic("bad comparison")
	}
	b1 := [4]int8{-1, 88, 88, 88}
	b2 := [4]int8{-1, 99, 99, 99}
	if b1 == b2 {
		panic("bad comparison")
	}
	c1 := [8]int8{-1, 88, 88, 88, 88, 88, 88, 88}
	c2 := [8]int8{-1, 99, 99, 99, 99, 99, 99, 99}
	if c1 == c2 {
		panic("bad comparison")
	}
}

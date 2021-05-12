// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that when a function recovers from a panic, it
// returns the correct results to the caller (in particular,
// setting the result registers correctly).

package main

type S struct {
	x uint8
	y uint16
	z uint32
	w float64
}

var a0, b0, c0, d0 = 10, "hello", S{1, 2, 3, 4}, [2]int{111, 222}

//go:noinline
//go:registerparams
func F() (a int, b string, _ int, c S, d [2]int) {
	a, b, c, d = a0, b0, c0, d0
	defer func() { recover() }()
	panic("XXX")
	return
}

func main() {
	a1, b1, zero, c1, d1 := F()
	if a1 != a0 || b1 != b0 || c1 != c0 || d1 != d0 || zero != 0 { // unnamed result gets zero value
		panic("FAIL")
	}
}

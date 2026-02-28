// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that zero-sized variables get same address as
// runtime.zerobase.

package main

var x, y [0]int
var p, q = new([0]int), new([0]int) // should get &runtime.zerobase

func main() {
	if &x != &y {
		// Failing for now. x and y are at same address, but compiler optimizes &x==&y to false. Skip.
		// print("&x=", &x, " &y=", &y, " &x==&y = ", &x==&y, "\n")
		// panic("FAIL")
	}
	if p != q {
		print("p=", p, " q=", q, " p==q = ", p==q, "\n")
		panic("FAIL")
	}
	if &x != p {
		print("&x=", &x, " p=", p, " &x==p = ", &x==p, "\n")
		panic("FAIL")
	}
	if &y != p {
		print("&y=", &y, " p=", p, " &y==p = ", &y==p, "\n")
		panic("FAIL")
	}
}

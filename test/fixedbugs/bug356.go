// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1808

package main

func main() {
	var i uint64
	var x int = 12345

	if y := x << (i&5); y != 12345<<0 {
		println("BUG bug344", y)
		return
	}
	
	i++
	if y := x << (i&5); y != 12345<<1 {
		println("BUG bug344a", y)
	}
	
	i = 70
	if y := x << i; y != 0 {
		println("BUG bug344b", y)
	}
	
	i = 1<<32
	if y := x << i; y != 0 {
		println("BUG bug344c", y)
	}
}
	

/*
typecheck [1008592b0]
.   INDREG a(1) l(15) x(24) tc(2) runtime.ret G0 string
bug343.go:15: internal compiler error: typecheck INDREG
*/

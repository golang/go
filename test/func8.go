// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test evaluation order.

package main

var calledf int

func f() int {
	calledf++
	return 0
}

func g() int {
	return calledf
}

var xy string

func x() bool {
	for false {
	} // no inlining
	xy += "x"
	return false
}

func y() string {
	for false {
	} // no inlining
	xy += "y"
	return "abc"
}

func main() {
	if f() == g() {
		println("wrong f,g order")
	}

	if x() == (y() == "abc") {
		panic("wrong compare")
	}
	if xy != "xy" {
		println("wrong x,y order")
	}
}

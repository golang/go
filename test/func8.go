// run

// Copyright 2011 The Go Authors. All rights reserved.
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

//go:noinline
func x() bool {
	xy += "x"
	return false
}

//go:noinline
func y() string {
	xy += "y"
	return "abc"
}

func main() {
	if f() == g() {
		panic("wrong f,g order")
	}

	if x() == (y() == "abc") {
		panic("wrong compare")
	}
	if xy != "xy" {
		panic("wrong x,y order")
	}
}

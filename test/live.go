// errorcheck -0 -l -live

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f1() {
	var x *int
	print(&x) // ERROR "live at call to printpointer: x$"
	print(&x) // ERROR "live at call to printpointer: x$"
}

func f2(b bool) {
	if b {
		print(0) // nothing live here
		return
	}
	var x *int
	print(&x) // ERROR "live at call to printpointer: x$"
	print(&x) // ERROR "live at call to printpointer: x$"
}

func f3(b bool) {
	print(0)
	if b == false {
		print(0) // nothing live here
		return
	}

	if b {
		var x *int
		print(&x) // ERROR "live at call to printpointer: x$"
		print(&x) // ERROR "live at call to printpointer: x$"
	} else {
		var y *int
		print(&y) // ERROR "live at call to printpointer: y$"
		print(&y) // ERROR "live at call to printpointer: y$"
	}
	print(0) // ERROR "live at call to printint: x y$"
}

// The old algorithm treated x as live on all code that
// could flow to a return statement, so it included the
// function entry and code above the declaration of x
// but would not include an indirect use of x in an infinite loop.
// Check that these cases are handled correctly.

func f4(b1, b2 bool) { // x not live here
	if b2 {
		print(0) // x not live here
		return
	}
	var z **int
	x := new(int)
	*x = 42
	z = &x
	print(**z) // ERROR "live at call to printint: x z$"
	if b2 {
		print(1) // ERROR "live at call to printint: x$"
		return
	}
	for {
		print(**z) // ERROR "live at call to printint: x z$"
	}
}

func f5(b1 bool) {
	var z **int
	if b1 {
		x := new(int)
		*x = 42
		z = &x
	} else {
		y := new(int)
		*y = 54
		z = &y
	}
	print(**z) // ERROR "live at call to printint: x y$"
}

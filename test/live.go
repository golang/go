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

// confusion about the _ result used to cause spurious "live at entry to f6: _".

func f6() (_, y string) {
	y = "hello"
	return
}

// confusion about addressed results used to cause "live at entry to f7: x".

func f7() (x string) {
	_ = &x
	x = "hello"
	return
}

// ignoring block returns used to cause "live at entry to f8: x, y".

func f8() (x, y string) {
	return g8()
}

func g8() (string, string)

// ignoring block assignments used to cause "live at entry to f9: x"
// issue 7205

var i9 interface{}

func f9() bool {
	g8()
	x := i9
	return x != 99
}

// liveness formerly confused by UNDEF followed by RET,
// leading to "live at entry to f10: ~r1" (unnamed result).

func f10() string {
	panic(1)
}

// liveness formerly confused by select, thinking runtime.selectgo
// can return to next instruction; it always jumps elsewhere.
// note that you have to use at least two cases in the select
// to get a true select; smaller selects compile to optimized helper functions.

var c chan *int
var b bool

// this used to have a spurious "live at entry to f11a: ~r0"
func f11a() *int {
	select { // ERROR "live at call to selectgo: autotmp"
	case <-c: // ERROR "live at call to selectrecv: autotmp"
		return nil
	case <-c: // ERROR "live at call to selectrecv: autotmp"
		return nil
	}
}

func f11b() *int {
	p := new(int)
	if b {
		// At this point p is dead: the code here cannot
		// get to the bottom of the function.
		// This used to have a spurious "live at call to printint: p".
		print(1) // nothing live here!
		select { // ERROR "live at call to selectgo: autotmp"
		case <-c: // ERROR "live at call to selectrecv: autotmp"
			return nil
		case <-c: // ERROR "live at call to selectrecv: autotmp"
			return nil
		}
	}
	println(*p)
	return nil
}

func f11c() *int {
	p := new(int)
	if b {
		// Unlike previous, the cases in this select fall through,
		// so we can get to the println, so p is not dead.
		print(1) // ERROR "live at call to printint: p"
		select { // ERROR "live at call to newselect: p" "live at call to selectgo: autotmp.* p"
		case <-c: // ERROR "live at call to selectrecv: autotmp.* p"
		case <-c: // ERROR "live at call to selectrecv: autotmp.* p"
		}
	}
	println(*p)
	return nil
}

// similarly, select{} does not fall through.
// this used to have a spurious "live at entry to f12: ~r0".

func f12() *int {
	if b {
		select{}
	} else {
		return nil
	}
}

// incorrectly placed VARDEF annotations can cause missing liveness annotations.
// this used to be missing the fact that s is live during the call to g13 (because it is
// needed for the call to h13).

func f13() {
	s := "hello"
	s = h13(s, g13(s)) // ERROR "live at call to g13: s"
}

func g13(string) string
func h13(string, string) string

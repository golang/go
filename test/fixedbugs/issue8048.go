// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8048. Incorrect handling of liveness when walking stack
// containing faulting frame.

package main

import "runtime"

func main() {
	test1()
	test2()
	test3()
}

func test1() {
	// test1f will panic without its own defer.
	// The runtime.GC checks that we can walk the stack
	// at that point and not get confused.
	// The recover lets test1 exit normally.
	defer func() {
		runtime.GC()
		recover()
	}()
	test1f()
}

func test1f() {
	// Because b == false, the if does not execute,
	// so x == nil, so the println(*x) faults reading
	// from nil. The compiler will lay out the code
	// so that the if body occurs above the *x,
	// so if the liveness info at the *x is used, it will
	// find the liveness at the call to runtime.GC.
	// It will think y is live, but y is uninitialized,
	// and the runtime will crash detecting a bad slice.
	// The runtime should see that there are no defers
	// corresponding to this panicked frame and ignore
	// the frame entirely.
	var x *int
	var b bool
	if b {
		y := make([]int, 1)
		runtime.GC()
		x = &y[0]
	}
	println(*x)
}

func test2() {
	// Same as test1, but the fault happens in the function with the defer.
	// The runtime should see the defer and garbage collect the frame
	// as if the PC were immediately after the defer statement.
	defer func() {
		runtime.GC()
		recover()
	}()
	var x *int
	var b bool
	if b {
		y := make([]int, 1)
		runtime.GC()
		x = &y[0]
	}
	println(*x)
}

func test3() {
	// Like test1 but avoid array index, which does not
	// move to end of function on ARM.
	defer func() {
		runtime.GC()
		recover()
	}()
	test3setup()
	test3f()
}

func test3setup() {
	var x uintptr
	var b bool
	b = true
	if b {
		y := uintptr(123)
		runtime.GC()
		x = y
	}
	runtime.GC()
	globl = x
}

var globl uintptr

func test3f() {
	var x *int
	var b bool
	if b {
		y := new(int)
		runtime.GC()
		x = y
	}
	println(*x)
}

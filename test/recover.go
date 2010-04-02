// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test of basic recover functionality.

package main

import "runtime"

func main() {
	test1()
	test1WithClosures()
	test2()
	test3()
	test4()
	test5()
	test6()
	test6WithClosures()
	test7()
}

func die() {
	runtime.Breakpoint() // can't depend on panic
}

func mustRecover(x interface{}) {
	mustNotRecover() // because it's not a defer call
	v := recover()
	if v == nil {
		println("missing recover")
		die() // panic is useless here
	}
	if v != x {
		println("wrong value", v, x)
		die()
	}

	// the value should be gone now regardless
	v = recover()
	if v != nil {
		println("recover didn't recover")
		die()
	}
}

func mustNotRecover() {
	v := recover()
	if v != nil {
		println("spurious recover", v)
		die()
	}
}

func withoutRecover() {
	mustNotRecover() // because it's a sub-call
}

func test1() {
	defer mustNotRecover() // because mustRecover will squelch it
	defer mustRecover(1)   // because of panic below
	defer withoutRecover() // should be no-op, leaving for mustRecover to find
	panic(1)
}

// Repeat test1 with closures instead of standard function.
// Interesting because recover bases its decision
// on the frame pointer of its caller, and a closure's
// frame pointer is in the middle of its actual arguments
// (after the hidden ones for the closed-over variables).
func test1WithClosures() {
	defer func() {
		v := recover()
		if v != nil {
			println("spurious recover in closure")
			die()
		}
	}()
	defer func(x interface{}) {
		mustNotRecover()
		v := recover()
		if v == nil {
			println("missing recover")
			die()
		}
		if v != x {
			println("wrong value", v, x)
			die()
		}
	}(1)
	defer func() {
		mustNotRecover()
	}()
	panic(1)
}

func test2() {
	// Recover only sees the panic argument
	// if it is called from a deferred call.
	// It does not see the panic when called from a call within a deferred call (too late)
	// nor does it see the panic when it *is* the deferred call (too early).
	defer mustRecover(2)
	defer recover() // should be no-op
	panic(2)
}

func test3() {
	defer mustNotRecover()
	defer func() {
		recover() // should squelch
	}()
	panic(3)
}

func test4() {
	// Equivalent to test3 but using defer to make the call.
	defer mustNotRecover()
	defer func() {
		defer recover() // should squelch
	}()
	panic(4)
}

// Check that closures can set output arguments.
// Run g().  If it panics, return x; else return deflt.
func try(g func(), deflt interface{}) (x interface{}) {
	defer func() {
		if v := recover(); v != nil {
			x = v
		}
	}()
	defer g()
	return deflt
}

// Check that closures can set output arguments.
// Run g().  If it panics, return x; else return deflt.
func try1(g func(), deflt interface{}) (x interface{}) {
	defer func() {
		if v := recover(); v != nil {
			x = v
		}
	}()
	defer g()
	x = deflt
	return
}

func test5() {
	v := try(func() { panic(5) }, 55).(int)
	if v != 5 {
		println("wrong value", v, 5)
		die()
	}

	s := try(func() {}, "hi").(string)
	if s != "hi" {
		println("wrong value", s, "hi")
		die()
	}

	v = try1(func() { panic(5) }, 55).(int)
	if v != 5 {
		println("try1 wrong value", v, 5)
		die()
	}

	s = try1(func() {}, "hi").(string)
	if s != "hi" {
		println("try1 wrong value", s, "hi")
		die()
	}
}

// When a deferred big call starts, it must first
// create yet another stack segment to hold the
// giant frame for x.  Make sure that doesn't
// confuse recover.
func big(mustRecover bool) {
	var x [100000]int
	x[0] = 1
	x[99999] = 1
	_ = x

	v := recover()
	if mustRecover {
		if v == nil {
			println("missing big recover")
			die()
		}
	} else {
		if v != nil {
			println("spurious big recover")
			die()
		}
	}
}

func test6() {
	defer big(false)
	defer big(true)
	panic(6)
}

func test6WithClosures() {
	defer func() {
		var x [100000]int
		x[0] = 1
		x[99999] = 1
		_ = x
		if recover() != nil {
			println("spurious big closure recover")
			die()
		}
	}()
	defer func() {
		var x [100000]int
		x[0] = 1
		x[99999] = 1
		_ = x
		if recover() == nil {
			println("missing big closure recover")
			die()
		}
	}()
	panic("6WithClosures")
}

func test7() {
	ok := false
	func() {
		// should panic, then call mustRecover 7, which stops the panic.
		// then should keep processing ordinary defers earlier than that one
		// before returning.
		// this test checks that the defer func on the next line actually runs.
		defer func() { ok = true }()
		defer mustRecover(7)
		panic(7)
	}()
	if !ok {
		println("did not run ok func")
		die()
	}
}

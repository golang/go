// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test of recover during recursive panics.
// Here be dragons.

package main

import "runtime"

func main() {
	test1()
	test2()
	test3()
	test4()
	test5()
	test6()
	test7()
}

func die() {
	runtime.Breakpoint()	// can't depend on panic
}

func mustRecover(x interface{}) {
	mustNotRecover()	// because it's not a defer call
	v := recover()
	if v == nil {
		println("missing recover")
		die()	// panic is useless here
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
		println("spurious recover")
		die()
	}
}

func withoutRecover() {
	mustNotRecover()	// because it's a sub-call
}

func test1() {
	// Easy nested recursive panic.
	defer mustRecover(1)
	defer func() {
		defer mustRecover(2)
		panic(2)
	}()
	panic(1)
}

func test2() {
	// Sequential panic.
	defer mustNotRecover()
	defer func() {
		v := recover()
		if v == nil || v.(int) != 2 {
			println("wrong value", v, 2)
			die()
		}
		defer mustRecover(3)
		panic(3)
	}()
	panic(2)
}

func test3() {
	// Sequential panic - like test2 but less picky.
	defer mustNotRecover()
	defer func() {
		recover()
		defer mustRecover(3)
		panic(3)
	}()
	panic(2)
}

func test4() {
	// Single panic.
	defer mustNotRecover()
	defer func() {
		recover()
	}()
	panic(4)
}

func test5() {
	// Single panic but recover called via defer
	defer mustNotRecover()
	defer func() {
		defer recover()
	}()
	panic(5)
}

func test6() {
	// Sequential panic.
	// Like test3, but changed recover to defer (same change as test4 → test5).
	defer mustNotRecover()
	defer func() {
		defer recover()	// like a normal call from this func; runs because mustRecover stops the panic
		defer mustRecover(3)
		panic(3)
	}()
	panic(2)
}

func test7() {
	// Like test6, but swapped defer order.
	// The recover in "defer recover()" is now a no-op,
	// because it runs called from panic, not from the func,
	// and therefore cannot see the panic of 2.
	// (Alternately, it cannot see the panic of 2 because
	// there is an active panic of 3.  And it cannot see the
	// panic of 3 because it is at the wrong level (too high on the stack).)
	defer mustRecover(2)
	defer func() {
		defer mustRecover(3)
		defer recover()	// now a no-op, unlike in test6.
		panic(3)
	}()
	panic(2)
}

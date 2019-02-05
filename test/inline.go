// errorcheck -0 -m

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that inlining is working.
// Compiles but does not run.

package foo

import (
	"errors"
	"runtime"
	"unsafe"
)

func add2(p *byte, n uintptr) *byte { // ERROR "can inline add2" "leaking param: p to result"
	return (*byte)(add1(unsafe.Pointer(p), n)) // ERROR "inlining call to add1"
}

func add1(p unsafe.Pointer, x uintptr) unsafe.Pointer { // ERROR "can inline add1" "leaking param: p to result"
	return unsafe.Pointer(uintptr(p) + x)
}

func f(x *byte) *byte { // ERROR "can inline f" "leaking param: x to result"
	return add2(x, 1) // ERROR "inlining call to add2" "inlining call to add1"
}

//go:noinline
func g(x int) int {
	return x + 1
}

func h(x int) int { // ERROR "can inline h"
	return x + 2
}

func i(x int) int { // ERROR "can inline i"
	const y = 2
	return x + y
}

func j(x int) int { // ERROR "can inline j"
	switch {
	case x > 0:
		return x + 2
	default:
		return x + 1
	}
}

var somethingWrong error = errors.New("something went wrong")

// local closures can be inlined
func l(x, y int) (int, int, error) {
	e := func(err error) (int, int, error) { // ERROR "can inline l.func1" "func literal does not escape" "leaking param: err to result"
		return 0, 0, err
	}
	if x == y {
		e(somethingWrong) // ERROR "inlining call to l.func1"
	}
	return y, x, nil
}

// any re-assignment prevents closure inlining
func m() int {
	foo := func() int { return 1 } // ERROR "can inline m.func1" "func literal does not escape"
	x := foo()
	foo = func() int { return 2 } // ERROR "can inline m.func2" "func literal does not escape"
	return x + foo()
}

// address taking prevents closure inlining
func n() int {
	foo := func() int { return 1 } // ERROR "can inline n.func1" "func literal does not escape"
	bar := &foo                    // ERROR "&foo does not escape"
	x := (*bar)() + foo()
	return x
}

// make sure assignment inside closure is detected
func o() int {
	foo := func() int { return 1 } // ERROR "can inline o.func1" "func literal does not escape"
	func(x int) {                  // ERROR "func literal does not escape"
		if x > 10 {
			foo = func() int { return 2 } // ERROR "can inline o.func2" "func literal escapes"
		}
	}(11)
	return foo()
}

func p() int {
	return func() int { return 42 }() // ERROR "can inline p.func1" "inlining call to p.func1"
}

func q(x int) int {
	foo := func() int { return x * 2 } // ERROR "can inline q.func1" "q func literal does not escape"
	return foo()                       // ERROR "inlining call to q.func1"
}

func r(z int) int {
	foo := func(x int) int { // ERROR "can inline r.func1" "r func literal does not escape"
		return x + z
	}
	bar := func(x int) int { // ERROR "r func literal does not escape"
		return x + func(y int) int { // ERROR "can inline r.func2.1"
			return 2*y + x*z
		}(x) // ERROR "inlining call to r.func2.1"
	}
	return foo(42) + bar(42) // ERROR "inlining call to r.func1"
}

func s0(x int) int {
	foo := func() { // ERROR "can inline s0.func1" "s0 func literal does not escape"
		x = x + 1
	}
	foo() // ERROR "inlining call to s0.func1" "&x does not escape"
	return x
}

func s1(x int) int {
	foo := func() int { // ERROR "can inline s1.func1" "s1 func literal does not escape"
		return x
	}
	x = x + 1
	return foo() // ERROR "inlining call to s1.func1" "&x does not escape"
}

// can't currently inline functions with a break statement
func switchBreak(x, y int) int {
	var n int
	switch x {
	case 0:
		n = 1
	Done:
		switch y {
		case 0:
			n += 10
			break Done
		}
		n = 2
	}
	return n
}

// can't currently inline functions with a type switch
func switchType(x interface{}) int { // ERROR "switchType x does not escape"
	switch x.(type) {
	case int:
		return x.(int)
	default:
		return 0
	}
}

type T struct{}

func (T) meth(int, int) {} // ERROR "can inline T.meth"

func k() (T, int, int) { return T{}, 0, 0 } // ERROR "can inline k"

func _() { // ERROR "can inline _"
	T.meth(k()) // ERROR "inlining call to k" "inlining call to T.meth"
}

func small1() { // ERROR "can inline small1"
	runtime.GC()
}
func small2() int { // ERROR "can inline small2"
	return runtime.GOMAXPROCS(0)
}
func small3(t T) { // ERROR "can inline small3"
	t.meth2(3, 5)
}
func small4(t T) { // not inlineable - has 2 calls.
	t.meth2(runtime.GOMAXPROCS(0), 5)
}
func (T) meth2(int, int) { // not inlineable - has 2 calls.
	runtime.GC()
	runtime.GC()
}

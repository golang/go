// errorcheckwithauto -0 -m -d=inlfuncswithclosures=1

//go:build !goexperiment.newinliner

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

func f2() int { // ERROR "can inline f2"
	tmp1 := h
	tmp2 := tmp1
	return tmp2(0) // ERROR "inlining call to h"
}

var abc = errors.New("abc") // ERROR "inlining call to errors.New"

var somethingWrong error

// local closures can be inlined
func l(x, y int) (int, int, error) { // ERROR "can inline l"
	e := func(err error) (int, int, error) { // ERROR "can inline l.func1" "func literal does not escape" "leaking param: err to result"
		return 0, 0, err
	}
	if x == y {
		e(somethingWrong) // ERROR "inlining call to l.func1"
	} else {
		f := e
		f(nil) // ERROR "inlining call to l.func1"
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
	bar := &foo
	x := (*bar)() + foo()
	return x
}

// make sure assignment inside closure is detected
func o() int {
	foo := func() int { return 1 } // ERROR "can inline o.func1" "func literal does not escape"
	func(x int) {                  // ERROR "can inline o.func2"
		if x > 10 {
			foo = func() int { return 2 } // ERROR "can inline o.func2"
		}
	}(11) // ERROR "func literal does not escape" "inlining call to o.func2"
	return foo()
}

func p() int { // ERROR "can inline p"
	return func() int { return 42 }() // ERROR "can inline p.func1" "inlining call to p.func1"
}

func q(x int) int { // ERROR "can inline q"
	foo := func() int { return x * 2 } // ERROR "can inline q.func1" "func literal does not escape"
	return foo()                       // ERROR "inlining call to q.func1"
}

func r(z int) int {
	foo := func(x int) int { // ERROR "can inline r.func1" "func literal does not escape"
		return x + z
	}
	bar := func(x int) int { // ERROR "func literal does not escape" "can inline r.func2"
		return x + func(y int) int { // ERROR "can inline r.func2.1" "can inline r.r.func2.func3"
			return 2*y + x*z
		}(x) // ERROR "inlining call to r.func2.1"
	}
	return foo(42) + bar(42) // ERROR "inlining call to r.func1" "inlining call to r.func2" "inlining call to r.r.func2.func3"
}

func s0(x int) int { // ERROR "can inline s0"
	foo := func() { // ERROR "can inline s0.func1" "func literal does not escape"
		x = x + 1
	}
	foo() // ERROR "inlining call to s0.func1"
	return x
}

func s1(x int) int { // ERROR "can inline s1"
	foo := func() int { // ERROR "can inline s1.func1" "func literal does not escape"
		return x
	}
	x = x + 1
	return foo() // ERROR "inlining call to s1.func1"
}

func switchBreak(x, y int) int { // ERROR "can inline switchBreak"
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

func switchType(x interface{}) int { // ERROR "can inline switchType" "x does not escape"
	switch x.(type) {
	case int:
		return x.(int)
	default:
		return 0
	}
}

// Test that switches on constant things, with constant cases, only cost anything for
// the case that matches. See issue 50253.
func switchConst1(p func(string)) { // ERROR "can inline switchConst" "p does not escape"
	const c = 1
	switch c {
	case 0:
		p("zero")
	case 1:
		p("one")
	case 2:
		p("two")
	default:
		p("other")
	}
}

func switchConst2() string { // ERROR "can inline switchConst2"
	switch runtime.GOOS {
	case "linux":
		return "Leenooks"
	case "windows":
		return "Windoze"
	case "darwin":
		return "MackBone"
	case "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100":
		return "Numbers"
	default:
		return "oh nose!"
	}
}
func switchConst3() string { // ERROR "can inline switchConst3"
	switch runtime.GOOS {
	case "Linux":
		panic("Linux")
	case "Windows":
		panic("Windows")
	case "Darwin":
		panic("Darwin")
	case "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100":
		panic("Numbers")
	default:
		return "oh nose!"
	}
}
func switchConst4() { // ERROR "can inline switchConst4"
	const intSize = 32 << (^uint(0) >> 63)
	want := func() string { // ERROR "can inline switchConst4.func1"
		switch intSize {
		case 32:
			return "32"
		case 64:
			return "64"
		default:
			panic("unreachable")
		}
	}() // ERROR "inlining call to switchConst4.func1"
	_ = want
}

func inlineRangeIntoMe(data []int) { // ERROR "can inline inlineRangeIntoMe" "data does not escape"
	rangeFunc(data, 12) // ERROR "inlining call to rangeFunc"
}

func rangeFunc(xs []int, b int) int { // ERROR "can inline rangeFunc" "xs does not escape"
	for i, x := range xs {
		if x == b {
			return i
		}
	}
	return -1
}

type T struct{}

func (T) meth(int, int) {} // ERROR "can inline T.meth"

func k() (T, int, int) { return T{}, 0, 0 } // ERROR "can inline k"

func f3() { // ERROR "can inline f3"
	T.meth(k()) // ERROR "inlining call to k" "inlining call to T.meth"
	// ERRORAUTO "inlining call to T.meth"
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

// Issue #29737 - make sure we can do inlining for a chain of recursive functions
func ee() { // ERROR "can inline ee"
	ff(100) // ERROR "inlining call to ff" "inlining call to gg" "inlining call to hh"
}

func ff(x int) { // ERROR "can inline ff"
	if x < 0 {
		return
	}
	gg(x - 1) // ERROR "inlining call to gg" "inlining call to hh"
}
func gg(x int) { // ERROR "can inline gg"
	hh(x - 1) // ERROR "inlining call to hh" "inlining call to ff"
}
func hh(x int) { // ERROR "can inline hh"
	ff(x - 1) // ERROR "inlining call to ff" "inlining call to gg"
}

// Issue #14768 - make sure we can inline for loops.
func for1(fn func() bool) { // ERROR "can inline for1" "fn does not escape"
	for {
		if fn() {
			break
		} else {
			continue
		}
	}
}

func for2(fn func() bool) { // ERROR "can inline for2" "fn does not escape"
Loop:
	for {
		if fn() {
			break Loop
		} else {
			continue Loop
		}
	}
}

// Issue #18493 - make sure we can do inlining of functions with a method value
type T1 struct{}

func (a T1) meth(val int) int { // ERROR "can inline T1.meth"
	return val + 5
}

func getMeth(t1 T1) func(int) int { // ERROR "can inline getMeth"
	return t1.meth // ERROR "t1.meth escapes to heap"
	// ERRORAUTO "inlining call to T1.meth"
}

func ii() { // ERROR "can inline ii"
	var t1 T1
	f := getMeth(t1) // ERROR "inlining call to getMeth" "t1.meth does not escape"
	_ = f(3)
}

// Issue #42194 - make sure that functions evaluated in
// go and defer statements can be inlined.
func gd1(int) {
	defer gd1(gd2()) // ERROR "inlining call to gd2"
	defer gd3()()    // ERROR "inlining call to gd3"
	go gd1(gd2())    // ERROR "inlining call to gd2"
	go gd3()()       // ERROR "inlining call to gd3"
}

func gd2() int { // ERROR "can inline gd2"
	return 1
}

func gd3() func() { // ERROR "can inline gd3"
	return ii
}

// Issue #42788 - ensure ODEREF OCONVNOP* OADDR is low cost.
func EncodeQuad(d []uint32, x [6]float32) { // ERROR "can inline EncodeQuad" "d does not escape"
	_ = d[:6]
	d[0] = float32bits(x[0]) // ERROR "inlining call to float32bits"
	d[1] = float32bits(x[1]) // ERROR "inlining call to float32bits"
	d[2] = float32bits(x[2]) // ERROR "inlining call to float32bits"
	d[3] = float32bits(x[3]) // ERROR "inlining call to float32bits"
	d[4] = float32bits(x[4]) // ERROR "inlining call to float32bits"
	d[5] = float32bits(x[5]) // ERROR "inlining call to float32bits"
}

// float32bits is a copy of math.Float32bits to ensure that
// these tests pass with `-gcflags=-l`.
func float32bits(f float32) uint32 { // ERROR "can inline float32bits"
	return *(*uint32)(unsafe.Pointer(&f))
}

// Ensure OCONVNOP is zero cost.
func Conv(v uint64) uint64 { // ERROR "can inline Conv"
	return conv2(conv2(conv2(v))) // ERROR "inlining call to (conv1|conv2)"
}
func conv2(v uint64) uint64 { // ERROR "can inline conv2"
	return conv1(conv1(conv1(conv1(v)))) // ERROR "inlining call to conv1"
}
func conv1(v uint64) uint64 { // ERROR "can inline conv1"
	return uint64(uint64(uint64(uint64(uint64(uint64(uint64(uint64(uint64(uint64(uint64(v)))))))))))
}

func select1(x, y chan bool) int { // ERROR "can inline select1" "x does not escape" "y does not escape"
	select {
	case <-x:
		return 1
	case <-y:
		return 2
	}
}

func select2(x, y chan bool) { // ERROR "can inline select2" "x does not escape" "y does not escape"
loop: // test that labeled select can be inlined.
	select {
	case <-x:
		break loop
	case <-y:
	}
}

func inlineSelect2(x, y chan bool) { // ERROR "can inline inlineSelect2" ERROR "x does not escape" "y does not escape"
loop:
	for i := 0; i < 5; i++ {
		if i == 3 {
			break loop
		}
		select2(x, y) // ERROR "inlining call to select2"
	}
}

// Issue #62211: inlining a function with unreachable "return"
// statements could trip up phi insertion.
func issue62211(x bool) { // ERROR "can inline issue62211"
	if issue62211F(x) { // ERROR "inlining call to issue62211F"
	}
	if issue62211G(x) { // ERROR "inlining call to issue62211G"
	}

	// Initial fix CL caused a "non-monotonic scope positions" failure
	// on code like this.
	if z := 0; false {
		panic(z)
	}
}

func issue62211F(x bool) bool { // ERROR "can inline issue62211F"
	if x || true {
		return true
	}
	return true
}

func issue62211G(x bool) bool { // ERROR "can inline issue62211G"
	if x || true {
		return true
	} else {
		return true
	}
}

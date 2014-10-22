// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test of basic recover functionality.

package main

import (
	"os"
	"reflect"
	"runtime"
)

func main() {
	// go.tools/ssa/interp still has:
	// - some lesser bugs in recover()
	// - incomplete support for reflection
	interp := os.Getenv("GOSSAINTERP") != ""

	test1()
	test1WithClosures()
	test2()
	test3()
	if !interp {
		test4()
	}
	test5()
	test6()
	test6WithClosures()
	test7()
	test8()
	test9()
	if !interp {
		test9reflect1()
		test9reflect2()
	}
	test10()
	if !interp {
		test10reflect1()
		test10reflect2()
	}
	test11()
	if !interp {
		test11reflect1()
		test11reflect2()
	}
	test111()
	test12()
	if !interp {
		test12reflect1()
		test12reflect2()
	}
	test13()
	if !interp {
		test13reflect1()
		test13reflect2()
	}
	test14()
	if !interp {
		test14reflect1()
		test14reflect2()
		test15()
		test16()
	}
}

func die() {
	runtime.Breakpoint() // can't depend on panic
}

func mustRecoverBody(v1, v2, v3, x interface{}) {
	v := v1
	if v != nil {
		println("spurious recover", v)
		die()
	}
	v = v2
	if v == nil {
		println("missing recover", x.(int))
		die() // panic is useless here
	}
	if v != x {
		println("wrong value", v, x)
		die()
	}

	// the value should be gone now regardless
	v = v3
	if v != nil {
		println("recover didn't recover")
		die()
	}
}

func doubleRecover() interface{} {
	return recover()
}

func mustRecover(x interface{}) {
	mustRecoverBody(doubleRecover(), recover(), recover(), x)
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

func withoutRecoverRecursive(n int) {
	if n == 0 {
		withoutRecoverRecursive(1)
	} else {
		v := recover()
		if v != nil {
			println("spurious recover (recursive)", v)
			die()
		}
	}
}

func test1() {
	defer mustNotRecover()           // because mustRecover will squelch it
	defer mustRecover(1)             // because of panic below
	defer withoutRecover()           // should be no-op, leaving for mustRecover to find
	defer withoutRecoverRecursive(0) // ditto
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
			println("missing recover", x.(int))
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

func varargs(s *int, a ...int) {
	*s = 0
	for _, v := range a {
		*s += v
	}
	if recover() != nil {
		*s += 100
	}
}

func test8a() (r int) {
	defer varargs(&r, 1, 2, 3)
	panic(0)
}

func test8b() (r int) {
	defer varargs(&r, 4, 5, 6)
	return
}

func test8() {
	if test8a() != 106 || test8b() != 15 {
		println("wrong value")
		die()
	}
}

type I interface {
	M()
}

// pointer receiver, so no wrapper in i.M()
type T1 struct{}

func (*T1) M() {
	mustRecoverBody(doubleRecover(), recover(), recover(), 9)
}

func test9() {
	var i I = &T1{}
	defer i.M()
	panic(9)
}

func test9reflect1() {
	f := reflect.ValueOf(&T1{}).Method(0).Interface().(func())
	defer f()
	panic(9)
}

func test9reflect2() {
	f := reflect.TypeOf(&T1{}).Method(0).Func.Interface().(func(*T1))
	defer f(&T1{})
	panic(9)
}

// word-sized value receiver, so no wrapper in i.M()
type T2 uintptr

func (T2) M() {
	mustRecoverBody(doubleRecover(), recover(), recover(), 10)
}

func test10() {
	var i I = T2(0)
	defer i.M()
	panic(10)
}

func test10reflect1() {
	f := reflect.ValueOf(T2(0)).Method(0).Interface().(func())
	defer f()
	panic(10)
}

func test10reflect2() {
	f := reflect.TypeOf(T2(0)).Method(0).Func.Interface().(func(T2))
	defer f(T2(0))
	panic(10)
}

// tiny receiver, so basic wrapper in i.M()
type T3 struct{}

func (T3) M() {
	mustRecoverBody(doubleRecover(), recover(), recover(), 11)
}

func test11() {
	var i I = T3{}
	defer i.M()
	panic(11)
}

func test11reflect1() {
	f := reflect.ValueOf(T3{}).Method(0).Interface().(func())
	defer f()
	panic(11)
}

func test11reflect2() {
	f := reflect.TypeOf(T3{}).Method(0).Func.Interface().(func(T3))
	defer f(T3{})
	panic(11)
}

// tiny receiver, so basic wrapper in i.M()
type T3deeper struct{}

func (T3deeper) M() {
	badstate() // difference from T3
	mustRecoverBody(doubleRecover(), recover(), recover(), 111)
}

func test111() {
	var i I = T3deeper{}
	defer i.M()
	panic(111)
}

type Tiny struct{}

func (Tiny) M() {
	panic(112)
}

// i.M is a wrapper, and i.M panics.
//
// This is a torture test for an old implementation of recover that
// tried to deal with wrapper functions by doing some argument
// positioning math on both entry and exit. Doing anything on exit
// is a problem because sometimes functions exit via panic instead
// of an ordinary return, so panic would have to know to do the
// same math when unwinding the stack. It gets complicated fast.
// This particular test never worked with the old scheme, because
// panic never did the right unwinding math.
//
// The new scheme adjusts Panic.argp on entry to a wrapper.
// It has no exit work, so if a wrapper is interrupted by a panic,
// there's no cleanup that panic itself must do.
// This test just works now.
func badstate() {
	defer func() {
		recover()
	}()
	var i I = Tiny{}
	i.M()
}

// large receiver, so basic wrapper in i.M()
type T4 [2]string

func (T4) M() {
	mustRecoverBody(doubleRecover(), recover(), recover(), 12)
}

func test12() {
	var i I = T4{}
	defer i.M()
	panic(12)
}

func test12reflect1() {
	f := reflect.ValueOf(T4{}).Method(0).Interface().(func())
	defer f()
	panic(12)
}

func test12reflect2() {
	f := reflect.TypeOf(T4{}).Method(0).Func.Interface().(func(T4))
	defer f(T4{})
	panic(12)
}

// enormous receiver, so wrapper splits stack to call M
type T5 [8192]byte

func (T5) M() {
	mustRecoverBody(doubleRecover(), recover(), recover(), 13)
}

func test13() {
	var i I = T5{}
	defer i.M()
	panic(13)
}

func test13reflect1() {
	f := reflect.ValueOf(T5{}).Method(0).Interface().(func())
	defer f()
	panic(13)
}

func test13reflect2() {
	f := reflect.TypeOf(T5{}).Method(0).Func.Interface().(func(T5))
	defer f(T5{})
	panic(13)
}

// enormous receiver + enormous method frame, so wrapper splits stack to call M,
// and then M splits stack to allocate its frame.
// recover must look back two frames to find the panic.
type T6 [8192]byte

var global byte

func (T6) M() {
	var x [8192]byte
	x[0] = 1
	x[1] = 2
	for i := range x {
		global += x[i]
	}
	mustRecoverBody(doubleRecover(), recover(), recover(), 14)
}

func test14() {
	var i I = T6{}
	defer i.M()
	panic(14)
}

func test14reflect1() {
	f := reflect.ValueOf(T6{}).Method(0).Interface().(func())
	defer f()
	panic(14)
}

func test14reflect2() {
	f := reflect.TypeOf(T6{}).Method(0).Func.Interface().(func(T6))
	defer f(T6{})
	panic(14)
}

// function created by reflect.MakeFunc

func reflectFunc(args []reflect.Value) (results []reflect.Value) {
	mustRecoverBody(doubleRecover(), recover(), recover(), 15)
	return nil
}

func test15() {
	f := reflect.MakeFunc(reflect.TypeOf((func())(nil)), reflectFunc).Interface().(func())
	defer f()
	panic(15)
}

func reflectFunc2(args []reflect.Value) (results []reflect.Value) {
	// This will call reflectFunc3
	args[0].Interface().(func())()
	return nil
}

func reflectFunc3(args []reflect.Value) (results []reflect.Value) {
	if v := recover(); v != nil {
		println("spurious recover", v)
		die()
	}
	return nil
}

func test16() {
	defer mustRecover(16)

	f2 := reflect.MakeFunc(reflect.TypeOf((func(func()))(nil)), reflectFunc2).Interface().(func(func()))
	f3 := reflect.MakeFunc(reflect.TypeOf((func())(nil)), reflectFunc3).Interface().(func())
	defer f2(f3)

	panic(16)
}

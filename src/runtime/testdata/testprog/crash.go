// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
)

func init() {
	register("Crash", Crash)
	register("DoublePanic", DoublePanic)
	register("ErrorPanic", ErrorPanic)
	register("StringerPanic", StringerPanic)
	register("DoubleErrorPanic", DoubleErrorPanic)
	register("DoubleStringerPanic", DoubleStringerPanic)
	register("StringPanic", StringPanic)
	register("NilPanic", NilPanic)
	register("CircularPanic", CircularPanic)
	register("RepanickedPanic", RepanickedPanic)
	register("RepanickedMiddlePanic", RepanickedMiddlePanic)
	register("RepanickedPanicSandwich", RepanickedPanicSandwich)
}

func test(name string) {
	defer func() {
		if x := recover(); x != nil {
			fmt.Printf(" recovered")
		}
		fmt.Printf(" done\n")
	}()
	fmt.Printf("%s:", name)
	var s *string
	_ = *s
	fmt.Print("SHOULD NOT BE HERE")
}

func testInNewThread(name string) {
	c := make(chan bool)
	go func() {
		runtime.LockOSThread()
		test(name)
		c <- true
	}()
	<-c
}

func Crash() {
	runtime.LockOSThread()
	test("main")
	testInNewThread("new-thread")
	testInNewThread("second-new-thread")
	test("main-again")
}

type P string

func (p P) String() string {
	// Try to free the "YYY" string header when the "XXX"
	// panic is stringified.
	runtime.GC()
	runtime.GC()
	runtime.GC()
	return string(p)
}

// Test that panic message is not clobbered.
// See issue 30150.
func DoublePanic() {
	defer func() {
		panic(P("YYY"))
	}()
	panic(P("XXX"))
}

// Test that panic while panicking discards error message
// See issue 52257
type exampleError struct{}

func (e exampleError) Error() string {
	panic("important multi-line\nerror message")
}

func ErrorPanic() {
	panic(exampleError{})
}

type examplePanicError struct{}

func (e examplePanicError) Error() string {
	panic(exampleError{})
}

func DoubleErrorPanic() {
	panic(examplePanicError{})
}

type exampleStringer struct{}

func (s exampleStringer) String() string {
	panic("important multi-line\nstringer message")
}

func StringerPanic() {
	panic(exampleStringer{})
}

type examplePanicStringer struct{}

func (s examplePanicStringer) String() string {
	panic(exampleStringer{})
}

func DoubleStringerPanic() {
	panic(examplePanicStringer{})
}

func StringPanic() {
	panic("important multi-line\nstring message")
}

func NilPanic() {
	panic(nil)
}

type exampleCircleStartError struct{}

func (e exampleCircleStartError) Error() string {
	panic(exampleCircleEndError{})
}

type exampleCircleEndError struct{}

func (e exampleCircleEndError) Error() string {
	panic(exampleCircleStartError{})
}

func CircularPanic() {
	panic(exampleCircleStartError{})
}

func RepanickedPanic() {
	defer func() {
		panic(recover())
	}()
	panic("message")
}

func RepanickedMiddlePanic() {
	defer func() {
		recover()
		panic("outer")
	}()
	func() {
		defer func() {
			panic(recover())
		}()
		func() {
			defer func() {
				recover()
				panic("middle")
			}()
			panic("inner")
		}()
	}()
}

// Panic sandwich:
//
//	panic("outer") =>
//	recovered, panic("inner") =>
//	panic(recovered outer panic value)
//
// Exercises the edge case where we repanic a panic value,
// but with another panic in the middle.
func RepanickedPanicSandwich() {
	var outer any
	defer func() {
		recover()
		panic(outer)
	}()
	func() {
		defer func() {
			outer = recover()
			panic("inner")
		}()
		panic("outer")
	}()
}

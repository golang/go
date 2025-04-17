// run

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Previously, cmd/compile would rewrite
//
//     check(unsafe.Pointer(testMeth(1).Pointer()), unsafe.Pointer(testMeth(2).Pointer()))
//
// to
//
//     var autotmp_1 uintptr = testMeth(1).Pointer()
//     var autotmp_2 uintptr = testMeth(2).Pointer()
//     check(unsafe.Pointer(autotmp_1), unsafe.Pointer(autotmp_2))
//
// However, that means autotmp_1 is the only reference to the int
// variable containing the value "1", but it's not a pointer type,
// so it was at risk of being garbage collected by the evaluation of
// testMeth(2).Pointer(), even though package unsafe's documentation
// says the original code was allowed.
//
// Now cmd/compile rewrites it to
//
//     var autotmp_1 unsafe.Pointer = unsafe.Pointer(testMeth(1).Pointer())
//     var autotmp_2 unsafe.Pointer = unsafe.Pointer(testMeth(2).Pointer())
//     check(autotmp_1, autotmp_2)
//
// to ensure the pointed-to variables are visible to the GC.

package main

import (
	"fmt"
	"reflect"
	"runtime"
	"unsafe"
)

func main() {
	// Test all the different ways we can invoke reflect.Value.Pointer.

	// Direct method invocation.
	check(unsafe.Pointer(testMeth(1).Pointer()), unsafe.Pointer(testMeth(2).Pointer()))

	// Invocation via method expression.
	check(unsafe.Pointer(reflect.Value.Pointer(testMeth(1))), unsafe.Pointer(reflect.Value.Pointer(testMeth(2))))

	// Invocation via interface.
	check(unsafe.Pointer(testInter(1).Pointer()), unsafe.Pointer(testInter(2).Pointer()))

	// Invocation via method value.
	check(unsafe.Pointer(testFunc(1)()), unsafe.Pointer(testFunc(2)()))
}

func check(p, q unsafe.Pointer) {
	a, b := *(*int)(p), *(*int)(q)
	if a != 1 || b != 2 {
		fmt.Printf("got %v, %v; expected 1, 2\n", a, b)
	}
}

func testMeth(x int) reflect.Value {
	// Force GC to run.
	runtime.GC()
	return reflect.ValueOf(&x)
}

type Pointerer interface {
	Pointer() uintptr
}

func testInter(x int) Pointerer {
	return testMeth(x)
}

func testFunc(x int) func() uintptr {
	return testMeth(x).Pointer
}

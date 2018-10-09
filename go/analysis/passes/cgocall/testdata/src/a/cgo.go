// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the cgo checker.

package a

// void f(void *ptr) {}
import "C"

import "unsafe"

func CgoTests() {
	var c chan bool
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&c))) // want "embedded pointer"
	C.f(unsafe.Pointer(&c))                     // want "embedded pointer"

	var m map[string]string
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&m))) // want "embedded pointer"
	C.f(unsafe.Pointer(&m))                     // want "embedded pointer"

	var f func()
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&f))) // want "embedded pointer"
	C.f(unsafe.Pointer(&f))                     // want "embedded pointer"

	var s []int
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&s))) // want "embedded pointer"
	C.f(unsafe.Pointer(&s))                     // want "embedded pointer"

	var a [1][]int
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&a))) // want "embedded pointer"
	C.f(unsafe.Pointer(&a))                     // want "embedded pointer"

	var st struct{ f []int }
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&st))) // want "embedded pointer"
	C.f(unsafe.Pointer(&st))                     // want "embedded pointer"

	// The following cases are OK.
	var i int
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&i)))
	C.f(unsafe.Pointer(&i))

	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&s[0])))
	C.f(unsafe.Pointer(&s[0]))

	var a2 [1]int
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&a2)))
	C.f(unsafe.Pointer(&a2))

	var st2 struct{ i int }
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&st2)))
	C.f(unsafe.Pointer(&st2))

	type cgoStruct struct{ p *cgoStruct }
	C.f(unsafe.Pointer(&cgoStruct{}))

	C.CBytes([]byte("hello"))
}

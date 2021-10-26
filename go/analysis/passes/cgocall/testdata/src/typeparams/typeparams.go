// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the cgo checker.

package a

// void f(void *ptr) {}
import "C"

import "unsafe"

func CgoTest[T any]() {
	var c chan bool
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&c))) // want "embedded pointer"
	C.f(unsafe.Pointer(&c))                     // want "embedded pointer"

	var schan S[chan bool]
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&schan))) // want "embedded pointer"
	C.f(unsafe.Pointer(&schan))                     // want "embedded pointer"

	var x T
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&x))) // no findings as T is not known compile-time
	C.f(unsafe.Pointer(&x))

	// instantiating CgoTest should not yield any warnings
	CgoTest[chan bool]()

	var sint S[int]
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&sint)))
	C.f(unsafe.Pointer(&sint))
}

type S[X any] struct {
	val X
}

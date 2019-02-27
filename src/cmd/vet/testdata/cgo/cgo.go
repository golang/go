// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the cgo checker.

package testdata

// void f(void *p) {}
import "C"

import "unsafe"

func CgoTests() {
	var c chan bool
	C.f(*(*unsafe.Pointer)(unsafe.Pointer(&c))) // ERROR "embedded pointer"
	C.f(unsafe.Pointer(&c))                     // ERROR "embedded pointer"
}

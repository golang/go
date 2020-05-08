// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cgo shouldn't crash if there is an extra argument with a C reference.

package main

// void F(void* p) {};
import "C"

import "unsafe"

func F() {
	var i int
	C.F(unsafe.Pointer(&i), C.int(0)) // ERROR HERE
}

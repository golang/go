// run -gcflags=-d=checkptr

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

func main() {
	var x [2]uint64
	a := unsafe.Pointer(&x[1])

	b := a
	b = unsafe.Pointer(uintptr(b) + 2)
	b = unsafe.Pointer(uintptr(b) - 1)
	b = unsafe.Pointer(uintptr(b) &^ 1)

	if a != b {
		panic("pointer arithmetic failed")
	}
}

// errorcheck -0 -m -l

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for runtime/internal/atomic.

package escape

import (
	"runtime/internal/atomic"
	"unsafe"
)

// BAD: should be "leaking param content".
func Loadp(addr unsafe.Pointer) unsafe.Pointer { // ERROR "leaking param: addr"
	return atomic.Loadp(addr)
}

var ptr unsafe.Pointer

func Storep() {
	var x int // ERROR "moved to heap: x"
	atomic.StorepNoWB(unsafe.Pointer(&ptr), unsafe.Pointer(&x))
}

func Casp1() {
	// BAD: x doesn't need to be heap allocated
	var x int // ERROR "moved to heap: x"
	var y int // ERROR "moved to heap: y"
	atomic.Casp1(&ptr, unsafe.Pointer(&x), unsafe.Pointer(&y))
}

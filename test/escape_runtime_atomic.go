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

// BAD: should always be "leaking param: addr to result ~r1 level=1$".
func Loadp(addr unsafe.Pointer) unsafe.Pointer { // ERROR "leaking param: addr( to result ~r1 level=1)?$"
	return atomic.Loadp(addr)
}

var ptr unsafe.Pointer

func Storep() {
	var x int // ERROR "moved to heap: x"
	atomic.StorepNoWB(unsafe.Pointer(&ptr), unsafe.Pointer(&x))
}

func Casp1() {
	// BAD: should always be "does not escape"
	x := new(int) // ERROR "escapes to heap|does not escape"
	var y int     // ERROR "moved to heap: y"
	atomic.Casp1(&ptr, unsafe.Pointer(x), unsafe.Pointer(&y))
}

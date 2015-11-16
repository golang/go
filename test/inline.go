// errorcheck -0 -m

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that inlining is working.
// Compiles but does not run.

package foo

import "unsafe"

func add2(p *byte, n uintptr) *byte { // ERROR "can inline add2" "leaking param: p to result"
	return (*byte)(add1(unsafe.Pointer(p), n)) // ERROR "inlining call to add1"
}

func add1(p unsafe.Pointer, x uintptr) unsafe.Pointer { // ERROR "can inline add1" "leaking param: p to result"
	return unsafe.Pointer(uintptr(p) + x)
}

func f(x *byte) *byte { // ERROR "can inline f" "leaking param: x to result"
	return add2(x, 1) // ERROR "inlining call to add2" "inlining call to add1"
}

//go:noinline
func g(x int) int {
	return x + 1
}

func h(x int) int { // ERROR "can inline h"
	return x + 2
}

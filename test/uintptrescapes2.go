// errorcheck -0 -m -live

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis and liveness inferred for uintptrescapes functions.

package p

import (
	"unsafe"
)

//go:uintptrescapes
//go:noinline
func F1(a uintptr) {} // ERROR "escaping uintptr"

//go:uintptrescapes
//go:noinline
func F2(a ...uintptr) {} // ERROR "escaping ...uintptr" "a does not escape"

//go:uintptrescapes
//go:noinline
func F3(uintptr) {} // ERROR "escaping uintptr"

//go:uintptrescapes
//go:noinline
func F4(...uintptr) {} // ERROR "escaping ...uintptr"

func G() {
	var t int                        // ERROR "moved to heap"
	F1(uintptr(unsafe.Pointer(&t)))  // ERROR "live at call to F1: .?autotmp" "&t escapes to heap" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
	var t2 int                       // ERROR "moved to heap"
	F3(uintptr(unsafe.Pointer(&t2))) // ERROR "live at call to F3: .?autotmp" "&t2 escapes to heap"
}

func H() {
	var v int                                 // ERROR "moved to heap"
	F2(0, 1, uintptr(unsafe.Pointer(&v)), 2)  // ERROR "live at call to newobject: .?autotmp" "live at call to F2: .?autotmp" "escapes to heap" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
	var v2 int                                // ERROR "moved to heap"
	F4(0, 1, uintptr(unsafe.Pointer(&v2)), 2) // ERROR "live at call to newobject: .?autotmp" "live at call to F4: .?autotmp" "escapes to heap"
}

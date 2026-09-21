// errorcheck -0 -l -m -live

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis and liveness inferred for uintptrescapes functions.

package p

import (
	"unsafe"
)

//go:uintptrescapes
func F1(a uintptr) {} // ERROR "escaping uintptr"

//go:uintptrescapes
func F2(a ...uintptr) {} // ERROR "escaping ...uintptr"

//go:uintptrescapes
func F3(uintptr) {} // ERROR "escaping uintptr"

//go:uintptrescapes
func F4(...uintptr) {} // ERROR "escaping ...uintptr"

type T struct{}

//go:uintptrescapes
func (T) M1(a uintptr) {} // ERROR "escaping uintptr"

//go:uintptrescapes
func (T) M2(a ...uintptr) {} // ERROR "escaping ...uintptr"

func TestF1() {
	var t int                       // ERROR "moved to heap"
	F1(uintptr(unsafe.Pointer(&t))) // ERROR "live at call to F1: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func TestF3() {
	var t2 int                       // ERROR "moved to heap"
	F3(uintptr(unsafe.Pointer(&t2))) // ERROR "live at call to F3: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func TestM1() {
	var t T
	var v int                         // ERROR "moved to heap"
	t.M1(uintptr(unsafe.Pointer(&v))) // ERROR "live at call to T.M1: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func TestF2() {
	var v int                                // ERROR "moved to heap"
	F2(0, 1, uintptr(unsafe.Pointer(&v)), 2) // ERROR "live at call to (newobject|mallocgcSmallNoScanSC[0-9]+): .?autotmp" "live at call to F2: .?autotmp" "escapes to heap" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func TestF4() {
	var v2 int                                // ERROR "moved to heap"
	F4(0, 1, uintptr(unsafe.Pointer(&v2)), 2) // ERROR "live at call to (newobject|mallocgcSmallNoScanSC[0-9]+): .?autotmp" "live at call to F4: .?autotmp" "escapes to heap" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func TestM2() {
	var t T
	var v int                                  // ERROR "moved to heap"
	t.M2(0, 1, uintptr(unsafe.Pointer(&v)), 2) // ERROR "live at call to (newobject|mallocgcSmallNoScanSC[0-9]+): .?autotmp" "live at call to T.M2: .?autotmp"  "escapes to heap" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

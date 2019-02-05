// errorcheck -0 -m -live

// +build !windows,!js

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis and liveness inferred for syscall.Syscall-like functions.

package p

import (
	"syscall"
	"unsafe"
)

func f(uintptr) // ERROR "f assuming arg#1 is unsafe uintptr"

func g() { // ERROR "can inline g"
	var t int
	f(uintptr(unsafe.Pointer(&t))) // ERROR "live at call to f: .?autotmp" "g &t does not escape" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func h() { // ERROR "can inline h"
	var v int
	syscall.Syscall(0, 1, uintptr(unsafe.Pointer(&v)), 2) // ERROR "live at call to Syscall: .?autotmp" "h &v does not escape" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func i() { // ERROR "can inline i"
	var t int
	p := unsafe.Pointer(&t) // ERROR "i &t does not escape"
	f(uintptr(p))           // ERROR "live at call to f: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func j() { // ERROR "can inline j"
	var v int
	p := unsafe.Pointer(&v)              // ERROR "j &v does not escape"
	syscall.Syscall(0, 1, uintptr(p), 2) // ERROR "live at call to Syscall: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

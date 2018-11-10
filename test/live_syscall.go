// errorcheck -0 -m -live

// +build !windows

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

func g() {
	var t int
	f(uintptr(unsafe.Pointer(&t))) // ERROR "live at call to f: .?autotmp" "g &t does not escape"
}

func h() {
	var v int
	syscall.Syscall(0, 1, uintptr(unsafe.Pointer(&v)), 2) // ERROR "live at call to Syscall: .?autotmp" "h &v does not escape"
}

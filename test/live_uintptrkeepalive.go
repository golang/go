// errorcheck -0 -m -live -std

//go:build !windows && !js && !wasip1

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis and liveness inferred for uintptrkeepalive functions.
//
// This behavior is enabled automatically for function declarations with no
// bodies (assembly, linkname), as well as explicitly on complete functions
// with //go:uintptrkeepalive.
//
// This is most important for syscall.Syscall (and similar functions), so we
// test it explicitly.

package p

import (
	"syscall"
	"unsafe"
)

func implicit(uintptr) // ERROR "assuming ~p0 is unsafe uintptr"

//go:uintptrkeepalive
//go:nosplit
func explicit(uintptr) {
}

func autotmpImplicit() { // ERROR "can inline autotmpImplicit"
	var t int
	implicit(uintptr(unsafe.Pointer(&t))) // ERROR "live at call to implicit: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func autotmpExplicit() { // ERROR "can inline autotmpExplicit"
	var t int
	explicit(uintptr(unsafe.Pointer(&t))) // ERROR "live at call to explicit: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func autotmpSyscall() { // ERROR "can inline autotmpSyscall"
	var v int
	syscall.Syscall(0, 1, uintptr(unsafe.Pointer(&v)), 2) // ERROR "live at call to Syscall: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func localImplicit() { // ERROR "can inline localImplicit"
	var t int
	p := unsafe.Pointer(&t)
	implicit(uintptr(p)) // ERROR "live at call to implicit: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func localExplicit() { // ERROR "can inline localExplicit"
	var t int
	p := unsafe.Pointer(&t)
	explicit(uintptr(p)) // ERROR "live at call to explicit: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

func localSyscall() { // ERROR "can inline localSyscall"
	var v int
	p := unsafe.Pointer(&v)
	syscall.Syscall(0, 1, uintptr(p), 2) // ERROR "live at call to Syscall: .?autotmp" "stack object .autotmp_[0-9]+ unsafe.Pointer$"
}

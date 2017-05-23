// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "syscall"

func init() {
	register("RaiseException", RaiseException)
	register("ZeroDivisionException", ZeroDivisionException)
}

func RaiseException() {
	const EXCEPTION_NONCONTINUABLE = 1
	mod := syscall.MustLoadDLL("kernel32.dll")
	proc := mod.MustFindProc("RaiseException")
	proc.Call(0xbad, EXCEPTION_NONCONTINUABLE, 0, 0)
	println("RaiseException should not return")
}

func ZeroDivisionException() {
	x := 1
	y := 0
	z := x / y
	println(z)
}

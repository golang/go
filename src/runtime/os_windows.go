// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type stdFunction *byte

func stdcall0(fn stdFunction) uintptr
func stdcall1(fn stdFunction, a0 uintptr) uintptr
func stdcall2(fn stdFunction, a0, a1 uintptr) uintptr
func stdcall3(fn stdFunction, a0, a1, a2 uintptr) uintptr
func stdcall4(fn stdFunction, a0, a1, a2, a3 uintptr) uintptr
func stdcall5(fn stdFunction, a0, a1, a2, a3, a4 uintptr) uintptr
func stdcall6(fn stdFunction, a0, a1, a2, a3, a4, a5 uintptr) uintptr
func stdcall7(fn stdFunction, a0, a1, a2, a3, a4, a5, a6 uintptr) uintptr

func asmstdcall(fn unsafe.Pointer)
func getlasterror() uint32
func setlasterror(err uint32)
func usleep1(usec uint32)
func netpollinit()
func netpollopen(fd uintptr, pd *pollDesc) int32
func netpollclose(fd uintptr) int32
func netpollarm(pd *pollDesc, mode int)

func os_sigpipe() {
	gothrow("too many writes on closed pipe")
}

func sigpanic() {
	g := getg()
	if !canpanic(g) {
		gothrow("unexpected signal during runtime execution")
	}

	switch uint32(g.sig) {
	case _EXCEPTION_ACCESS_VIOLATION:
		if g.sigcode1 < 0x1000 || g.paniconfault {
			panicmem()
		}
		print("unexpected fault address ", hex(g.sigcode1), "\n")
		gothrow("fault")
	case _EXCEPTION_INT_DIVIDE_BY_ZERO:
		panicdivide()
	case _EXCEPTION_INT_OVERFLOW:
		panicoverflow()
	case _EXCEPTION_FLT_DENORMAL_OPERAND,
		_EXCEPTION_FLT_DIVIDE_BY_ZERO,
		_EXCEPTION_FLT_INEXACT_RESULT,
		_EXCEPTION_FLT_OVERFLOW,
		_EXCEPTION_FLT_UNDERFLOW:
		panicfloat()
	}
	gothrow("fault")
}

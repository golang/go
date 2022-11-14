// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && (darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris)

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

func dumpregs(c *sigctxt) {
	print("rax    ", hex(c.rax()), "\n")
	print("rbx    ", hex(c.rbx()), "\n")
	print("rcx    ", hex(c.rcx()), "\n")
	print("rdx    ", hex(c.rdx()), "\n")
	print("rdi    ", hex(c.rdi()), "\n")
	print("rsi    ", hex(c.rsi()), "\n")
	print("rbp    ", hex(c.rbp()), "\n")
	print("rsp    ", hex(c.rsp()), "\n")
	print("r8     ", hex(c.r8()), "\n")
	print("r9     ", hex(c.r9()), "\n")
	print("r10    ", hex(c.r10()), "\n")
	print("r11    ", hex(c.r11()), "\n")
	print("r12    ", hex(c.r12()), "\n")
	print("r13    ", hex(c.r13()), "\n")
	print("r14    ", hex(c.r14()), "\n")
	print("r15    ", hex(c.r15()), "\n")
	print("rip    ", hex(c.rip()), "\n")
	print("rflags ", hex(c.rflags()), "\n")
	print("cs     ", hex(c.cs()), "\n")
	print("fs     ", hex(c.fs()), "\n")
	print("gs     ", hex(c.gs()), "\n")
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) sigpc() uintptr { return uintptr(c.rip()) }

func (c *sigctxt) setsigpc(x uint64) { c.set_rip(x) }
func (c *sigctxt) sigsp() uintptr    { return uintptr(c.rsp()) }
func (c *sigctxt) siglr() uintptr    { return 0 }
func (c *sigctxt) fault() uintptr    { return uintptr(c.sigaddr()) }

// preparePanic sets up the stack to look like a call to sigpanic.
func (c *sigctxt) preparePanic(sig uint32, gp *g) {
	// Work around Leopard bug that doesn't set FPE_INTDIV.
	// Look at instruction to see if it is a divide.
	// Not necessary in Snow Leopard (si_code will be != 0).
	if GOOS == "darwin" && sig == _SIGFPE && gp.sigcode0 == 0 {
		pc := (*[4]byte)(unsafe.Pointer(gp.sigpc))
		i := 0
		if pc[i]&0xF0 == 0x40 { // 64-bit REX prefix
			i++
		} else if pc[i] == 0x66 { // 16-bit instruction prefix
			i++
		}
		if pc[i] == 0xF6 || pc[i] == 0xF7 {
			gp.sigcode0 = _FPE_INTDIV
		}
	}

	pc := uintptr(c.rip())
	sp := uintptr(c.rsp())

	// In case we are panicking from external code, we need to initialize
	// Go special registers. We inject sigpanic0 (instead of sigpanic),
	// which takes care of that.
	if shouldPushSigpanic(gp, pc, *(*uintptr)(unsafe.Pointer(sp))) {
		c.pushCall(abi.FuncPCABI0(sigpanic0), pc)
	} else {
		// Not safe to push the call. Just clobber the frame.
		c.set_rip(uint64(abi.FuncPCABI0(sigpanic0)))
	}
}

func (c *sigctxt) pushCall(targetPC, resumePC uintptr) {
	// Make it look like we called target at resumePC.
	sp := uintptr(c.rsp())
	sp -= goarch.PtrSize
	*(*uintptr)(unsafe.Pointer(sp)) = resumePC
	c.set_rsp(uint64(sp))
	c.set_rip(uint64(targetPC))
}

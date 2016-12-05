// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 amd64p32
// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package runtime

import (
	"runtime/internal/sys"
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

func (c *sigctxt) sigsp() uintptr { return uintptr(c.rsp()) }
func (c *sigctxt) siglr() uintptr { return 0 }
func (c *sigctxt) fault() uintptr { return uintptr(c.sigaddr()) }

// preparePanic sets up the stack to look like a call to sigpanic.
func (c *sigctxt) preparePanic(sig uint32, gp *g) {
	if GOOS == "darwin" {
		// Work around Leopard bug that doesn't set FPE_INTDIV.
		// Look at instruction to see if it is a divide.
		// Not necessary in Snow Leopard (si_code will be != 0).
		if sig == _SIGFPE && gp.sigcode0 == 0 {
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
	}

	pc := uintptr(c.rip())
	sp := uintptr(c.rsp())

	// If we don't recognize the PC as code
	// but we do recognize the top pointer on the stack as code,
	// then assume this was a call to non-code and treat like
	// pc == 0, to make unwinding show the context.
	if pc != 0 && findfunc(pc) == nil && findfunc(*(*uintptr)(unsafe.Pointer(sp))) != nil {
		pc = 0
	}

	// Only push runtime.sigpanic if pc != 0.
	// If pc == 0, probably panicked because of a
	// call to a nil func. Not pushing that onto sp will
	// make the trace look like a call to runtime.sigpanic instead.
	// (Otherwise the trace will end at runtime.sigpanic and we
	// won't get to see who faulted.)
	if pc != 0 {
		if sys.RegSize > sys.PtrSize {
			sp -= sys.PtrSize
			*(*uintptr)(unsafe.Pointer(sp)) = 0
		}
		sp -= sys.PtrSize
		*(*uintptr)(unsafe.Pointer(sp)) = pc
		c.set_rsp(uint64(sp))
	}
	c.set_rip(uint64(funcPC(sigpanic)))
}

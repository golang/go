// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

func dumpregs(c *sigctxt) {
	print("eax    ", hex(c.eax()), "\n")
	print("ebx    ", hex(c.ebx()), "\n")
	print("ecx    ", hex(c.ecx()), "\n")
	print("edx    ", hex(c.edx()), "\n")
	print("edi    ", hex(c.edi()), "\n")
	print("esi    ", hex(c.esi()), "\n")
	print("ebp    ", hex(c.ebp()), "\n")
	print("esp    ", hex(c.esp()), "\n")
	print("eip    ", hex(c.eip()), "\n")
	print("eflags ", hex(c.eflags()), "\n")
	print("cs     ", hex(c.cs()), "\n")
	print("fs     ", hex(c.fs()), "\n")
	print("gs     ", hex(c.gs()), "\n")
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) sigpc() uintptr { return uintptr(c.eip()) }

func (c *sigctxt) sigsp() uintptr { return uintptr(c.esp()) }
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
			if pc[i] == 0x66 { // 16-bit instruction prefix
				i++
			}
			if pc[i] == 0xF6 || pc[i] == 0xF7 {
				gp.sigcode0 = _FPE_INTDIV
			}
		}
	}

	pc := uintptr(c.eip())
	sp := uintptr(c.esp())

	if shouldPushSigpanic(gp, pc, *(*uintptr)(unsafe.Pointer(sp))) {
		c.pushCall(funcPC(sigpanic))
	} else {
		// Not safe to push the call. Just clobber the frame.
		c.set_eip(uint32(funcPC(sigpanic)))
	}
}

const pushCallSupported = true

func (c *sigctxt) pushCall(targetPC uintptr) {
	// Make it look like the signaled instruction called target.
	pc := uintptr(c.eip())
	sp := uintptr(c.esp())
	sp -= sys.PtrSize
	*(*uintptr)(unsafe.Pointer(sp)) = pc
	c.set_esp(uint32(sp))
	c.set_eip(uint32(targetPC))
}

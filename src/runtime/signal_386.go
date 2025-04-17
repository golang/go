// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd

package runtime

import (
	"internal/abi"
	"internal/goarch"
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
	pc := uintptr(c.eip())
	sp := uintptr(c.esp())

	if shouldPushSigpanic(gp, pc, *(*uintptr)(unsafe.Pointer(sp))) {
		c.pushCall(abi.FuncPCABIInternal(sigpanic), pc)
	} else {
		// Not safe to push the call. Just clobber the frame.
		c.set_eip(uint32(abi.FuncPCABIInternal(sigpanic)))
	}
}

func (c *sigctxt) pushCall(targetPC, resumePC uintptr) {
	// Make it look like we called target at resumePC.
	sp := uintptr(c.esp())
	sp -= goarch.PtrSize
	*(*uintptr)(unsafe.Pointer(sp)) = resumePC
	c.set_esp(uint32(sp))
	c.set_eip(uint32(targetPC))
}

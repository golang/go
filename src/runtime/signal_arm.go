// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd

package runtime

import "unsafe"

func dumpregs(c *sigctxt) {
	print("trap    ", hex(c.trap()), "\n")
	print("error   ", hex(c.error()), "\n")
	print("oldmask ", hex(c.oldmask()), "\n")
	print("r0      ", hex(c.r0()), "\n")
	print("r1      ", hex(c.r1()), "\n")
	print("r2      ", hex(c.r2()), "\n")
	print("r3      ", hex(c.r3()), "\n")
	print("r4      ", hex(c.r4()), "\n")
	print("r5      ", hex(c.r5()), "\n")
	print("r6      ", hex(c.r6()), "\n")
	print("r7      ", hex(c.r7()), "\n")
	print("r8      ", hex(c.r8()), "\n")
	print("r9      ", hex(c.r9()), "\n")
	print("r10     ", hex(c.r10()), "\n")
	print("fp      ", hex(c.fp()), "\n")
	print("ip      ", hex(c.ip()), "\n")
	print("sp      ", hex(c.sp()), "\n")
	print("lr      ", hex(c.lr()), "\n")
	print("pc      ", hex(c.pc()), "\n")
	print("cpsr    ", hex(c.cpsr()), "\n")
	print("fault   ", hex(c.fault()), "\n")
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) sigpc() uintptr { return uintptr(c.pc()) }

func (c *sigctxt) sigsp() uintptr { return uintptr(c.sp()) }
func (c *sigctxt) siglr() uintptr { return uintptr(c.lr()) }

// preparePanic sets up the stack to look like a call to sigpanic.
func (c *sigctxt) preparePanic(sig uint32, gp *g) {
	// We arrange lr, and pc to pretend the panicking
	// function calls sigpanic directly.
	// Always save LR to stack so that panics in leaf
	// functions are correctly handled. This smashes
	// the stack frame but we're not going back there
	// anyway.
	sp := c.sp() - 4
	c.set_sp(sp)
	*(*uint32)(unsafe.Pointer(uintptr(sp))) = c.lr()

	pc := gp.sigpc

	// If we don't recognize the PC as code
	// but we do recognize the link register as code,
	// then assume this was a call to non-code and treat like
	// pc == 0, to make unwinding show the context.
	if pc != 0 && findfunc(pc) == nil && findfunc(uintptr(c.lr())) != nil {
		pc = 0
	}

	// Don't bother saving PC if it's zero, which is
	// probably a call to a nil func: the old link register
	// is more useful in the stack trace.
	if pc != 0 {
		c.set_lr(uint32(pc))
	}

	// In case we are panicking from external C code
	c.set_r10(uint32(uintptr(unsafe.Pointer(gp))))
	c.set_pc(uint32(funcPC(sigpanic)))
}

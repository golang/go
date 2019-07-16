// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin linux netbsd openbsd

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

func dumpregs(c *sigctxt) {
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
	print("r11     ", hex(c.r11()), "\n")
	print("r12     ", hex(c.r12()), "\n")
	print("r13     ", hex(c.r13()), "\n")
	print("r14     ", hex(c.r14()), "\n")
	print("r15     ", hex(c.r15()), "\n")
	print("r16     ", hex(c.r16()), "\n")
	print("r17     ", hex(c.r17()), "\n")
	print("r18     ", hex(c.r18()), "\n")
	print("r19     ", hex(c.r19()), "\n")
	print("r20     ", hex(c.r20()), "\n")
	print("r21     ", hex(c.r21()), "\n")
	print("r22     ", hex(c.r22()), "\n")
	print("r23     ", hex(c.r23()), "\n")
	print("r24     ", hex(c.r24()), "\n")
	print("r25     ", hex(c.r25()), "\n")
	print("r26     ", hex(c.r26()), "\n")
	print("r27     ", hex(c.r27()), "\n")
	print("r28     ", hex(c.r28()), "\n")
	print("r29     ", hex(c.r29()), "\n")
	print("lr      ", hex(c.lr()), "\n")
	print("sp      ", hex(c.sp()), "\n")
	print("pc      ", hex(c.pc()), "\n")
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
	sp := c.sp() - sys.SpAlign // needs only sizeof uint64, but must align the stack
	c.set_sp(sp)
	*(*uint64)(unsafe.Pointer(uintptr(sp))) = c.lr()

	pc := gp.sigpc

	if shouldPushSigpanic(gp, pc, uintptr(c.lr())) {
		// Make it look the like faulting PC called sigpanic.
		c.set_lr(uint64(pc))
	}

	// In case we are panicking from external C code
	c.set_r28(uint64(uintptr(unsafe.Pointer(gp))))
	c.set_pc(uint64(funcPC(sigpanic)))
}

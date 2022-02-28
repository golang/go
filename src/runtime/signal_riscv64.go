// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && riscv64

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

func dumpregs(c *sigctxt) {
	print("ra  ", hex(c.ra()), "\t")
	print("sp  ", hex(c.sp()), "\n")
	print("gp  ", hex(c.gp()), "\t")
	print("tp  ", hex(c.tp()), "\n")
	print("t0  ", hex(c.t0()), "\t")
	print("t1  ", hex(c.t1()), "\n")
	print("t2  ", hex(c.t2()), "\t")
	print("s0  ", hex(c.s0()), "\n")
	print("s1  ", hex(c.s1()), "\t")
	print("a0  ", hex(c.a0()), "\n")
	print("a1  ", hex(c.a1()), "\t")
	print("a2  ", hex(c.a2()), "\n")
	print("a3  ", hex(c.a3()), "\t")
	print("a4  ", hex(c.a4()), "\n")
	print("a5  ", hex(c.a5()), "\t")
	print("a6  ", hex(c.a6()), "\n")
	print("a7  ", hex(c.a7()), "\t")
	print("s2  ", hex(c.s2()), "\n")
	print("s3  ", hex(c.s3()), "\t")
	print("s4  ", hex(c.s4()), "\n")
	print("s5  ", hex(c.s5()), "\t")
	print("s6  ", hex(c.s6()), "\n")
	print("s7  ", hex(c.s7()), "\t")
	print("s8  ", hex(c.s8()), "\n")
	print("s9  ", hex(c.s9()), "\t")
	print("s10 ", hex(c.s10()), "\n")
	print("s11 ", hex(c.s11()), "\t")
	print("t3  ", hex(c.t3()), "\n")
	print("t4  ", hex(c.t4()), "\t")
	print("t5  ", hex(c.t5()), "\n")
	print("t6  ", hex(c.t6()), "\t")
	print("pc  ", hex(c.pc()), "\n")
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) sigpc() uintptr { return uintptr(c.pc()) }

func (c *sigctxt) sigsp() uintptr { return uintptr(c.sp()) }
func (c *sigctxt) siglr() uintptr { return uintptr(c.ra()) }
func (c *sigctxt) fault() uintptr { return uintptr(c.sigaddr()) }

// preparePanic sets up the stack to look like a call to sigpanic.
func (c *sigctxt) preparePanic(sig uint32, gp *g) {
	// We arrange RA, and pc to pretend the panicking
	// function calls sigpanic directly.
	// Always save RA to stack so that panics in leaf
	// functions are correctly handled. This smashes
	// the stack frame but we're not going back there
	// anyway.
	sp := c.sp() - goarch.PtrSize
	c.set_sp(sp)
	*(*uint64)(unsafe.Pointer(uintptr(sp))) = c.ra()

	pc := gp.sigpc

	if shouldPushSigpanic(gp, pc, uintptr(c.ra())) {
		// Make it look the like faulting PC called sigpanic.
		c.set_ra(uint64(pc))
	}

	// In case we are panicking from external C code
	c.set_gp(uint64(uintptr(unsafe.Pointer(gp))))
	c.set_pc(uint64(abi.FuncPCABIInternal(sigpanic)))
}

func (c *sigctxt) pushCall(targetPC, resumePC uintptr) {
	// Push the LR to stack, as we'll clobber it in order to
	// push the call. The function being pushed is responsible
	// for restoring the LR and setting the SP back.
	// This extra slot is known to gentraceback.
	sp := c.sp() - goarch.PtrSize
	c.set_sp(sp)
	*(*uint64)(unsafe.Pointer(uintptr(sp))) = c.ra()
	// Set up PC and LR to pretend the function being signaled
	// calls targetPC at resumePC.
	c.set_ra(uint64(resumePC))
	c.set_pc(uint64(targetPC))
}

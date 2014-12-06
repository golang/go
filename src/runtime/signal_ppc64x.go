// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build ppc64 ppc64le

package runtime

import "unsafe"

func dumpregs(c *sigctxt) {
	print("r0   ", hex(c.r0()), "\t")
	print("r1   ", hex(c.r1()), "\n")
	print("r2   ", hex(c.r2()), "\t")
	print("r3   ", hex(c.r3()), "\n")
	print("r4   ", hex(c.r4()), "\t")
	print("r5   ", hex(c.r5()), "\n")
	print("r6   ", hex(c.r6()), "\t")
	print("r7   ", hex(c.r7()), "\n")
	print("r8   ", hex(c.r8()), "\t")
	print("r9   ", hex(c.r9()), "\n")
	print("r10  ", hex(c.r10()), "\t")
	print("r11  ", hex(c.r11()), "\n")
	print("r12  ", hex(c.r12()), "\t")
	print("r13  ", hex(c.r13()), "\n")
	print("r14  ", hex(c.r14()), "\t")
	print("r15  ", hex(c.r15()), "\n")
	print("r16  ", hex(c.r16()), "\t")
	print("r17  ", hex(c.r17()), "\n")
	print("r18  ", hex(c.r18()), "\t")
	print("r19  ", hex(c.r19()), "\n")
	print("r20  ", hex(c.r20()), "\t")
	print("r21  ", hex(c.r21()), "\n")
	print("r22  ", hex(c.r22()), "\t")
	print("r23  ", hex(c.r23()), "\n")
	print("r24  ", hex(c.r24()), "\t")
	print("r25  ", hex(c.r25()), "\n")
	print("r26  ", hex(c.r26()), "\t")
	print("r27  ", hex(c.r27()), "\n")
	print("r28  ", hex(c.r28()), "\t")
	print("r29  ", hex(c.r29()), "\n")
	print("r30  ", hex(c.r30()), "\t")
	print("r31  ", hex(c.r31()), "\n")
	print("pc   ", hex(c.pc()), "\t")
	print("ctr  ", hex(c.ctr()), "\n")
	print("link ", hex(c.link()), "\t")
	print("xer  ", hex(c.xer()), "\n")
	print("ccr  ", hex(c.ccr()), "\t")
	print("trap ", hex(c.trap()), "\n")
}

func sighandler(sig uint32, info *siginfo, ctxt unsafe.Pointer, gp *g) {
	_g_ := getg()
	c := &sigctxt{info, ctxt}

	if sig == _SIGPROF {
		sigprof((*byte)(unsafe.Pointer(uintptr(c.pc()))), (*byte)(unsafe.Pointer(uintptr(c.sp()))), (*byte)(unsafe.Pointer(uintptr(c.link()))), gp, _g_.m)
		return
	}
	flags := int32(_SigThrow)
	if sig < uint32(len(sigtable)) {
		flags = sigtable[sig].flags
	}
	if c.sigcode() != _SI_USER && flags&_SigPanic != 0 {
		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp.sig = sig
		gp.sigcode0 = uintptr(c.sigcode())
		gp.sigcode1 = uintptr(c.fault())
		gp.sigpc = uintptr(c.pc())

		// We arrange link, and pc to pretend the panicking
		// function calls sigpanic directly.
		// Always save LINK to stack so that panics in leaf
		// functions are correctly handled. This smashes
		// the stack frame but we're not going back there
		// anyway.
		sp := c.sp() - ptrSize
		c.set_sp(sp)
		*(*uint64)(unsafe.Pointer(uintptr(sp))) = c.link()

		// Don't bother saving PC if it's zero, which is
		// probably a call to a nil func: the old link register
		// is more useful in the stack trace.
		if gp.sigpc != 0 {
			c.set_link(uint64(gp.sigpc))
		}

		// In case we are panicking from external C code
		c.set_r0(0)
		c.set_r30(uint64(uintptr(unsafe.Pointer(gp))))
		c.set_pc(uint64(funcPC(sigpanic)))
		return
	}

	if c.sigcode() == _SI_USER || flags&_SigNotify != 0 {
		if sigsend(sig) {
			return
		}
	}

	if flags&_SigKill != 0 {
		exit(2)
	}

	if flags&_SigThrow == 0 {
		return
	}

	_g_.m.throwing = 1
	_g_.m.caughtsig = gp
	startpanic()

	if sig < uint32(len(sigtable)) {
		print(sigtable[sig].name, "\n")
	} else {
		print("Signal ", sig, "\n")
	}

	print("PC=", hex(c.pc()), "\n")
	if _g_.m.lockedg != nil && _g_.m.ncgo > 0 && gp == _g_.m.g0 {
		print("signal arrived during cgo execution\n")
		gp = _g_.m.lockedg
	}
	print("\n")

	var docrash bool
	if gotraceback(&docrash) > 0 {
		goroutineheader(gp)
		tracebacktrap(uintptr(c.pc()), uintptr(c.sp()), uintptr(c.link()), gp)
		tracebackothers(gp)
		print("\n")
		dumpregs(c)
	}

	if docrash {
		crash()
	}

	exit(2)
}

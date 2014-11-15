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

func sighandler(sig uint32, info *siginfo, ctxt unsafe.Pointer, gp *g) {
	_g_ := getg()
	c := &sigctxt{info, ctxt}

	if sig == _SIGPROF {
		sigprof((*byte)(unsafe.Pointer(uintptr(c.pc()))), (*byte)(unsafe.Pointer(uintptr(c.sp()))), (*byte)(unsafe.Pointer(uintptr(c.lr()))), gp, _g_.m)
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

		// We arrange lr, and pc to pretend the panicking
		// function calls sigpanic directly.
		// Always save LR to stack so that panics in leaf
		// functions are correctly handled. This smashes
		// the stack frame but we're not going back there
		// anyway.
		sp := c.sp() - 4
		c.set_sp(sp)
		*(*uint32)(unsafe.Pointer(uintptr(sp))) = c.lr()

		// Don't bother saving PC if it's zero, which is
		// probably a call to a nil func: the old link register
		// is more useful in the stack trace.
		if gp.sigpc != 0 {
			c.set_lr(uint32(gp.sigpc))
		}

		// In case we are panicking from external C code
		c.set_r10(uint32(uintptr(unsafe.Pointer(gp))))
		c.set_pc(uint32(funcPC(sigpanic)))
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
		tracebacktrap(uintptr(c.pc()), uintptr(c.sp()), uintptr(c.lr()), gp)
		tracebackothers(gp)
		print("\n")
		dumpregs(c)
	}

	if docrash {
		crash()
	}

	exit(2)
}

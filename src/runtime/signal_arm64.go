// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux darwin

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

var crashing int32

// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func sighandler(sig uint32, info *siginfo, ctxt unsafe.Pointer, gp *g) {
	_g_ := getg()
	c := &sigctxt{info, ctxt}

	if sig == _SIGPROF {
		sigprof(uintptr(c.pc()), uintptr(c.sp()), uintptr(c.lr()), gp, _g_.m)
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
		sp := c.sp() - sys.SpAlign // needs only sizeof uint64, but must align the stack
		c.set_sp(sp)
		*(*uint64)(unsafe.Pointer(uintptr(sp))) = c.lr()

		pc := uintptr(gp.sigpc)

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
			c.set_lr(uint64(pc))
		}

		// In case we are panicking from external C code
		c.set_r28(uint64(uintptr(unsafe.Pointer(gp))))
		c.set_pc(uint64(funcPC(sigpanic)))
		return
	}

	if c.sigcode() == _SI_USER || flags&_SigNotify != 0 {
		if sigsend(sig) {
			return
		}
	}

	if c.sigcode() == _SI_USER && signal_ignored(sig) {
		return
	}

	if flags&_SigKill != 0 {
		dieFromSignal(int32(sig))
	}

	if flags&_SigThrow == 0 {
		return
	}

	_g_.m.throwing = 1
	_g_.m.caughtsig.set(gp)

	if crashing == 0 {
		startpanic()
	}

	if sig < uint32(len(sigtable)) {
		print(sigtable[sig].name, "\n")
	} else {
		print("Signal ", sig, "\n")
	}

	print("PC=", hex(c.pc()), " m=", _g_.m.id, "\n")
	if _g_.m.lockedg != nil && _g_.m.ncgo > 0 && gp == _g_.m.g0 {
		print("signal arrived during cgo execution\n")
		gp = _g_.m.lockedg
	}
	print("\n")

	level, _, docrash := gotraceback()
	if level > 0 {
		goroutineheader(gp)
		tracebacktrap(uintptr(c.pc()), uintptr(c.sp()), uintptr(c.lr()), gp)
		if crashing > 0 && gp != _g_.m.curg && _g_.m.curg != nil && readgstatus(_g_.m.curg)&^_Gscan == _Grunning {
			// tracebackothers on original m skipped this one; trace it now.
			goroutineheader(_g_.m.curg)
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
		} else if crashing == 0 {
			tracebackothers(gp)
			print("\n")
		}
		dumpregs(c)
	}

	if docrash {
		crashing++
		if crashing < sched.mcount {
			// There are other m's that need to dump their stacks.
			// Relay SIGQUIT to the next m by sending it to the current process.
			// All m's that have already received SIGQUIT have signal masks blocking
			// receipt of any signals, so the SIGQUIT will go to an m that hasn't seen it yet.
			// When the last m receives the SIGQUIT, it will fall through to the call to
			// crash below. Just in case the relaying gets botched, each m involved in
			// the relay sleeps for 5 seconds and then does the crash/exit itself.
			// In expected operation, the last m has received the SIGQUIT and run
			// crash/exit and the process is gone, all long before any of the
			// 5-second sleeps have finished.
			print("\n-----\n\n")
			raiseproc(_SIGQUIT)
			usleep(5 * 1000 * 1000)
		}
		crash()
	}

	exit(2)
}

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

type sigctxt struct {
	info *siginfo
	ctxt unsafe.Pointer
}

func (c *sigctxt) regs() *sigcontext {
	return (*sigcontext)(unsafe.Pointer(&(*ucontext)(c.ctxt).uc_mcontext))
}
func (c *sigctxt) r0() uint64      { return c.regs().gregs[0] }
func (c *sigctxt) r1() uint64      { return c.regs().gregs[1] }
func (c *sigctxt) r2() uint64      { return c.regs().gregs[2] }
func (c *sigctxt) r3() uint64      { return c.regs().gregs[3] }
func (c *sigctxt) r4() uint64      { return c.regs().gregs[4] }
func (c *sigctxt) r5() uint64      { return c.regs().gregs[5] }
func (c *sigctxt) r6() uint64      { return c.regs().gregs[6] }
func (c *sigctxt) r7() uint64      { return c.regs().gregs[7] }
func (c *sigctxt) r8() uint64      { return c.regs().gregs[8] }
func (c *sigctxt) r9() uint64      { return c.regs().gregs[9] }
func (c *sigctxt) r10() uint64     { return c.regs().gregs[10] }
func (c *sigctxt) r11() uint64     { return c.regs().gregs[11] }
func (c *sigctxt) r12() uint64     { return c.regs().gregs[12] }
func (c *sigctxt) r13() uint64     { return c.regs().gregs[13] }
func (c *sigctxt) r14() uint64     { return c.regs().gregs[14] }
func (c *sigctxt) r15() uint64     { return c.regs().gregs[15] }
func (c *sigctxt) link() uint64    { return c.regs().gregs[14] }
func (c *sigctxt) sp() uint64      { return c.regs().gregs[15] }
func (c *sigctxt) pc() uint64      { return c.regs().psw_addr }
func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 { return c.info.si_addr }

func (c *sigctxt) set_r0(x uint64)      { c.regs().gregs[0] = x }
func (c *sigctxt) set_r13(x uint64)     { c.regs().gregs[13] = x }
func (c *sigctxt) set_link(x uint64)    { c.regs().gregs[14] = x }
func (c *sigctxt) set_sp(x uint64)      { c.regs().gregs[15] = x }
func (c *sigctxt) set_pc(x uint64)      { c.regs().psw_addr = x }
func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	*(*uintptr)(add(unsafe.Pointer(c.info), 2*sys.PtrSize)) = uintptr(x)
}

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
	print("pc   ", hex(c.pc()), "\t")
	print("link ", hex(c.link()), "\n")
}

var crashing int32

// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func sighandler(sig uint32, info *siginfo, ctxt unsafe.Pointer, gp *g) {
	_g_ := getg()
	c := &sigctxt{info, ctxt}

	if sig == _SIGPROF {
		sigprof(uintptr(c.pc()), uintptr(c.sp()), uintptr(c.link()), gp, _g_.m)
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
		gp.sigcode1 = uintptr(c.sigaddr())
		gp.sigpc = uintptr(c.pc())

		// We arrange link, and pc to pretend the panicking
		// function calls sigpanic directly.
		// Always save LINK to stack so that panics in leaf
		// functions are correctly handled. This smashes
		// the stack frame but we're not going back there
		// anyway.
		sp := c.sp() - sys.MinFrameSize
		c.set_sp(sp)
		*(*uint64)(unsafe.Pointer(uintptr(sp))) = c.link()

		pc := uintptr(gp.sigpc)

		// If we don't recognize the PC as code
		// but we do recognize the link register as code,
		// then assume this was a call to non-code and treat like
		// pc == 0, to make unwinding show the context.
		if pc != 0 && findfunc(pc) == nil && findfunc(uintptr(c.link())) != nil {
			pc = 0
		}

		// Don't bother saving PC if it's zero, which is
		// probably a call to a nil func: the old link register
		// is more useful in the stack trace.
		if pc != 0 {
			c.set_link(uint64(pc))
		}

		// In case we are panicking from external C code
		c.set_r0(0)
		c.set_r13(uint64(uintptr(unsafe.Pointer(gp))))
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
		tracebacktrap(uintptr(c.pc()), uintptr(c.sp()), uintptr(c.link()), gp)
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

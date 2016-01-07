// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd

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

var crashing int32

// May run during STW, so write barriers are not allowed.
//
//go:nowritebarrierrec
func sighandler(sig uint32, info *siginfo, ctxt unsafe.Pointer, gp *g) {
	_g_ := getg()
	c := &sigctxt{info, ctxt}

	if sig == _SIGPROF {
		sigprof(uintptr(c.eip()), uintptr(c.esp()), 0, gp, _g_.m)
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
		gp.sigpc = uintptr(c.eip())

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

		// If we don't recognize the PC as code
		// but we do recognize the top pointer on the stack as code,
		// then assume this was a call to non-code and treat like
		// pc == 0, to make unwinding show the context.
		if pc != 0 && findfunc(pc) == nil && findfunc(*(*uintptr)(unsafe.Pointer(sp))) != nil {
			pc = 0
		}

		// Only push runtime.sigpanic if pc != 0.
		// If pc == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
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
			c.set_esp(uint32(sp))
		}
		c.set_eip(uint32(funcPC(sigpanic)))
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

	print("PC=", hex(c.eip()), " m=", _g_.m.id, "\n")
	if _g_.m.lockedg != nil && _g_.m.ncgo > 0 && gp == _g_.m.g0 {
		print("signal arrived during cgo execution\n")
		gp = _g_.m.lockedg
	}
	print("\n")

	level, _, docrash := gotraceback()
	if level > 0 {
		goroutineheader(gp)

		// On Linux/386, all system calls go through the vdso kernel_vsyscall routine.
		// Normally we don't see those PCs, but during signals we can.
		// If we see a PC in the vsyscall area (it moves around, but near the top of memory),
		// assume we're blocked in the vsyscall routine, which has saved
		// three words on the stack after the initial call saved the caller PC.
		// Pop all four words off SP and use the saved PC.
		// The check of the stack bounds here should suffice to avoid a fault
		// during the actual PC pop.
		// If we do load a bogus PC, not much harm done: we weren't going
		// to get a decent traceback anyway.
		// TODO(rsc): Make this more precise: we should do more checks on the PC,
		// and we should find out whether different versions of the vdso page
		// use different prologues that store different amounts on the stack.
		pc := uintptr(c.eip())
		sp := uintptr(c.esp())
		if GOOS == "linux" && pc >= 0xf4000000 && gp.stack.lo <= sp && sp+16 <= gp.stack.hi {
			// Assume in vsyscall page.
			sp += 16
			pc = *(*uintptr)(unsafe.Pointer(sp - 4))
			print("runtime: unwind vdso kernel_vsyscall: pc=", hex(pc), " sp=", hex(sp), "\n")
		}

		tracebacktrap(pc, sp, 0, gp)
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

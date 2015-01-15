// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 amd64p32
// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package runtime

import (
	"unsafe"
)

func dumpregs(c *sigctxt) {
	print("rax    ", hex(c.rax()), "\n")
	print("rbx    ", hex(c.rbx()), "\n")
	print("rcx    ", hex(c.rcx()), "\n")
	print("rdx    ", hex(c.rdx()), "\n")
	print("rdi    ", hex(c.rdi()), "\n")
	print("rsi    ", hex(c.rsi()), "\n")
	print("rbp    ", hex(c.rbp()), "\n")
	print("rsp    ", hex(c.rsp()), "\n")
	print("r8     ", hex(c.r8()), "\n")
	print("r9     ", hex(c.r9()), "\n")
	print("r10    ", hex(c.r10()), "\n")
	print("r11    ", hex(c.r11()), "\n")
	print("r12    ", hex(c.r12()), "\n")
	print("r13    ", hex(c.r13()), "\n")
	print("r14    ", hex(c.r14()), "\n")
	print("r15    ", hex(c.r15()), "\n")
	print("rip    ", hex(c.rip()), "\n")
	print("rflags ", hex(c.rflags()), "\n")
	print("cs     ", hex(c.cs()), "\n")
	print("fs     ", hex(c.fs()), "\n")
	print("gs     ", hex(c.gs()), "\n")
}

var crashing int32

func sighandler(sig uint32, info *siginfo, ctxt unsafe.Pointer, gp *g) {
	_g_ := getg()
	c := &sigctxt{info, ctxt}

	if sig == _SIGPROF {
		sigprof((*byte)(unsafe.Pointer(uintptr(c.rip()))), (*byte)(unsafe.Pointer(uintptr(c.rsp()))), nil, gp, _g_.m)
		return
	}

	if GOOS == "darwin" {
		// x86-64 has 48-bit virtual addresses. The top 16 bits must echo bit 47.
		// The hardware delivers a different kind of fault for a malformed address
		// than it does for an attempt to access a valid but unmapped address.
		// OS X 10.9.2 mishandles the malformed address case, making it look like
		// a user-generated signal (like someone ran kill -SEGV ourpid).
		// We pass user-generated signals to os/signal, or else ignore them.
		// Doing that here - and returning to the faulting code - results in an
		// infinite loop. It appears the best we can do is rewrite what the kernel
		// delivers into something more like the truth. The address used below
		// has very little chance of being the one that caused the fault, but it is
		// malformed, it is clearly not a real pointer, and if it does get printed
		// in real life, people will probably search for it and find this code.
		// There are no Google hits for b01dfacedebac1e or 0xb01dfacedebac1e
		// as I type this comment.
		if sig == _SIGSEGV && c.sigcode() == _SI_USER {
			c.set_sigcode(_SI_USER + 1)
			c.set_sigaddr(0xb01dfacedebac1e)
		}
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
		gp.sigpc = uintptr(c.rip())

		if GOOS == "darwin" {
			// Work around Leopard bug that doesn't set FPE_INTDIV.
			// Look at instruction to see if it is a divide.
			// Not necessary in Snow Leopard (si_code will be != 0).
			if sig == _SIGFPE && gp.sigcode0 == 0 {
				pc := (*[4]byte)(unsafe.Pointer(gp.sigpc))
				i := 0
				if pc[i]&0xF0 == 0x40 { // 64-bit REX prefix
					i++
				} else if pc[i] == 0x66 { // 16-bit instruction prefix
					i++
				}
				if pc[i] == 0xF6 || pc[i] == 0xF7 {
					gp.sigcode0 = _FPE_INTDIV
				}
			}
		}

		// Only push runtime.sigpanic if rip != 0.
		// If rip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime.sigpanic instead.
		// (Otherwise the trace will end at runtime.sigpanic and we
		// won't get to see who faulted.)
		if c.rip() != 0 {
			sp := c.rsp()
			if regSize > ptrSize {
				sp -= ptrSize
				*(*uintptr)(unsafe.Pointer(uintptr(sp))) = 0
			}
			sp -= ptrSize
			*(*uintptr)(unsafe.Pointer(uintptr(sp))) = uintptr(c.rip())
			c.set_rsp(sp)
		}
		c.set_rip(uint64(funcPC(sigpanic)))
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

	if crashing == 0 {
		startpanic()
	}

	if sig < uint32(len(sigtable)) {
		print(sigtable[sig].name, "\n")
	} else {
		print("Signal ", sig, "\n")
	}

	print("PC=", hex(c.rip()), " m=", _g_.m.id, "\n")
	if _g_.m.lockedg != nil && _g_.m.ncgo > 0 && gp == _g_.m.g0 {
		print("signal arrived during cgo execution\n")
		gp = _g_.m.lockedg
	}
	print("\n")

	var docrash bool
	if gotraceback(&docrash) > 0 {
		goroutineheader(gp)
		tracebacktrap(uintptr(c.rip()), uintptr(c.rsp()), 0, gp)
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
		// TODO(rsc): Implement raiseproc on other systems
		// and then add to this if condition.
		if GOOS == "darwin" || GOOS == "linux" {
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
		}
		crash()
	}

	exit(2)
}

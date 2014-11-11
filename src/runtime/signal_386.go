// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd

package runtime

import "unsafe"

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

func sighandler(sig uint32, info *siginfo, ctxt unsafe.Pointer, gp *g) {
	_g_ := getg()
	c := &sigctxt{info, ctxt}

	if sig == _SIGPROF {
		sigprof((*byte)(unsafe.Pointer(uintptr(c.eip()))), (*byte)(unsafe.Pointer(uintptr(c.esp()))), nil, gp, _g_.m)
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

		// Only push runtime.sigpanic if rip != 0.
		// If rip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime.sigpanic instead.
		// (Otherwise the trace will end at runtime.sigpanic and we
		// won't get to see who faulted.)
		if c.eip() != 0 {
			sp := c.esp()
			if regSize > ptrSize {
				sp -= ptrSize
				*(*uintptr)(unsafe.Pointer(uintptr(sp))) = 0
			}
			sp -= ptrSize
			*(*uintptr)(unsafe.Pointer(uintptr(sp))) = uintptr(c.eip())
			c.set_esp(sp)
		}
		c.set_eip(uint32(funcPC(sigpanic)))
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

	print("PC=", hex(c.eip()), "\n")
	if _g_.m.lockedg != nil && _g_.m.ncgo > 0 && gp == _g_.m.g0 {
		print("signal arrived during cgo execution\n")
		gp = _g_.m.lockedg
	}
	print("\n")

	var docrash bool
	if gotraceback(&docrash) > 0 {
		goroutineheader(gp)
		tracebacktrap(uintptr(c.eip()), uintptr(c.esp()), 0, gp)
		tracebackothers(gp)
		print("\n")
		dumpregs(c)
	}

	if docrash {
		crash()
	}

	exit(2)
}

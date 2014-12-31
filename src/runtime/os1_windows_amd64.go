// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

var text struct{}

func dumpregs(r *context) {
	print("rax     ", hex(r.rax), "\n")
	print("rbx     ", hex(r.rbx), "\n")
	print("rcx     ", hex(r.rcx), "\n")
	print("rdi     ", hex(r.rdi), "\n")
	print("rsi     ", hex(r.rsi), "\n")
	print("rbp     ", hex(r.rbp), "\n")
	print("rsp     ", hex(r.rsp), "\n")
	print("r8      ", hex(r.r8), "\n")
	print("r9      ", hex(r.r9), "\n")
	print("r10     ", hex(r.r10), "\n")
	print("r11     ", hex(r.r11), "\n")
	print("r12     ", hex(r.r12), "\n")
	print("r13     ", hex(r.r13), "\n")
	print("r14     ", hex(r.r14), "\n")
	print("r15     ", hex(r.r15), "\n")
	print("rip     ", hex(r.rip), "\n")
	print("rflags  ", hex(r.eflags), "\n")
	print("cs      ", hex(r.segcs), "\n")
	print("fs      ", hex(r.segfs), "\n")
	print("gs      ", hex(r.seggs), "\n")
}

func isgoexception(info *exceptionrecord, r *context) bool {
	// Only handle exception if executing instructions in Go binary
	// (not Windows library code).
	if r.rip < uint64(uintptr(unsafe.Pointer(&text))) || uint64(uintptr(unsafe.Pointer(&etext))) < r.rip {
		return false
	}

	if issigpanic(info.exceptioncode) == 0 {
		return false
	}

	return true
}

// Called by sigtramp from Windows VEH handler.
// Return value signals whether the exception has been handled (EXCEPTION_CONTINUE_EXECUTION)
// or should be made available to other handlers in the chain (EXCEPTION_CONTINUE_SEARCH).

func exceptionhandler(info *exceptionrecord, r *context, gp *g) int32 {
	if !isgoexception(info, r) {
		return _EXCEPTION_CONTINUE_SEARCH
	}

	// Make it look like a call to the signal func.
	// Have to pass arguments out of band since
	// augmenting the stack frame would break
	// the unwinding code.
	gp.sig = info.exceptioncode
	gp.sigcode0 = uintptr(info.exceptioninformation[0])
	gp.sigcode1 = uintptr(info.exceptioninformation[1])
	gp.sigpc = uintptr(r.rip)

	// Only push runtime·sigpanic if r->rip != 0.
	// If r->rip == 0, probably panicked because of a
	// call to a nil func.  Not pushing that onto sp will
	// make the trace look like a call to runtime·sigpanic instead.
	// (Otherwise the trace will end at runtime·sigpanic and we
	// won't get to see who faulted.)
	if r.rip != 0 {
		sp := unsafe.Pointer(uintptr(r.rsp))
		sp = add(sp, ^uintptr(unsafe.Sizeof(uintptr(0))-1)) // sp--
		*((*uintptr)(sp)) = uintptr(r.rip)
		r.rsp = uint64(uintptr(sp))
	}
	r.rip = uint64(funcPC(sigpanic))
	return _EXCEPTION_CONTINUE_EXECUTION
}

// It seems Windows searches ContinueHandler's list even
// if ExceptionHandler returns EXCEPTION_CONTINUE_EXECUTION.
// firstcontinuehandler will stop that search,
// if exceptionhandler did the same earlier.
func firstcontinuehandler(info *exceptionrecord, r *context, gp *g) int32 {
	if !isgoexception(info, r) {
		return _EXCEPTION_CONTINUE_SEARCH
	}
	return _EXCEPTION_CONTINUE_EXECUTION
}

var testingWER bool

// lastcontinuehandler is reached, because runtime cannot handle
// current exception. lastcontinuehandler will print crash info and exit.
func lastcontinuehandler(info *exceptionrecord, r *context, gp *g) uint32 {
	if testingWER {
		return _EXCEPTION_CONTINUE_SEARCH
	}

	_g_ := getg()

	if panicking != 0 { // traceback already printed
		exit(2)
	}
	panicking = 1

	print("Exception ", hex(info.exceptioncode), " ", hex(info.exceptioninformation[0]), " ", hex(info.exceptioninformation[1]), " ", hex(r.rip), "\n")

	print("PC=", hex(r.rip), "\n")
	if _g_.m.lockedg != nil && _g_.m.ncgo > 0 && gp == _g_.m.g0 {
		print("signal arrived during cgo execution\n")
		gp = _g_.m.lockedg
	}
	print("\n")

	var docrash bool
	if gotraceback(&docrash) > 0 {
		tracebacktrap(uintptr(r.rip), uintptr(r.rsp), 0, gp)
		tracebackothers(gp)
		dumpregs(r)
	}

	if docrash {
		crash()
	}

	exit(2)
	return 0 // not reached
}

func sigenable(sig uint32) {
}

func sigdisable(sig uint32) {
}

func dosigprof(r *context, gp *g, mp *m) {
	sigprof((*byte)(unsafe.Pointer(uintptr(r.rip))), (*byte)(unsafe.Pointer(uintptr(r.rsp))), nil, gp, mp)
}

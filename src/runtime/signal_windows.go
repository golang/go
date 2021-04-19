// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

func disableWER() {
	// do not display Windows Error Reporting dialogue
	const (
		SEM_FAILCRITICALERRORS     = 0x0001
		SEM_NOGPFAULTERRORBOX      = 0x0002
		SEM_NOALIGNMENTFAULTEXCEPT = 0x0004
		SEM_NOOPENFILEERRORBOX     = 0x8000
	)
	errormode := uint32(stdcall1(_SetErrorMode, SEM_NOGPFAULTERRORBOX))
	stdcall1(_SetErrorMode, uintptr(errormode)|SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX)
}

// in sys_windows_386.s and sys_windows_amd64.s
func exceptiontramp()
func firstcontinuetramp()
func lastcontinuetramp()

func initExceptionHandler() {
	stdcall2(_AddVectoredExceptionHandler, 1, funcPC(exceptiontramp))
	if _AddVectoredContinueHandler == nil || unsafe.Sizeof(&_AddVectoredContinueHandler) == 4 {
		// use SetUnhandledExceptionFilter for windows-386 or
		// if VectoredContinueHandler is unavailable.
		// note: SetUnhandledExceptionFilter handler won't be called, if debugging.
		stdcall1(_SetUnhandledExceptionFilter, funcPC(lastcontinuetramp))
	} else {
		stdcall2(_AddVectoredContinueHandler, 1, funcPC(firstcontinuetramp))
		stdcall2(_AddVectoredContinueHandler, 0, funcPC(lastcontinuetramp))
	}
}

func isgoexception(info *exceptionrecord, r *context) bool {
	// Only handle exception if executing instructions in Go binary
	// (not Windows library code).
	// TODO(mwhudson): needs to loop to support shared libs
	if r.ip() < firstmoduledata.text || firstmoduledata.etext < r.ip() {
		return false
	}

	// Go will only handle some exceptions.
	switch info.exceptioncode {
	default:
		return false
	case _EXCEPTION_ACCESS_VIOLATION:
	case _EXCEPTION_INT_DIVIDE_BY_ZERO:
	case _EXCEPTION_INT_OVERFLOW:
	case _EXCEPTION_FLT_DENORMAL_OPERAND:
	case _EXCEPTION_FLT_DIVIDE_BY_ZERO:
	case _EXCEPTION_FLT_INEXACT_RESULT:
	case _EXCEPTION_FLT_OVERFLOW:
	case _EXCEPTION_FLT_UNDERFLOW:
	case _EXCEPTION_BREAKPOINT:
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
	gp.sigpc = r.ip()

	// Only push runtime·sigpanic if r.ip() != 0.
	// If r.ip() == 0, probably panicked because of a
	// call to a nil func. Not pushing that onto sp will
	// make the trace look like a call to runtime·sigpanic instead.
	// (Otherwise the trace will end at runtime·sigpanic and we
	// won't get to see who faulted.)
	if r.ip() != 0 {
		sp := unsafe.Pointer(r.sp())
		sp = add(sp, ^(unsafe.Sizeof(uintptr(0)) - 1)) // sp--
		*((*uintptr)(sp)) = r.ip()
		r.setsp(uintptr(sp))
	}
	r.setip(funcPC(sigpanic))
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
func lastcontinuehandler(info *exceptionrecord, r *context, gp *g) int32 {
	if testingWER {
		return _EXCEPTION_CONTINUE_SEARCH
	}

	_g_ := getg()

	if panicking != 0 { // traceback already printed
		exit(2)
	}
	panicking = 1

	print("Exception ", hex(info.exceptioncode), " ", hex(info.exceptioninformation[0]), " ", hex(info.exceptioninformation[1]), " ", hex(r.ip()), "\n")

	print("PC=", hex(r.ip()), "\n")
	if _g_.m.lockedg != nil && _g_.m.ncgo > 0 && gp == _g_.m.g0 {
		if iscgo {
			print("signal arrived during external code execution\n")
		}
		gp = _g_.m.lockedg
	}
	print("\n")

	level, _, docrash := gotraceback()
	if level > 0 {
		tracebacktrap(r.ip(), r.sp(), 0, gp)
		tracebackothers(gp)
		dumpregs(r)
	}

	if docrash {
		crash()
	}

	exit(2)
	return 0 // not reached
}

func sigpanic() {
	g := getg()
	if !canpanic(g) {
		throw("unexpected signal during runtime execution")
	}

	switch g.sig {
	case _EXCEPTION_ACCESS_VIOLATION:
		if g.sigcode1 < 0x1000 || g.paniconfault {
			panicmem()
		}
		print("unexpected fault address ", hex(g.sigcode1), "\n")
		throw("fault")
	case _EXCEPTION_INT_DIVIDE_BY_ZERO:
		panicdivide()
	case _EXCEPTION_INT_OVERFLOW:
		panicoverflow()
	case _EXCEPTION_FLT_DENORMAL_OPERAND,
		_EXCEPTION_FLT_DIVIDE_BY_ZERO,
		_EXCEPTION_FLT_INEXACT_RESULT,
		_EXCEPTION_FLT_OVERFLOW,
		_EXCEPTION_FLT_UNDERFLOW:
		panicfloat()
	}
	throw("fault")
}

var (
	badsignalmsg [100]byte
	badsignallen int32
)

func setBadSignalMsg() {
	const msg = "runtime: signal received on thread not created by Go.\n"
	for i, c := range msg {
		badsignalmsg[i] = byte(c)
		badsignallen++
	}
}

// Following are not implemented.

func initsig(preinit bool) {
}

func sigenable(sig uint32) {
}

func sigdisable(sig uint32) {
}

func sigignore(sig uint32) {
}

func badsignal2()

func raisebadsignal(sig uint32) {
	badsignal2()
}

func signame(sig uint32) string {
	return ""
}

func crash() {
	// TODO: This routine should do whatever is needed
	// to make the Windows program abort/crash as it
	// would if Go was not intercepting signals.
	// On Unix the routine would remove the custom signal
	// handler and then raise a signal (like SIGABRT).
	// Something like that should happen here.
	// It's okay to leave this empty for now: if crash returns
	// the ordinary exit-after-panic happens.
}

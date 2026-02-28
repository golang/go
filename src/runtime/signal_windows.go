// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/runtime/syscall/windows"
	"unsafe"
)

func preventErrorDialogs() {
	errormode := stdcall(_GetErrorMode)
	stdcall(_SetErrorMode, errormode|windows.SEM_FAILCRITICALERRORS|windows.SEM_NOGPFAULTERRORBOX|windows.SEM_NOOPENFILEERRORBOX)

	// Disable WER fault reporting UI.
	// Do this even if WER is disabled as a whole,
	// as WER might be enabled later with setTraceback("wer")
	// and we still want the fault reporting UI to be disabled if this happens.
	var werflags uintptr
	stdcall(_WerGetFlags, windows.CurrentProcess, uintptr(unsafe.Pointer(&werflags)))
	stdcall(_WerSetFlags, werflags|windows.WER_FAULT_REPORTING_NO_UI)
}

// enableWER re-enables Windows error reporting without fault reporting UI.
func enableWER() {
	// re-enable Windows Error Reporting
	errormode := stdcall(_GetErrorMode)
	if errormode&windows.SEM_NOGPFAULTERRORBOX != 0 {
		stdcall(_SetErrorMode, errormode^windows.SEM_NOGPFAULTERRORBOX)
	}
}

// in sys_windows_386.s, sys_windows_amd64.s, and sys_windows_arm64.s
func exceptiontramp()
func firstcontinuetramp()
func lastcontinuetramp()
func sehtramp()
func sigresume()

func initExceptionHandler() {
	stdcall(_AddVectoredExceptionHandler, 1, abi.FuncPCABI0(exceptiontramp))
	if GOARCH == "386" {
		// use SetUnhandledExceptionFilter for windows-386.
		// note: SetUnhandledExceptionFilter handler won't be called, if debugging.
		stdcall(_SetUnhandledExceptionFilter, abi.FuncPCABI0(lastcontinuetramp))
	} else {
		stdcall(_AddVectoredContinueHandler, 1, abi.FuncPCABI0(firstcontinuetramp))
		stdcall(_AddVectoredContinueHandler, 0, abi.FuncPCABI0(lastcontinuetramp))
	}
}

// isAbort returns true, if context r describes exception raised
// by calling runtime.abort function.
//
//go:nosplit
func isAbort(r *windows.Context) bool {
	pc := r.PC()
	if GOARCH == "386" || GOARCH == "amd64" {
		// In the case of an abort, the exception IP is one byte after
		// the INT3 (this differs from UNIX OSes).
		pc--
	}
	return isAbortPC(pc)
}

// isgoexception reports whether this exception should be translated
// into a Go panic or throw.
//
// It is nosplit to avoid growing the stack in case we're aborting
// because of a stack overflow.
//
//go:nosplit
func isgoexception(info *windows.ExceptionRecord, r *windows.Context) bool {
	// Only handle exception if executing instructions in Go binary
	// (not Windows library code).
	// TODO(mwhudson): needs to loop to support shared libs
	if r.PC() < firstmoduledata.text || firstmoduledata.etext < r.PC() {
		return false
	}

	// Go will only handle some exceptions.
	switch info.ExceptionCode {
	default:
		return false
	case windows.EXCEPTION_ACCESS_VIOLATION:
	case windows.EXCEPTION_IN_PAGE_ERROR:
	case windows.EXCEPTION_INT_DIVIDE_BY_ZERO:
	case windows.EXCEPTION_INT_OVERFLOW:
	case windows.EXCEPTION_FLT_DENORMAL_OPERAND:
	case windows.EXCEPTION_FLT_DIVIDE_BY_ZERO:
	case windows.EXCEPTION_FLT_INEXACT_RESULT:
	case windows.EXCEPTION_FLT_OVERFLOW:
	case windows.EXCEPTION_FLT_UNDERFLOW:
	case windows.EXCEPTION_BREAKPOINT:
	case windows.EXCEPTION_ILLEGAL_INSTRUCTION: // breakpoint arrives this way on arm64
	}
	return true
}

const (
	callbackVEH = iota
	callbackFirstVCH
	callbackLastVCH
)

// sigFetchGSafe is like getg() but without panicking
// when TLS is not set.
// Only implemented on windows/386, which is the only
// arch that loads TLS when calling getg(). Others
// use a dedicated register.
func sigFetchGSafe() *g

func sigFetchG() *g {
	if GOARCH == "386" {
		return sigFetchGSafe()
	}
	return getg()
}

// sigtrampgo is called from the exception handler function, sigtramp,
// written in assembly code.
// Return EXCEPTION_CONTINUE_EXECUTION if the exception is handled,
// else return EXCEPTION_CONTINUE_SEARCH.
//
// It is nosplit for the same reason as exceptionhandler.
//
//go:nosplit
func sigtrampgo(ep *windows.ExceptionPointers, kind int) int32 {
	gp := sigFetchG()
	if gp == nil {
		return windows.EXCEPTION_CONTINUE_SEARCH
	}

	var fn func(info *windows.ExceptionRecord, r *windows.Context, gp *g) int32
	switch kind {
	case callbackVEH:
		fn = exceptionhandler
	case callbackFirstVCH:
		fn = firstcontinuehandler
	case callbackLastVCH:
		fn = lastcontinuehandler
	default:
		throw("unknown sigtramp callback")
	}

	// Check if we are running on g0 stack, and if we are,
	// call fn directly instead of creating the closure.
	// for the systemstack argument.
	//
	// A closure can't be marked as nosplit, so it might
	// call morestack if we are at the g0 stack limit.
	// If that happens, the runtime will call abort
	// and end up in sigtrampgo again.
	// TODO: revisit this workaround if/when closures
	// can be compiled as nosplit.
	//
	// Note that this scenario should only occur on
	// TestG0StackOverflow. Any other occurrence should
	// be treated as a bug.
	var ret int32
	if gp != gp.m.g0 {
		systemstack(func() {
			ret = fn(ep.Record, ep.Context, gp)
		})
	} else {
		ret = fn(ep.Record, ep.Context, gp)
	}
	if ret == windows.EXCEPTION_CONTINUE_SEARCH {
		return ret
	}

	// Check if we need to set up the control flow guard workaround.
	// On Windows, the stack pointer in the context must lie within
	// system stack limits when we resume from exception.
	// Store the resume SP and PC in alternate registers
	// and return to sigresume on the g0 stack.
	// sigresume makes no use of the stack at all,
	// loading SP from RX and jumping to RY, being RX and RY two scratch registers.
	// Note that blindly smashing RX and RY is only safe because we know sigpanic
	// will not actually return to the original frame, so the registers
	// are effectively dead. But this does mean we can't use the
	// same mechanism for async preemption.
	if ep.Context.PC() == abi.FuncPCABI0(sigresume) {
		// sigresume has already been set up by a previous exception.
		return ret
	}
	prepareContextForSigResume(ep.Context)
	ep.Context.SetSP(gp.m.g0.sched.sp)
	ep.Context.SetPC(abi.FuncPCABI0(sigresume))
	return ret
}

// Called by sigtramp from Windows VEH handler.
// Return value signals whether the exception has been handled (EXCEPTION_CONTINUE_EXECUTION)
// or should be made available to other handlers in the chain (EXCEPTION_CONTINUE_SEARCH).
//
// This is nosplit to avoid growing the stack until we've checked for
// _EXCEPTION_BREAKPOINT, which is raised by abort() if we overflow the g0 stack.
//
//go:nosplit
func exceptionhandler(info *windows.ExceptionRecord, r *windows.Context, gp *g) int32 {
	if !isgoexception(info, r) {
		return windows.EXCEPTION_CONTINUE_SEARCH
	}

	if gp.throwsplit || isAbort(r) {
		// We can't safely sigpanic because it may grow the stack.
		// Or this is a call to abort.
		// Don't go through any more of the Windows handler chain.
		// Crash now.
		winthrow(info, r, gp)
	}

	// After this point, it is safe to grow the stack.

	// Make it look like a call to the signal func.
	// Have to pass arguments out of band since
	// augmenting the stack frame would break
	// the unwinding code.
	gp.sig = info.ExceptionCode
	gp.sigcode0 = info.ExceptionInformation[0]
	gp.sigcode1 = info.ExceptionInformation[1]
	gp.sigpc = r.PC()

	// Only push runtime·sigpanic if r.ip() != 0.
	// If r.ip() == 0, probably panicked because of a
	// call to a nil func. Not pushing that onto sp will
	// make the trace look like a call to runtime·sigpanic instead.
	// (Otherwise the trace will end at runtime·sigpanic and we
	// won't get to see who faulted.)
	// Also don't push a sigpanic frame if the faulting PC
	// is the entry of asyncPreempt. In this case, we suspended
	// the thread right between the fault and the exception handler
	// starting to run, and we have pushed an asyncPreempt call.
	// The exception is not from asyncPreempt, so not to push a
	// sigpanic call to make it look like that. Instead, just
	// overwrite the PC. (See issue #35773)
	if r.PC() != 0 && r.PC() != abi.FuncPCABI0(asyncPreempt) {
		r.PushCall(abi.FuncPCABI0(sigpanic0), r.PC())
	} else {
		// Not safe to push the call. Just clobber the frame.
		r.SetPC(abi.FuncPCABI0(sigpanic0))
	}
	return windows.EXCEPTION_CONTINUE_EXECUTION
}

// sehhandler is reached as part of the SEH chain.
//
// It is nosplit for the same reason as exceptionhandler.
//
//go:nosplit
func sehhandler(_ *windows.ExceptionRecord, _ uint64, _ *windows.Context, dctxt *windows.DISPATCHER_CONTEXT) int32 {
	g0 := getg()
	if g0 == nil || g0.m.curg == nil {
		// No g available, nothing to do here.
		return windows.EXCEPTION_CONTINUE_SEARCH_SEH
	}
	// The Windows SEH machinery will unwind the stack until it finds
	// a frame with a handler for the exception or until the frame is
	// outside the stack boundaries, in which case it will call the
	// UnhandledExceptionFilter. Unfortunately, it doesn't know about
	// the goroutine stack, so it will stop unwinding when it reaches the
	// first frame not running in g0. As a result, neither non-Go exceptions
	// handlers higher up the stack nor UnhandledExceptionFilter will be called.
	//
	// To work around this, manually unwind the stack until the top of the goroutine
	// stack is reached, and then pass the control back to Windows.
	gp := g0.m.curg
	ctxt := dctxt.Ctx()
	var base, sp uintptr
	for {
		entry := stdcall(_RtlLookupFunctionEntry, ctxt.PC(), uintptr(unsafe.Pointer(&base)), 0)
		if entry == 0 {
			break
		}
		stdcall(_RtlVirtualUnwind, 0, base, ctxt.PC(), entry, uintptr(unsafe.Pointer(ctxt)), 0, uintptr(unsafe.Pointer(&sp)), 0)
		if sp < gp.stack.lo || gp.stack.hi <= sp {
			break
		}
	}
	return windows.EXCEPTION_CONTINUE_SEARCH_SEH
}

// It seems Windows searches ContinueHandler's list even
// if ExceptionHandler returns EXCEPTION_CONTINUE_EXECUTION.
// firstcontinuehandler will stop that search,
// if exceptionhandler did the same earlier.
//
// It is nosplit for the same reason as exceptionhandler.
//
//go:nosplit
func firstcontinuehandler(info *windows.ExceptionRecord, r *windows.Context, gp *g) int32 {
	if !isgoexception(info, r) {
		return windows.EXCEPTION_CONTINUE_SEARCH
	}
	return windows.EXCEPTION_CONTINUE_EXECUTION
}

// lastcontinuehandler is reached, because runtime cannot handle
// current exception. lastcontinuehandler will print crash info and exit.
//
// It is nosplit for the same reason as exceptionhandler.
//
//go:nosplit
func lastcontinuehandler(info *windows.ExceptionRecord, r *windows.Context, gp *g) int32 {
	if islibrary || isarchive {
		// Go DLL/archive has been loaded in a non-go program.
		// If the exception does not originate from go, the go runtime
		// should not take responsibility of crashing the process.
		return windows.EXCEPTION_CONTINUE_SEARCH
	}

	// VEH is called before SEH, but arm64 MSVC DLLs use SEH to trap
	// illegal instructions during runtime initialization to determine
	// CPU features, so if we make it to the last handler and we're
	// arm64 and it's an illegal instruction and this is coming from
	// non-Go code, then assume it's this runtime probing happen, and
	// pass that onward to SEH.
	if GOARCH == "arm64" && info.ExceptionCode == windows.EXCEPTION_ILLEGAL_INSTRUCTION &&
		(r.PC() < firstmoduledata.text || firstmoduledata.etext < r.PC()) {
		return windows.EXCEPTION_CONTINUE_SEARCH
	}

	winthrow(info, r, gp)
	return 0 // not reached
}

// Always called on g0. gp is the G where the exception occurred.
//
//go:nosplit
func winthrow(info *windows.ExceptionRecord, r *windows.Context, gp *g) {
	g0 := getg()

	if panicking.Load() != 0 { // traceback already printed
		exit(2)
	}
	panicking.Store(1)

	// In case we're handling a g0 stack overflow, blow away the
	// g0 stack bounds so we have room to print the traceback. If
	// this somehow overflows the stack, the OS will trap it.
	g0.stack.lo = 0
	g0.stackguard0 = g0.stack.lo + stackGuard
	g0.stackguard1 = g0.stackguard0

	print("Exception ", hex(info.ExceptionCode), " ", hex(info.ExceptionInformation[0]), " ", hex(info.ExceptionInformation[1]), " ", hex(r.PC()), "\n")

	print("PC=", hex(r.PC()), "\n")
	if g0.m.incgo && gp == g0.m.g0 && g0.m.curg != nil {
		if iscgo {
			print("signal arrived during external code execution\n")
		}
		gp = g0.m.curg
	}
	print("\n")

	g0.m.throwing = throwTypeRuntime
	g0.m.caughtsig.set(gp)

	level, _, docrash := gotraceback()
	if level > 0 {
		tracebacktrap(r.PC(), r.SP(), r.LR(), gp)
		tracebackothers(gp)
		dumpregs(r)
	}

	if docrash {
		dieFromException(info, r)
	}

	exit(2)
}

func sigpanic() {
	gp := getg()
	if !canpanic() {
		throw("unexpected signal during runtime execution")
	}

	switch gp.sig {
	case windows.EXCEPTION_ACCESS_VIOLATION, windows.EXCEPTION_IN_PAGE_ERROR:
		if gp.sigcode1 < 0x1000 {
			panicmem()
		}
		if gp.paniconfault {
			panicmemAddr(gp.sigcode1)
		}
		if inUserArenaChunk(gp.sigcode1) {
			// We could check that the arena chunk is explicitly set to fault,
			// but the fact that we faulted on accessing it is enough to prove
			// that it is.
			print("accessed data from freed user arena ", hex(gp.sigcode1), "\n")
		} else {
			print("unexpected fault address ", hex(gp.sigcode1), "\n")
		}
		throw("fault")
	case windows.EXCEPTION_INT_DIVIDE_BY_ZERO:
		panicdivide()
	case windows.EXCEPTION_INT_OVERFLOW:
		panicoverflow()
	case windows.EXCEPTION_FLT_DENORMAL_OPERAND,
		windows.EXCEPTION_FLT_DIVIDE_BY_ZERO,
		windows.EXCEPTION_FLT_INEXACT_RESULT,
		windows.EXCEPTION_FLT_OVERFLOW,
		windows.EXCEPTION_FLT_UNDERFLOW:
		panicfloat()
	}
	throw("fault")
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

func signame(sig uint32) string {
	return ""
}

//go:nosplit
func crash() {
	dieFromException(nil, nil)
}

// dieFromException raises an exception that bypasses all exception handlers.
// This provides the expected exit status for the shell.
//
//go:nosplit
func dieFromException(info *windows.ExceptionRecord, r *windows.Context) {
	if info == nil {
		gp := getg()
		if gp.sig != 0 {
			// Try to reconstruct an exception record from
			// the exception information stored in gp.
			info = &windows.ExceptionRecord{
				ExceptionAddress: gp.sigpc,
				ExceptionCode:    gp.sig,
				NumberParameters: 2,
			}
			info.ExceptionInformation[0] = gp.sigcode0
			info.ExceptionInformation[1] = gp.sigcode1
		} else {
			// By default, a failing Go application exits with exit code 2.
			// Use this value when gp does not contain exception info.
			info = &windows.ExceptionRecord{
				ExceptionCode: 2,
			}
		}
	}
	stdcall(_RaiseFailFastException, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(r)), windows.FAIL_FAST_GENERATE_EXCEPTION_ADDRESS)
}

// gsignalStack is unused on Windows.
type gsignalStack struct{}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Though the debug call function feature is not enabled on
// ppc64, inserted ppc64 to avoid missing Go declaration error
// for debugCallPanicked while building runtime.test
//go:build amd64 || arm64 || ppc64le || ppc64

package runtime

import (
	"internal/abi"
	"unsafe"
)

const (
	debugCallSystemStack = "executing on Go runtime stack"
	debugCallUnknownFunc = "call from unknown function"
	debugCallRuntime     = "call from within the Go runtime"
	debugCallUnsafePoint = "call not at safe point"
)

func debugCallV2()
func debugCallPanicked(val any)

// debugCallCheck checks whether it is safe to inject a debugger
// function call with return PC pc. If not, it returns a string
// explaining why.
//
//go:nosplit
func debugCallCheck(pc uintptr) string {
	// No user calls from the system stack.
	if getg() != getg().m.curg {
		return debugCallSystemStack
	}
	if sp := getcallersp(); !(getg().stack.lo < sp && sp <= getg().stack.hi) {
		// Fast syscalls (nanotime) and racecall switch to the
		// g0 stack without switching g. We can't safely make
		// a call in this state. (We can't even safely
		// systemstack.)
		return debugCallSystemStack
	}

	// Switch to the system stack to avoid overflowing the user
	// stack.
	var ret string
	systemstack(func() {
		f := findfunc(pc)
		if !f.valid() {
			ret = debugCallUnknownFunc
			return
		}

		name := funcname(f)

		switch name {
		case "debugCall32",
			"debugCall64",
			"debugCall128",
			"debugCall256",
			"debugCall512",
			"debugCall1024",
			"debugCall2048",
			"debugCall4096",
			"debugCall8192",
			"debugCall16384",
			"debugCall32768",
			"debugCall65536":
			// These functions are allowed so that the debugger can initiate multiple function calls.
			// See: https://golang.org/cl/161137/
			return
		}

		// Disallow calls from the runtime. We could
		// potentially make this condition tighter (e.g., not
		// when locks are held), but there are enough tightly
		// coded sequences (e.g., defer handling) that it's
		// better to play it safe.
		if pfx := "runtime."; len(name) > len(pfx) && name[:len(pfx)] == pfx {
			ret = debugCallRuntime
			return
		}

		// Check that this isn't an unsafe-point.
		if pc != f.entry() {
			pc--
		}
		up := pcdatavalue(f, abi.PCDATA_UnsafePoint, pc)
		if up != abi.UnsafePointSafe {
			// Not at a safe point.
			ret = debugCallUnsafePoint
		}
	})
	return ret
}

// debugCallWrap starts a new goroutine to run a debug call and blocks
// the calling goroutine. On the goroutine, it prepares to recover
// panics from the debug call, and then calls the call dispatching
// function at PC dispatch.
//
// This must be deeply nosplit because there are untyped values on the
// stack from debugCallV2.
//
//go:nosplit
func debugCallWrap(dispatch uintptr) {
	var lockedExt uint32
	callerpc := getcallerpc()
	gp := getg()

	// Lock ourselves to the OS thread.
	//
	// Debuggers rely on us running on the same thread until we get to
	// dispatch the function they asked as to.
	//
	// We're going to transfer this to the new G we just created.
	lockOSThread()

	// Create a new goroutine to execute the call on. Run this on
	// the system stack to avoid growing our stack.
	systemstack(func() {
		// TODO(mknyszek): It would be nice to wrap these arguments in an allocated
		// closure and start the goroutine with that closure, but the compiler disallows
		// implicit closure allocation in the runtime.
		fn := debugCallWrap1
		newg := newproc1(*(**funcval)(unsafe.Pointer(&fn)), gp, callerpc, false, waitReasonZero)
		args := &debugCallWrapArgs{
			dispatch: dispatch,
			callingG: gp,
		}
		newg.param = unsafe.Pointer(args)

		// Transfer locked-ness to the new goroutine.
		// Save lock state to restore later.
		mp := gp.m
		if mp != gp.lockedm.ptr() {
			throw("inconsistent lockedm")
		}
		// Save the external lock count and clear it so
		// that it can't be unlocked from the debug call.
		// Note: we already locked internally to the thread,
		// so if we were locked before we're still locked now.
		lockedExt = mp.lockedExt
		mp.lockedExt = 0

		mp.lockedg.set(newg)
		newg.lockedm.set(mp)
		gp.lockedm = 0

		// Mark the calling goroutine as being at an async
		// safe-point, since it has a few conservative frames
		// at the bottom of the stack. This also prevents
		// stack shrinks.
		gp.asyncSafePoint = true

		// Stash newg away so we can execute it below (mcall's
		// closure can't capture anything).
		gp.schedlink.set(newg)
	})

	// Switch to the new goroutine.
	mcall(func { gp ->
		// Get newg.
		newg := gp.schedlink.ptr()
		gp.schedlink = 0

		// Park the calling goroutine.
		trace := traceAcquire()
		if trace.ok() {
			// Trace the event before the transition. It may take a
			// stack trace, but we won't own the stack after the
			// transition anymore.
			trace.GoPark(traceBlockDebugCall, 1)
		}
		casGToWaiting(gp, _Grunning, waitReasonDebugCall)
		if trace.ok() {
			traceRelease(trace)
		}
		dropg()

		// Directly execute the new goroutine. The debug
		// protocol will continue on the new goroutine, so
		// it's important we not just let the scheduler do
		// this or it may resume a different goroutine.
		execute(newg, true)
	})

	// We'll resume here when the call returns.

	// Restore locked state.
	mp := gp.m
	mp.lockedExt = lockedExt
	mp.lockedg.set(gp)
	gp.lockedm.set(mp)

	// Undo the lockOSThread we did earlier.
	unlockOSThread()

	gp.asyncSafePoint = false
}

type debugCallWrapArgs struct {
	dispatch uintptr
	callingG *g
}

// debugCallWrap1 is the continuation of debugCallWrap on the callee
// goroutine.
func debugCallWrap1() {
	gp := getg()
	args := (*debugCallWrapArgs)(gp.param)
	dispatch, callingG := args.dispatch, args.callingG
	gp.param = nil

	// Dispatch call and trap panics.
	debugCallWrap2(dispatch)

	// Resume the caller goroutine.
	getg().schedlink.set(callingG)
	mcall(func { gp ->
		callingG := gp.schedlink.ptr()
		gp.schedlink = 0

		// Unlock this goroutine from the M if necessary. The
		// calling G will relock.
		if gp.lockedm != 0 {
			gp.lockedm = 0
			gp.m.lockedg = 0
		}

		// Switch back to the calling goroutine. At some point
		// the scheduler will schedule us again and we'll
		// finish exiting.
		trace := traceAcquire()
		if trace.ok() {
			// Trace the event before the transition. It may take a
			// stack trace, but we won't own the stack after the
			// transition anymore.
			trace.GoSched()
		}
		casgstatus(gp, _Grunning, _Grunnable)
		if trace.ok() {
			traceRelease(trace)
		}
		dropg()
		lock(&sched.lock)
		globrunqput(gp)
		unlock(&sched.lock)

		trace = traceAcquire()
		casgstatus(callingG, _Gwaiting, _Grunnable)
		if trace.ok() {
			trace.GoUnpark(callingG, 0)
			traceRelease(trace)
		}
		execute(callingG, true)
	})
}

func debugCallWrap2(dispatch uintptr) {
	// Call the dispatch function and trap panics.
	var dispatchF func()
	dispatchFV := funcval{dispatch}
	*(*unsafe.Pointer)(unsafe.Pointer(&dispatchF)) = noescape(unsafe.Pointer(&dispatchFV))

	var ok bool
	defer func() {
		if !ok {
			err := recover()
			debugCallPanicked(err)
		}
	}()
	dispatchF()
	ok = true
}

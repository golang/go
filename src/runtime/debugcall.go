// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64

package runtime

import "unsafe"

const (
	debugCallSystemStack = "executing on Go runtime stack"
	debugCallUnknownFunc = "call from unknown function"
	debugCallRuntime     = "call from within the Go runtime"
	debugCallUnsafePoint = "call not at safe point"
)

func debugCallV1()
func debugCallPanicked(val interface{})

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
		if pc != f.entry {
			pc--
		}
		up := pcdatavalue(f, _PCDATA_UnsafePoint, pc, nil)
		if up != _PCDATA_UnsafePointSafe {
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
// stack from debugCallV1.
//
//go:nosplit
func debugCallWrap(dispatch uintptr) {
	var lockedm bool
	var lockedExt uint32
	callerpc := getcallerpc()
	gp := getg()

	// Create a new goroutine to execute the call on. Run this on
	// the system stack to avoid growing our stack.
	systemstack(func() {
		var args struct {
			dispatch uintptr
			callingG *g
		}
		args.dispatch = dispatch
		args.callingG = gp
		fn := debugCallWrap1
		newg := newproc1(*(**funcval)(unsafe.Pointer(&fn)), unsafe.Pointer(&args), int32(unsafe.Sizeof(args)), gp, callerpc)

		// If the current G is locked, then transfer that
		// locked-ness to the new goroutine.
		if gp.lockedm != 0 {
			// Save lock state to restore later.
			mp := gp.m
			if mp != gp.lockedm.ptr() {
				throw("inconsistent lockedm")
			}

			lockedm = true
			lockedExt = mp.lockedExt

			// Transfer external lock count to internal so
			// it can't be unlocked from the debug call.
			mp.lockedInt++
			mp.lockedExt = 0

			mp.lockedg.set(newg)
			newg.lockedm.set(mp)
			gp.lockedm = 0
		}

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
	mcall(func(gp *g) {
		// Get newg.
		newg := gp.schedlink.ptr()
		gp.schedlink = 0

		// Park the calling goroutine.
		gp.waitreason = waitReasonDebugCall
		if trace.enabled {
			traceGoPark(traceEvGoBlock, 1)
		}
		casgstatus(gp, _Grunning, _Gwaiting)
		dropg()

		// Directly execute the new goroutine. The debug
		// protocol will continue on the new goroutine, so
		// it's important we not just let the scheduler do
		// this or it may resume a different goroutine.
		execute(newg, true)
	})

	// We'll resume here when the call returns.

	// Restore locked state.
	if lockedm {
		mp := gp.m
		mp.lockedExt = lockedExt
		mp.lockedInt--
		mp.lockedg.set(gp)
		gp.lockedm.set(mp)
	}

	gp.asyncSafePoint = false
}

// debugCallWrap1 is the continuation of debugCallWrap on the callee
// goroutine.
func debugCallWrap1(dispatch uintptr, callingG *g) {
	// Dispatch call and trap panics.
	debugCallWrap2(dispatch)

	// Resume the caller goroutine.
	getg().schedlink.set(callingG)
	mcall(func(gp *g) {
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
		if trace.enabled {
			traceGoSched()
		}
		casgstatus(gp, _Grunning, _Grunnable)
		dropg()
		lock(&sched.lock)
		globrunqput(gp)
		unlock(&sched.lock)

		if trace.enabled {
			traceGoUnpark(callingG, 0)
		}
		casgstatus(callingG, _Gwaiting, _Grunnable)
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

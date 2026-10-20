// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goexperiment"
	"internal/runtime/sys"
	"unsafe"
)

// logosCgocallFast is the fast-call entry for C functions carrying the
// #cgo noescape plus #cgo nocallback pair, or //logos:cgo_fastcall in
// .lgo packages (M20-S1). cmd/cgo emits a call here instead of cgocall
// for such functions; there is no second cgo pipeline.
//
// The path performs the same goroutine status dance as a syscall
// (reentersyscall around asmcgocall, exitsyscall on return), so sysmon
// retakes the P under the same rule and threshold, and it keeps the
// profiling state (syscallpc, syscallsp, ncgo) so SIGPROF and CPU
// profiles attribute C time to the Go call site. It skips the cgo
// extras that cost but add no safety here: the ncgocall statistic and
// the cgoCallers traceback reset. In return it records fast-call
// bookkeeping: m.inFastCall, the argument frame pinned in
// p.cgoPinnedPtrs[0], and the call start in p.fastCallEnterNano.
// The pin slots hold plain uintptr and are informational only: the
// argument frame stays valid through the ordinary goroutine stack
// scan, and nothing in the tree reads the slots yet.
//
// Until the M20 gate passes the path stays behind GOEXPERIMENT
// logosfastcgo and falls back to cgocall otherwise.
//
//go:linkname logosCgocallFast
func logosCgocallFast(fn, arg unsafe.Pointer) int32 {
	if !goexperiment.LogosFastCgo {
		return cgocall(fn, arg)
	}
	if fn == nil {
		throw("cgocall nil")
	}

	mp := getg().m
	if raceenabled {
		racereleasemerge(unsafe.Pointer(&racecgosync))
	}

	// Without a P there is nowhere to record the call; keep the exact
	// upstream behavior instead of inventing a new failure mode.
	pp := mp.p.ptr()
	if pp == nil {
		return cgocall(fn, arg)
	}
	logosEnterFastCall(mp, pp, arg, int64(nanotime()))

	mp.ncgo++
	osPreemptExtEnter(mp)
	mp.incgo = true

	// N.B. getcallerfp cannot be written directly as an argument for
	// the same spilling reason stated on entersyscall.
	fp := getcallerfp()
	reentersyscall(sys.GetCallerPC(), sys.GetCallerSP(), fp)

	errno := asmcgocall(fn, arg)

	mp.incgo = false
	mp.ncgo--
	if mp.p.ptr() == pp && pp.m.ptr() == mp {
		logosExitFastCall(mp, pp)
	} else {
		// Sysmon stole the P mid-call (takeP in releaseP): the
		// M-owned flag is still ours to clear, but the P-owned
		// record belongs to the new owner now. Nothing reads these
		// fields; the next fast call on that P overwrites them. A
		// steal landing between the check and the clear above stays
		// possible in principle over a few instructions, with the
		// same benign outcome.
		mp.inFastCall = false
	}
	osPreemptExtExit(mp)

	// After exitsyscall we may be rescheduled on a different M, so
	// save and restore the original M's winsyscall like cgocall does.
	winsyscall := mp.winsyscall

	exitsyscall()

	getg().m.winsyscall = winsyscall

	if raceenabled {
		raceacquire(unsafe.Pointer(&racecgosync))
	}

	if sys.DITSupported {
		// C code may have enabled or disabled DIT on this thread, restore
		// our state to the expected one.
		ditEnabled := sys.DITEnabled()
		gp := getg()
		if !gp.ditWanted && ditEnabled {
			sys.DisableDIT()
		} else if gp.ditWanted && !ditEnabled {
			sys.EnableDIT()
		}
	}

	// Keep fn, arg, and mp live across the same time warp cgocall
	// documents: a callback into Go rolls the syscall PC/SP back.
	KeepAlive(fn)
	KeepAlive(arg)
	KeepAlive(mp)

	return errno
}

// logosEnterFastCall records a fast-call entry: the argument frame goes
// in pin slot 0, the call start in the timestamp, and the M flag is set.
//
// Must be nosplit: the exit helper runs between reentersyscall and
// exitsyscall, where the stack guard is fake and any growth throws.
//
//go:nosplit
func logosEnterFastCall(mp *m, pp *p, arg unsafe.Pointer, now int64) {
	pp.cgoPinnedPtrs = [4]uintptr{uintptr(arg), 0, 0, 0}
	pp.fastCallEnterNano = now
	mp.inFastCall = true
}

// logosExitFastCall clears a fast-call entry: the flag, the timestamp,
// and all pin slots read zero after return. See logosEnterFastCall for
// why this must be nosplit.
//
//go:nosplit
func logosExitFastCall(mp *m, pp *p) {
	mp.inFastCall = false
	pp.fastCallEnterNano = 0
	pp.cgoPinnedPtrs = [4]uintptr{}
}

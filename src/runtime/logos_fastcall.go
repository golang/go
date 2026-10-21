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
// profiles attribute C time to the Go call site. It records fast-call
// diagnostics on the original M: m.inFastCall, the argument frame in
// m.fastCgoFrame[0], and the call start in m.fastCallEnterNano.
// The slots hold plain uintptr and are informational only: the
// argument frame stays valid through the ordinary goroutine stack
// scan, nothing in the tree reads them, and they are not pins.
//
// Until the M20 gate passes the path stays behind GOEXPERIMENT
// logosfastcgo and falls back to cgocall otherwise.
//
//go:linkname logosCgocallFast
//go:nosplit
func logosCgocallFast(fn, arg unsafe.Pointer) int32 {
	if !goexperiment.LogosFastCgo {
		return cgocall(fn, arg)
	}
	if !iscgo && GOOS != "solaris" && GOOS != "illumos" && GOOS != "windows" {
		throw("cgocall unavailable")
	}
	if fn == nil {
		throw("cgocall nil")
	}

	mp := getg().m
	if raceenabled {
		racereleasemerge(unsafe.Pointer(&racecgosync))
	}

	mp.ncgocall++
	if mp.cgoCallers != nil {
		mp.cgoCallers[0] = 0
	}
	logosRecordFastCall(mp, arg, int64(nanotime()))

	// N.B. getcallerfp cannot be written directly as an argument for
	// the same spilling reason stated on entersyscall.
	fp := getcallerfp()
	reentersyscall(sys.GetCallerPC(), sys.GetCallerSP(), fp)

	osPreemptExtEnter(mp)
	mp.incgo = true
	mp.ncgo++

	errno := asmcgocall(fn, arg)

	mp.incgo = false
	mp.ncgo--
	logosClearFastCall(mp)

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

// logosRecordFastCall records a fast-call entry on the original M: the
// argument frame in slot 0, the call start in the timestamp, and the
// flag is set. Slots are diagnostics, not pins.
//
// Must be nosplit: the clear helper runs between reentersyscall and
// exitsyscall, where the stack guard is fake and any growth throws.
//
//go:nosplit
func logosRecordFastCall(mp *m, arg unsafe.Pointer, now int64) {
	mp.fastCgoFrame = [4]uintptr{uintptr(arg), 0, 0, 0}
	mp.fastCallEnterNano = now
	mp.inFastCall = true
}

// logosClearFastCall clears a fast-call record: flag, timestamp, and
// slots all read zero. See logosRecordFastCall for why this must be
// nosplit.
//
//go:nosplit
func logosClearFastCall(mp *m) {
	mp.inFastCall = false
	mp.fastCallEnterNano = 0
	mp.fastCgoFrame = [4]uintptr{}
}

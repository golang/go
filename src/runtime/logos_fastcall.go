// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goexperiment"
	"internal/runtime/sys"
	"unsafe"
)

var logosFastcallOffHook func(fn, arg unsafe.Pointer) int32

//go:linkname logosCgocallFast
//go:nosplit
func logosCgocallFast(fn, arg unsafe.Pointer) int32 {
	if !iscgo && GOOS != "solaris" && GOOS != "illumos" && GOOS != "windows" {
		throw("cgocall unavailable")
	}
	if fn == nil {
		throw("cgocall nil")
	}
	gp := getg()
	if gp.nocgocallback {
		panic("runtime: unexpected setting cgoNoCallback")
	}
	if !goexperiment.LogosFastCgo {
		if logosFastcallOffHook != nil {
			return logosFastcallOffHook(fn, arg)
		}
		return cgocall(fn, arg)
	}
	gp.nocgocallback = true
	mp := gp.m
	if raceenabled {
		racereleasemerge(unsafe.Pointer(&racecgosync))
	}
	mp.ncgocall++
	if mp.cgoCallers != nil {
		mp.cgoCallers[0] = 0
	}
	logosRecordFastCall(mp, arg)

	fp := getcallerfp()
	reentersyscall(sys.GetCallerPC(), sys.GetCallerSP(), fp)

	osPreemptExtEnter(mp)
	mp.incgo = true
	mp.ncgo++

	errno := asmcgocall(fn, arg)

	gp.nocgocallback = false
	mp.incgo = false
	mp.ncgo--
	logosClearFastCall(mp)

	osPreemptExtExit(mp)

	winsyscall := mp.winsyscall

	exitsyscall()

	getg().m.winsyscall = winsyscall

	if raceenabled {
		raceacquire(unsafe.Pointer(&racecgosync))
	}

	if sys.DITSupported {
		ditEnabled := sys.DITEnabled()
		gp := getg()
		if !gp.ditWanted && ditEnabled {
			sys.DisableDIT()
		} else if gp.ditWanted && !ditEnabled {
			sys.EnableDIT()
		}
	}

	KeepAlive(fn)
	KeepAlive(arg)
	KeepAlive(mp)

	return errno
}

//go:nosplit
func logosRecordFastCall(mp *m, arg unsafe.Pointer) {
	mp.fastCgoFrame[0] = uintptr(arg)
	mp.fastCgoFrame[1] = 0
	mp.fastCgoFrame[2] = 0
	mp.fastCgoFrame[3] = 0
	mp.inFastCall = true
}

//go:nosplit
func logosClearFastCall(mp *m) {
	mp.inFastCall = false
	mp.fastCgoFrame[0] = 0
	mp.fastCgoFrame[1] = 0
	mp.fastCgoFrame[2] = 0
	mp.fastCgoFrame[3] = 0
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime

import (
	"internal/abi"
	"internal/runtime/sys"
	"unsafe"
)

// LogosArenaScanWork scans the caller's regions through scanstack. Every
// pointer slot in those regions must be nil. It returns the GC work and the
// separate physical-stack size sample from that scan.
func LogosArenaScanWork() (work int64, stackSample uint64) {
	caller := getg()
	var gp g
	var frame [16]uintptr
	systemstack(func() {
		// Use the same terminal frame as a new goroutine. It has no locals
		// or arguments. The syscall state prevents stack shrinking; this
		// synthetic G and its backing storage never enter the scheduler.
		lo := uintptr(unsafe.Pointer(&frame[0]))
		hi := alignDown(lo+unsafe.Sizeof(frame), sys.StackAlign)
		sp := hi - 8*unsafe.Sizeof(uintptr(0))
		gp.stack = stack{lo: lo, hi: hi}
		if usesLR {
			prepGoExitFrame(sp)
		}
		gp.sched.sp = sp
		gp.sched.pc = abi.FuncPCABI0(goexit) + sys.PCQuantum
		gp.stktopsp = sp
		gp.syscallsp = sp
		gp.syscallpc = gp.sched.pc
		gp.atomicstatus.Store(_Gscan | _Gsyscall)
		gp.logosArena = caller.logosArena
		gp.logosRegions = caller.logosRegions

		pp := getg().m.p.ptr()
		beforeSize, beforeCount := pp.scannedStackSize, pp.scannedStacks
		// The frame has no pointers, and all region pointer slots are nil,
		// so no heap object can be marked. A nil work buffer is safe here.
		// systemstack also keeps this P's sampling counters stable.
		work = scanstack(&gp, nil)
		stackSample = pp.scannedStackSize - beforeSize
		pp.scannedStackSize, pp.scannedStacks = beforeSize, beforeCount
	})
	return
}

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
	"unsafe"
)

// userFrameRegion describes a region of code that is not managed by the
// Go compiler (e.g., JIT-compiled code, WASM engines, embedded VMs).
//
// The runtime consults user frame regions when it encounters a PC that
// does not belong to any Go module. This allows the unwinder, panic
// handler, and GC to handle these frames gracefully instead of crashing.
type userFrameRegion struct {
	start uintptr // inclusive
	end   uintptr // exclusive

	unwindMode userFrameUnwindMode

	// describe returns human-readable frame info for a PC in this region.
	// Must not allocate. May be nil.
	describe func(pc uintptr) (name string, file string, line int, ok bool)

	// next returns the caller PC and SP for a frame at the given PC/SP.
	// This allows the unwinder to continue past user frames.
	// Must not allocate. May be nil (treated as unwindStop).
	next func(pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool)

	// scanStack is called during GC to report Go pointers held in
	// user frame stack segments. Must not allocate. May be nil.
	scanStack func(report func(ptr uintptr))

	handle uintptr
}

type userFrameUnwindMode uint8

const (
	userFrameUnwindStop    userFrameUnwindMode = iota // traceback ends at boundary
	userFrameUnwindSkip                               // frames are skipped, traceback continues
	userFrameUnwindDeclare                            // frames are described via callbacks
)

// User frame registry.
//
// Uses a copy-on-write scheme: writers (register/unregister) allocate a
// new snapshot via persistentalloc and atomically swap the pointer.
// Readers (findUserFrameRegion) load the pointer atomically and operate
// on an immutable snapshot. This avoids data races without requiring
// the reader to take a lock (which is important because the reader runs
// on the system stack during stack unwinding and signal handling).
const maxUserFrameRegions = 64

// userFrameSnapshot is the immutable snapshot published to readers.
type userFrameSnapshot struct {
	regions [maxUserFrameRegions]userFrameRegion
	count   int
}

var (
	userFrameLock   mutex
	userFrameSnap   unsafe.Pointer // *userFrameSnapshot, accessed atomically
	userFrameHandle uintptr        // monotonic counter for handles
)

// loadUserFrameSnapshot atomically loads the current snapshot.
//
//go:nosplit
func loadUserFrameSnapshot() *userFrameSnapshot {
	p := atomic.Loadp(unsafe.Pointer(&userFrameSnap))
	return (*userFrameSnapshot)(p)
}

func registerUserFrameRegion(r userFrameRegion) uintptr {
	if r.start >= r.end {
		throw("registerUserFrameRegion: invalid address range")
	}
	if r.unwindMode != userFrameUnwindStop && r.next == nil {
		throw("registerUserFrameRegion: Next callback required for UnwindSkip/UnwindDeclare")
	}

	lock(&userFrameLock)

	// Read current snapshot (may be nil on first call).
	old := loadUserFrameSnapshot()
	var oldCount int
	if old != nil {
		oldCount = old.count
	}

	if oldCount >= maxUserFrameRegions {
		unlock(&userFrameLock)
		throw("registerUserFrameRegion: too many regions")
	}

	// Check for overlaps.
	for i := 0; i < oldCount; i++ {
		fr := &old.regions[i]
		if r.start < fr.end && r.end > fr.start {
			unlock(&userFrameLock)
			throw("registerUserFrameRegion: overlapping region")
		}
	}

	// Assign handle.
	userFrameHandle++
	r.handle = userFrameHandle

	// Build new snapshot with the region inserted in sorted order.
	mem := persistentalloc(unsafe.Sizeof(userFrameSnapshot{}), unsafe.Sizeof(uintptr(0)), &memstats.other_sys)
	snap := (*userFrameSnapshot)(mem)

	pos := oldCount
	for i := 0; i < oldCount; i++ {
		if r.start < old.regions[i].start {
			pos = i
			break
		}
	}
	// Copy elements before pos.
	for i := 0; i < pos; i++ {
		snap.regions[i] = old.regions[i]
	}
	// Insert new region.
	snap.regions[pos] = r
	// Copy elements after pos.
	for i := pos; i < oldCount; i++ {
		snap.regions[i+1] = old.regions[i]
	}
	snap.count = oldCount + 1

	// Publish atomically. Readers will see the complete new snapshot.
	atomic.StorepNoWB(unsafe.Pointer(&userFrameSnap), unsafe.Pointer(snap))

	h := r.handle
	unlock(&userFrameLock)
	return h
}

func unregisterUserFrameRegion(handle uintptr) {
	lock(&userFrameLock)
	old := loadUserFrameSnapshot()
	if old == nil {
		unlock(&userFrameLock)
		throw("unregisterUserFrameRegion: handle not found")
	}

	// Find the region.
	idx := -1
	for i := 0; i < old.count; i++ {
		if old.regions[i].handle == handle {
			idx = i
			break
		}
	}
	if idx < 0 {
		unlock(&userFrameLock)
		throw("unregisterUserFrameRegion: handle not found")
	}

	// Build new snapshot without the region.
	mem := persistentalloc(unsafe.Sizeof(userFrameSnapshot{}), unsafe.Sizeof(uintptr(0)), &memstats.other_sys)
	snap := (*userFrameSnapshot)(mem)
	j := 0
	for i := 0; i < old.count; i++ {
		if i != idx {
			snap.regions[j] = old.regions[i]
			j++
		}
	}
	snap.count = old.count - 1

	atomic.StorepNoWB(unsafe.Pointer(&userFrameSnap), unsafe.Pointer(snap))

	unlock(&userFrameLock)
}

// findUserFrameRegion returns the user frame region containing pc, or nil.
// Uses an atomic load to read an immutable snapshot — no lock needed.
//
//go:nosplit
func findUserFrameRegion(pc uintptr) *userFrameRegion {
	snap := loadUserFrameSnapshot()
	if snap == nil {
		return nil
	}
	// Binary search over sorted regions.
	lo, hi := 0, snap.count
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if snap.regions[mid].end <= pc {
			lo = mid + 1
		} else if snap.regions[mid].start > pc {
			hi = mid
		} else {
			return &snap.regions[mid]
		}
	}
	return nil
}

func userFrameNext(r *userFrameRegion, pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool) {
	if r.next == nil {
		return 0, 0, 0, false
	}
	return r.next(pc, sp)
}

// userFrameScanRoots calls all registered ScanStack callbacks during GC.
// Called from markroot as a fixed root.
//
// The report callback passed to ScanStack uses a package-level variable
// to avoid creating a closure (which is forbidden in the runtime package).
// This is safe because markroot is single-threaded per root index.
var userFrameScanGCW *gcWork

func userFrameScanRoots(gcw *gcWork) {
	snap := loadUserFrameSnapshot()
	if snap == nil {
		return
	}
	userFrameScanGCW = gcw
	for i := 0; i < snap.count; i++ {
		if snap.regions[i].scanStack != nil {
			snap.regions[i].scanStack(userFrameScanReport)
		}
	}
	userFrameScanGCW = nil
}

// userFrameScanReport is the report function passed to ScanStack callbacks.
// It marks a pointer-sized slot as a GC root.
//
//go:nosplit
func userFrameScanReport(ptr uintptr) {
	gcw := userFrameScanGCW
	if gcw == nil {
		return
	}
	// ptr is the address of a slot containing a Go pointer.
	scanblock(ptr, unsafe.Sizeof(uintptr(0)), &oneptrmask[0], gcw, nil)
}

// userFramePreempt reports whether the current goroutine has been
// asked to yield by the runtime (GC, scheduler). JIT code calls this
// at safepoints via jit.Preempt().
//
//go:nosplit
func userFramePreempt() bool {
	gp := getg()
	if gp == nil {
		return false
	}
	// If called from g0, check the user goroutine.
	if gp.m != nil && gp.m.curg != nil {
		gp = gp.m.curg
	}
	return gp.preempt
}

// Linknames for the runtime/jit package.

//go:linkname jit_registerUserFrameRegion runtime/jit.registerUserFrameRegion
func jit_registerUserFrameRegion(start, end uintptr, unwindMode uint8,
	describe func(pc uintptr) (string, string, int, bool),
	next func(pc, sp uintptr) (uintptr, uintptr, uintptr, bool),
	scanStack func(report func(ptr uintptr)),
) uintptr {
	r := userFrameRegion{
		start:      start,
		end:        end,
		unwindMode: userFrameUnwindMode(unwindMode),
		describe:   describe,
		next:       next,
		scanStack:  scanStack,
	}
	return registerUserFrameRegion(r)
}

//go:linkname jit_unregisterUserFrameRegion runtime/jit.unregisterUserFrameRegion
func jit_unregisterUserFrameRegion(handle uintptr) {
	unregisterUserFrameRegion(handle)
}

//go:linkname jit_userFramePreempt runtime/jit.userFramePreempt
//go:nosplit
func jit_userFramePreempt() bool {
	return userFramePreempt()
}

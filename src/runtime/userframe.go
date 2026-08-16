// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
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

	stackMaps *userFrameStackMapTable

	handle uintptr
}

// userFrameStackMapInput must have the same memory layout as
// runtime/jit.StackMap. It is copied into runtime-owned metadata by Register.
type userFrameStackMapInput struct {
	pcOffset              uintptr
	frameOffset           uintptr
	frameWords            uintptr
	pointerMask           []byte
	hasUnwind             bool
	unwindBaseOffset      uintptr
	unwindBaseDeltaOffset uintptr
	unwindBaseUsesDelta   bool
	callerPCOffset        uintptr
	callerSPOffset        uintptr
	callerBPOffset        uintptr
}

type userFrameStackMap struct {
	pcOffset              uintptr
	frameOffset           uintptr
	frameWords            uintptr
	pointerMask           *byte
	hasUnwind             bool
	unwindBaseOffset      uintptr
	unwindBaseDeltaOffset uintptr
	unwindBaseUsesDelta   bool
	callerPCOffset        uintptr
	callerSPOffset        uintptr
	callerBPOffset        uintptr
}

// userFrameStackMapTable is append-only. Writers serialize through
// userFrameLock. Readers atomically load count and maps without locking. Old
// backing arrays are persistentalloc'd and never freed, so a concurrent grow
// cannot invalidate a reader's snapshot.
type userFrameStackMapTable struct {
	maps     unsafe.Pointer // *userFrameStackMap
	count    uintptr
	capacity uintptr // protected by userFrameLock
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
	if m := findUserFrameStackMap(r, pc); m != nil && m.hasUnwind {
		base := sp + m.unwindBaseOffset
		if base < sp {
			return 0, 0, 0, false
		}
		if m.unwindBaseUsesDelta {
			deltaAddr := sp + m.unwindBaseDeltaOffset
			if deltaAddr < sp {
				return 0, 0, 0, false
			}
			delta := *(*uintptr)(unsafe.Pointer(deltaAddr))
			if base+delta < base {
				return 0, 0, 0, false
			}
			base += delta
		}
		callerPCAddr := base + m.callerPCOffset
		callerSP = base + m.callerSPOffset
		callerBPAddr := base + m.callerBPOffset
		if callerPCAddr < base || callerSP < base || callerBPAddr < base {
			return 0, 0, 0, false
		}
		return *(*uintptr)(unsafe.Pointer(callerPCAddr)), callerSP, *(*uintptr)(unsafe.Pointer(callerBPAddr)), true
	}
	return 0, 0, 0, false
}

// userFramePointerMap resolves one immutable map without calling user code.
// Looking the region up again keeps unwinder pointer-free.
//
//go:nosplit
func userFramePointerMap(pc, sp uintptr) (base, words uintptr, pointerMask *byte, ok bool) {
	r := findUserFrameRegion(pc)
	m := findUserFrameStackMap(r, pc)
	if m == nil || m.frameWords == 0 || sp+m.frameOffset < sp {
		return 0, 0, nil, false
	}
	return sp + m.frameOffset, m.frameWords, m.pointerMask, true
}

//go:nosplit
func findUserFrameStackMap(r *userFrameRegion, pc uintptr) *userFrameStackMap {
	if r == nil || r.stackMaps == nil {
		return nil
	}
	count := atomic.Loaduintptr(&r.stackMaps.count)
	maps := atomic.Loadp(unsafe.Pointer(&r.stackMaps.maps))
	if count == 0 || maps == nil {
		return nil
	}
	offset := pc - r.start
	lo, hi := uintptr(0), count
	for lo < hi {
		mid := (lo + hi) >> 1
		m := (*userFrameStackMap)(add(maps, mid*unsafe.Sizeof(userFrameStackMap{})))
		if m.pcOffset < offset {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo == count {
		return nil
	}
	m := (*userFrameStackMap)(add(maps, lo*unsafe.Sizeof(userFrameStackMap{})))
	if m.pcOffset != offset {
		return nil
	}
	return m
}

func appendUserFrameStackMaps(table *userFrameStackMapTable, start, end uintptr, unwindMode userFrameUnwindMode, input []userFrameStackMapInput) {
	if len(input) == 0 {
		return
	}
	oldCount := atomic.Loaduintptr(&table.count)
	newCount := oldCount + uintptr(len(input))
	if newCount < oldCount {
		throw("registerUserFrameRegion: too many stack maps")
	}
	maps := atomic.Loadp(unsafe.Pointer(&table.maps))
	if newCount > table.capacity {
		newCapacity := table.capacity * 2
		if newCapacity < 16 {
			newCapacity = 16
		}
		for newCapacity < newCount {
			if newCapacity > ^uintptr(0)/2 {
				throw("registerUserFrameRegion: too many stack maps")
			}
			newCapacity *= 2
		}
		bytes := newCapacity * unsafe.Sizeof(userFrameStackMap{})
		if bytes/unsafe.Sizeof(userFrameStackMap{}) != newCapacity {
			throw("registerUserFrameRegion: stack map table too large")
		}
		newMaps := persistentalloc(bytes, unsafe.Alignof(userFrameStackMap{}), &memstats.other_sys)
		if oldCount != 0 {
			memmove(newMaps, maps, oldCount*unsafe.Sizeof(userFrameStackMap{}))
		}
		maps = newMaps
		table.capacity = newCapacity
		atomic.StorepNoWB(unsafe.Pointer(&table.maps), maps)
	}
	var previousPC uintptr
	if oldCount != 0 {
		previous := (*userFrameStackMap)(add(maps, (oldCount-1)*unsafe.Sizeof(userFrameStackMap{})))
		previousPC = previous.pcOffset
	}
	for i := range input {
		in := &input[i]
		if in.pcOffset >= end-start || ((oldCount != 0 || i > 0) && in.pcOffset <= previousPC) {
			throw("registerUserFrameRegion: unordered or invalid stack map PC")
		}
		if unwindMode != userFrameUnwindStop && !in.hasUnwind {
			throw("registerUserFrameRegion: declarative unwind required at every safepoint")
		}
		if in.frameWords > uintptr(^uint32(0)>>1) {
			throw("registerUserFrameRegion: stack map too large")
		}
		frameBytes := in.frameWords * goarch.PtrSize
		if frameBytes/goarch.PtrSize != in.frameWords || in.frameOffset+frameBytes < in.frameOffset {
			throw("registerUserFrameRegion: invalid stack map range")
		}
		maskBytes := (in.frameWords + 7) / 8
		if maskBytes < in.frameWords/8 || uintptr(len(in.pointerMask)) < maskBytes {
			throw("registerUserFrameRegion: short stack map bitmap")
		}
		var mask *byte
		if maskBytes != 0 {
			mask = (*byte)(persistentalloc(maskBytes, 1, &memstats.other_sys))
			memmove(unsafe.Pointer(mask), unsafe.Pointer(&in.pointerMask[0]), maskBytes)
		}
		out := (*userFrameStackMap)(add(maps, (oldCount+uintptr(i))*unsafe.Sizeof(userFrameStackMap{})))
		*out = userFrameStackMap{
			pcOffset:              in.pcOffset,
			frameOffset:           in.frameOffset,
			frameWords:            in.frameWords,
			pointerMask:           mask,
			hasUnwind:             in.hasUnwind,
			unwindBaseOffset:      in.unwindBaseOffset,
			unwindBaseDeltaOffset: in.unwindBaseDeltaOffset,
			unwindBaseUsesDelta:   in.unwindBaseUsesDelta,
			callerPCOffset:        in.callerPCOffset,
			callerSPOffset:        in.callerSPOffset,
			callerBPOffset:        in.callerBPOffset,
		}
		previousPC = in.pcOffset
	}
	// Publish only after every entry and bitmap is fully initialized.
	atomic.Storeuintptr(&table.count, newCount)
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
	stackMaps []userFrameStackMapInput,
) uintptr {
	if start >= end {
		throw("registerUserFrameRegion: invalid address range")
	}
	if userFrameUnwindMode(unwindMode) > userFrameUnwindDeclare {
		throw("registerUserFrameRegion: invalid unwind mode")
	}
	table := (*userFrameStackMapTable)(persistentalloc(unsafe.Sizeof(userFrameStackMapTable{}), unsafe.Alignof(userFrameStackMapTable{}), &memstats.other_sys))
	appendUserFrameStackMaps(table, start, end, userFrameUnwindMode(unwindMode), stackMaps)
	r := userFrameRegion{
		start:      start,
		end:        end,
		unwindMode: userFrameUnwindMode(unwindMode),
		describe:   describe,
		stackMaps:  table,
	}
	return registerUserFrameRegion(r)
}

//go:linkname jit_addUserFrameStackMaps runtime/jit.addUserFrameStackMaps
func jit_addUserFrameStackMaps(handle uintptr, stackMaps []userFrameStackMapInput) {
	lock(&userFrameLock)
	snap := loadUserFrameSnapshot()
	if snap != nil {
		for i := 0; i < snap.count; i++ {
			r := &snap.regions[i]
			if r.handle == handle {
				appendUserFrameStackMaps(r.stackMaps, r.start, r.end, r.unwindMode, stackMaps)
				unlock(&userFrameLock)
				return
			}
		}
	}
	unlock(&userFrameLock)
	throw("addUserFrameStackMaps: handle not found")
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

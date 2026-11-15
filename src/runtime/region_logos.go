// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/goexperiment"
	"internal/runtime/atomic"
	"internal/runtime/math"
	"internal/runtime/sys"
	"unsafe"
)

const logosRegionMaxBytes = logosArenaChunkBytes - 1024

type logosRegion struct {
	_     sys.NotInHeap
	chunk *sideStackChunk
	gen   uintptr
	prev  *logosRegion
	owner atomic.Pointer[g]
	depth uintptr
	next  *logosRegion
}

func logosRegionValid(h unsafe.Pointer, gen uintptr) bool {
	if h == nil {
		return false
	}
	rec := (*logosRegion)(h)
	return atomic.Loaduintptr(&rec.gen) == gen && rec.owner.Load() == getg()
}

func logosRegionOpen() (h unsafe.Pointer, gen uintptr) {
	if !goexperiment.LogosArena {
		return nil, 0
	}
	gp := getg()
	if !logosArenaUsable(gp) {
		return nil, 0
	}
	rec := gp.logosRegionFree
	if rec != nil {
		gp.logosRegionFree = rec.next
		rec.next = nil
	} else {
		rec = logosRegionTakeP()
		if rec == nil {
			rec = (*logosRegion)(persistentalloc(unsafe.Sizeof(logosRegion{}), goarch.PtrSize, &memstats.other_sys))
			if rec == nil {
				return nil, 0
			}
		}
	}
	mp := acquirem()
	sp := sys.GetCallerSP()
	rec.chunk = nil
	rec.owner.Store(gp)
	rec.depth = gp.stack.hi - sp
	rec.prev = gp.logosRegions
	gp.logosRegions = rec
	gen = atomic.Loaduintptr(&rec.gen)
	releasem(mp)
	return unsafe.Pointer(rec), gen
}

func logosRegionClose(h unsafe.Pointer, gen uintptr) {
	if h == nil {
		return
	}
	gp := getg()
	rec := (*logosRegion)(h)
	if gp.logosRegions != rec {
		throw("logosRegionClose: region is not the innermost open region")
	}
	if atomic.Loaduintptr(&rec.gen) != gen {
		throw("logosRegionClose: stale region handle")
	}
	logosRegionReleaseChunks(rec)
	atomic.Xadduintptr(&rec.gen, 1)
	gp.logosRegions = rec.prev
	rec.prev = nil
	rec.next = gp.logosRegionFree
	gp.logosRegionFree = rec
}

func logosRegionReleaseChunks(rec *logosRegion) {
	c := rec.chunk
	for c != nil {
		next := c.next
		c.popScan(c.scanBase)
		if raceenabled && c.noscanCur > c.noscanBase {
			racefree(unsafe.Pointer(c.noscanBase), c.noscanCur-c.noscanBase)
		}
		c.reset()
		logosArenaRelease(c)
		c = next
	}
	rec.chunk = nil
}

func logosRegionBump(h unsafe.Pointer, gen uintptr, size uintptr) unsafe.Pointer {
	if raceenabled || !logosRegionValid(h, gen) {
		return nil
	}
	rec := (*logosRegion)(h)
	c := rec.chunk
	if c == nil {
		return nil
	}
	cur := alignUp(c.noscanCur, logosArenaUnit/2)
	if cur+size > c.scanCur {
		return nil
	}
	c.noscanCur = cur + size
	return unsafe.Pointer(cur)
}

func logosRegionAllocNoscan(h unsafe.Pointer, gen uintptr, size, align uintptr) unsafe.Pointer {
	if size == 0 {
		return unsafe.Pointer(&zerobase)
	}
	if size > logosRegionMaxBytes || !logosRegionValid(h, gen) || !logosArenaNoscanFits(size, align) {
		return mallocgc(size, nil, true)
	}
	rec := (*logosRegion)(h)
	p := rec.chunk.tryNoscan(size, align)
	if p == nil {
		p = logosRegionLinkedNoscan(rec, size, align)
	}
	if p == nil {
		return mallocgc(size, nil, true)
	}
	memclrNoHeapPointers(p, size)
	if raceenabled {
		racemalloc(p, size)
	}
	return p
}

func logosRegionAllocScan(h unsafe.Pointer, gen uintptr, typ *_type, size uintptr) unsafe.Pointer {
	if size == 0 {
		return unsafe.Pointer(&zerobase)
	}
	if typ == nil || !typ.Pointers() {
		throw("logosRegionAllocScan: type without pointers")
	}
	if size > logosRegionMaxBytes || !logosRegionValid(h, gen) || !logosArenaScanFits(typ, size) {
		return mallocgc(size, typ, true)
	}
	rec := (*logosRegion)(h)
	p := rec.chunk.tryScan(typ, size)
	if p == nil {
		p = logosRegionLinkedScan(rec, typ, size)
	}
	if p == nil {
		return mallocgc(size, typ, true)
	}
	if raceenabled {
		racemalloc(p, size)
	}
	return p
}

func logosRegionLinkedNoscan(rec *logosRegion, size, align uintptr) unsafe.Pointer {
	c := logosRegionChunkTake(rec)
	if c == nil {
		return nil
	}
	p := c.tryNoscan(size, align)
	if p == nil {
		throw("logosRegionAllocNoscan: fresh chunk rejected a fitting object")
	}
	return p
}

func logosRegionLinkedScan(rec *logosRegion, typ *_type, size uintptr) unsafe.Pointer {
	c := logosRegionChunkTake(rec)
	if c == nil {
		return nil
	}
	p := c.tryScan(typ, size)
	if p == nil {
		throw("logosRegionAllocScan: fresh chunk rejected a fitting object")
	}
	return p
}

func logosRegionChunkTake(rec *logosRegion) *sideStackChunk {
	c := logosArenaTake()
	if c == nil {
		return nil
	}
	if raceenabled {
		racemalloc(unsafe.Pointer(c), logosArenaChunkBytes)
	}
	c.next = rec.chunk
	rec.chunk = c
	return c
}

func logosRegionMakeSliceNoscan(h unsafe.Pointer, gen uintptr, typ *_type, len64, cap64 int64) unsafe.Pointer {
	mem := logosRegionSliceMem(typ, len64, cap64)
	return logosRegionAllocNoscan(h, gen, mem, logosArenaAlign(typ))
}

func logosRegionMakeSliceScan(h unsafe.Pointer, gen uintptr, typ *_type, len64, cap64 int64) unsafe.Pointer {
	mem := logosRegionSliceMem(typ, len64, cap64)
	return logosRegionAllocScan(h, gen, typ, mem)
}

func logosRegionSliceMem(typ *_type, len64, cap64 int64) uintptr {
	length := int(len64)
	if int64(length) != len64 {
		panicmakeslicelen()
	}
	capacity := int(cap64)
	if int64(capacity) != cap64 {
		panicmakeslicecap()
	}
	mem, overflow := math.MulUintptr(typ.Size_, uintptr(capacity))
	if overflow || mem > maxAlloc || length < 0 || length > capacity {
		mem, overflow := math.MulUintptr(typ.Size_, uintptr(length))
		if overflow || mem > maxAlloc || length < 0 {
			panicmakeslicelen()
		}
		panicmakeslicecap()
	}
	return mem
}

type logosRegionGrow struct {
	h        unsafe.Pointer
	gen      uintptr
	oldPtr   unsafe.Pointer
	newLen   int
	oldCap   int
	num      int
	et       *_type
	noscan   bool
	callerpc uintptr
}

func logosRegionGrowSliceNoscan(h unsafe.Pointer, gen uintptr, oldPtr unsafe.Pointer, newLen, oldCap, num int, et *_type) slice {
	return logosRegionGrowSliceOf(&logosRegionGrow{h, gen, oldPtr, newLen, oldCap, num, et, true, sys.GetCallerPC()})
}

func logosRegionGrowSliceScan(h unsafe.Pointer, gen uintptr, oldPtr unsafe.Pointer, newLen, oldCap, num int, et *_type) slice {
	return logosRegionGrowSliceOf(&logosRegionGrow{h, gen, oldPtr, newLen, oldCap, num, et, false, sys.GetCallerPC()})
}

func logosRegionGrowSliceOf(q *logosRegionGrow) slice {
	h, gen, oldPtr, newLen, oldCap, num, et := q.h, q.gen, q.oldPtr, q.newLen, q.oldCap, q.num, q.et
	oldLen := newLen - num
	if raceenabled {
		callerpc := q.callerpc
		racereadrangepc(oldPtr, uintptr(oldLen*int(et.Size_)), callerpc, abi.FuncPCABIInternal(logosRegionGrowSliceOf))
	}
	if msanenabled {
		msanread(oldPtr, uintptr(oldLen*int(et.Size_)))
	}
	if asanenabled {
		asanread(oldPtr, uintptr(oldLen*int(et.Size_)))
	}
	if newLen < 0 {
		panic(errorString("growslice: len out of range"))
	}
	if et.Size_ == 0 {
		return slice{unsafe.Pointer(&zerobase), newLen, newLen}
	}

	newcap := nextslicecap(newLen, oldCap)

	var overflow bool
	var lenmem, newlenmem, capmem uintptr
	noscan := q.noscan
	switch {
	case et.Size_ == 1:
		lenmem = uintptr(oldLen)
		newlenmem = uintptr(newLen)
		capmem = roundupsize(uintptr(newcap), noscan)
		overflow = uintptr(newcap) > maxAlloc
		newcap = int(capmem)
	case et.Size_ == goarch.PtrSize:
		lenmem = uintptr(oldLen) * goarch.PtrSize
		newlenmem = uintptr(newLen) * goarch.PtrSize
		capmem = roundupsize(uintptr(newcap)*goarch.PtrSize, noscan)
		overflow = uintptr(newcap) > maxAlloc/goarch.PtrSize
		newcap = int(capmem / goarch.PtrSize)
	case isPowerOfTwo(et.Size_):
		var shift uintptr
		if goarch.PtrSize == 8 {
			shift = uintptr(sys.TrailingZeros64(uint64(et.Size_))) & 63
		} else {
			shift = uintptr(sys.TrailingZeros32(uint32(et.Size_))) & 31
		}
		lenmem = uintptr(oldLen) << shift
		newlenmem = uintptr(newLen) << shift
		capmem = roundupsize(uintptr(newcap)<<shift, noscan)
		overflow = uintptr(newcap) > (maxAlloc >> shift)
		newcap = int(capmem >> shift)
		capmem = uintptr(newcap) << shift
	default:
		lenmem = uintptr(oldLen) * et.Size_
		newlenmem = uintptr(newLen) * et.Size_
		capmem, overflow = math.MulUintptr(et.Size_, uintptr(newcap))
		capmem = roundupsize(capmem, noscan)
		newcap = int(capmem / et.Size_)
		capmem = uintptr(newcap) * et.Size_
	}
	if overflow || capmem > maxAlloc {
		panic(errorString("growslice: len out of range"))
	}

	var p unsafe.Pointer
	if !noscan {
		p = logosRegionAllocScan(h, gen, et, capmem)
		if lenmem > 0 && writeBarrier.enabled {
			bulkBarrierPreWriteSrcOnly(uintptr(p), uintptr(oldPtr), lenmem-et.Size_+et.PtrBytes, et)
		}
	} else {
		p = logosRegionAllocNoscan(h, gen, capmem, logosArenaAlign(et))
		memclrNoHeapPointers(add(p, newlenmem), capmem-newlenmem)
	}
	memmove(p, oldPtr, lenmem)
	return slice{p, newLen, newcap}
}

func logosRegionUnwind(gp *g, sp uintptr) {
	rec := gp.logosRegions
	if rec == nil {
		return
	}
	threshold := gp.stack.hi - sp
	for rec != nil && rec.depth >= threshold {
		top := rec
		logosRegionReleaseChunks(top)
		atomic.Xadduintptr(&top.gen, 1)
		rec = top.prev
		top.prev = nil
		top.next = gp.logosRegionFree
		gp.logosRegionFree = top
	}
	gp.logosRegions = rec
}

//go:nowritebarrier
func logosRegionScan(gp *g, gcw *gcWork, state *stackScanState) {
	for rec := gp.logosRegions; rec != nil; rec = rec.prev {
		for c := rec.chunk; c != nil; c = c.next {
			logosArenaScanChunk(c, gcw, state)
		}
	}
}

func logosRegionPark(gp *g) {
	for rec := gp.logosRegions; rec != nil; rec = rec.prev {
		var prev *sideStackChunk
		c := rec.chunk
		for c != nil {
			next := c.next
			if c.empty() {
				if prev == nil {
					rec.chunk = next
				} else {
					prev.next = next
				}
				c.popScan(c.scanBase)
				c.reset()
				logosArenaRelease(c)
			} else {
				prev = c
			}
			c = next
		}
	}
}

func logosRegionExit(gp *g) {
	var list *logosRegion
	rec := gp.logosRegions
	for rec != nil {
		next := rec.prev
		logosRegionReleaseChunks(rec)
		atomic.Xadduintptr(&rec.gen, 1)
		rec.owner.Store(nil)
		rec.prev = nil
		rec.next = list
		list = rec
		rec = next
	}
	if gp.logosRegionFree != nil {
		tail := list
		if tail == nil {
			list = gp.logosRegionFree
		} else {
			for tail.next != nil {
				tail = tail.next
			}
			tail.next = gp.logosRegionFree
		}
	}
	gp.logosRegions = nil
	gp.logosRegionFree = nil
	if list != nil {
		mp := acquirem()
		pp := mp.p.ptr()
		tail := list
		for tail.next != nil {
			tail = tail.next
		}
		tail.next = pp.logosRegionPool
		pp.logosRegionPool = list
		releasem(mp)
	}
}

func logosRegionTakeP() *logosRegion {
	mp := acquirem()
	pp := mp.p.ptr()
	rec := pp.logosRegionPool
	if rec != nil {
		pp.logosRegionPool = rec.next
		rec.next = nil
	}
	releasem(mp)
	return rec
}

//go:linkname logosRegionBytes
func logosRegionBytes() int {
	gp := getg()
	n := uintptr(0)
	for rec := gp.logosRegions; rec != nil; rec = rec.prev {
		for c := rec.chunk; c != nil; c = c.next {
			n += c.noscanCur - c.noscanBase
			n += c.scanBase - c.scanCur
		}
	}
	return int(n)
}

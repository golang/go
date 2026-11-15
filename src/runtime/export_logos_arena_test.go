// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime

import "unsafe"

type LogosArenaMark = logosArenaMark

const (
	LogosArenaChunkBytes      = logosArenaChunkBytes
	LogosArenaCapacity        = logosArenaChunkBytes - logosArenaHeaderBytes
	LogosArenaUnit            = logosArenaUnit
	LogosArenaScanHeaderBytes = logosArenaScanHeaderBytes
)

type LogosArenaInfo struct {
	Chunks     int
	NoscanUsed uintptr
	ScanUsed   uintptr
	Free       uintptr
}

func logosEfaceType(elem any) *_type {
	return efaceOf(&elem)._type
}

func LogosArenaAlloc(elem any, size uintptr) unsafe.Pointer {
	return logosArenaAlloc(logosEfaceType(elem), size)
}

func LogosArenaMakeSlice(elem any, len64, cap64 int64) unsafe.Pointer {
	return logosArenaMakeSlice(logosEfaceType(elem), len64, cap64)
}

const LogosArenaMaxAlloc uintptr = maxAlloc

func LogosArenaAllocScan(elem any, size uintptr) unsafe.Pointer {
	return logosArenaAllocScan(logosEfaceType(elem), size)
}

func LogosArenaSave() LogosArenaMark {
	return logosArenaSave()
}

func LogosArenaRestore(m LogosArenaMark) {
	logosArenaRestore(m)
}

func LogosArenaMarkSave() (chunk, noscan, scan uintptr) {
	return logosArenaMarkSave()
}

func LogosArenaMarkRestore(chunk, noscan, scan uintptr) {
	logosArenaMarkRestore(chunk, noscan, scan)
}

func LogosArenaRetSet(ok bool) {
	logosArenaRetSet(ok)
}

func LogosArenaRetTake() bool {
	return logosArenaRetTake()
}

var LogosArenaFieldOffset = unsafe.Offsetof(g{}.logosArena)

func LogosArenaStat() LogosArenaInfo {
	var info LogosArenaInfo
	c := getg().logosArena
	for n := c; n != nil; n = n.next {
		info.Chunks++
	}
	if c != nil {
		info.NoscanUsed = c.noscanCur - c.noscanBase
		info.ScanUsed = c.scanBase - c.scanCur
		info.Free = c.scanCur - c.noscanCur
	}
	return info
}

type LogosArenaChunk struct {
	Addr       uintptr
	NoscanUsed uintptr
	ScanUsed   uintptr
}

func LogosArenaGoid() uint64 {
	return getg().goid
}

func LogosArenaChainOf(goid uint64, buf []LogosArenaChunk) (n int, waiting bool) {
	stw := stopTheWorld(stwForTestCountPagesInUse)
	forEachG(func(gp *g) {
		if gp.goid != goid {
			return
		}
		waiting = readgstatus(gp)&^_Gscan == _Gwaiting
		for c := gp.logosArena; c != nil; c = c.next {
			if n < len(buf) {
				buf[n] = LogosArenaChunk{
					Addr:       uintptr(unsafe.Pointer(c)),
					NoscanUsed: c.noscanCur - c.noscanBase,
					ScanUsed:   c.scanBase - c.scanCur,
				}
			}
			n++
		}
	})
	startTheWorld(stw)
	return n, waiting
}

func LogosArenaEmptyChunk(addr uintptr) bool {
	for c := getg().logosArena; c != nil; c = c.next {
		if uintptr(unsafe.Pointer(c)) == addr {
			c.popScan(c.scanBase)
			c.noscanCur = c.noscanBase
			return true
		}
	}
	return false
}

func LogosArenaCounts() (total, leased, pooled int) {
	stw := stopTheWorld(stwForTestCountPagesInUse)
	total = int(logosArenaChunkTotal.Load())
	forEachG(func(gp *g) {
		for c := gp.logosArena; c != nil; c = c.next {
			leased++
		}
	})
	for _, pp := range allp {
		pooled += int(pp.logosArenaPoolN)
	}
	startTheWorld(stw)
	return
}

func LogosArenaPoolCounts() []int {
	stw := stopTheWorld(stwForTestCountPagesInUse)
	counts := make([]int, len(allp))
	for i, pp := range allp {
		counts[i] = int(pp.logosArenaPoolN)
	}
	startTheWorld(stw)
	return counts
}

func LogosArenaSetLimit(n int64) int64 {
	return logosArenaChunkLimit.Swap(n)
}

func LogosArenaInChunk(p unsafe.Pointer) bool {
	return inLogosArenaChunk(uintptr(p))
}

func LogosArenaSpanKind(p unsafe.Pointer) (manual, flagged bool) {
	s := spanOf(uintptr(p))
	if s == nil {
		return false, false
	}
	return s.state.get() == mSpanManual, s.isLogosArenaChunk
}

func LogosArenaType(elem any) unsafe.Pointer {
	return unsafe.Pointer(logosEfaceType(elem))
}

func LogosArenaScanHeader(p unsafe.Pointer) (typ unsafe.Pointer, size uintptr) {
	h := (*logosArenaScanHeader)(unsafe.Pointer(uintptr(p) - logosArenaScanHeaderBytes))
	return unsafe.Pointer(h.typ), h.size
}

func LogosArenaIsGoPointer(p unsafe.Pointer) bool {
	return cgoIsGoPointer(p)
}

func LogosArenaCgoCheckPointer(ptr, arg any) {
	cgoCheckPointer(ptr, arg)
}

func LogosArenaCgoCheckResult(val any) {
	cgoCheckResult(val)
}

func LogosArenaCgoCheckPtrWrite(dst *unsafe.Pointer, src unsafe.Pointer) {
	cgoCheckPtrWrite(dst, src)
}

func LogosArenaCgoCheckMemmove(elem any, dst, src unsafe.Pointer) {
	typ := logosEfaceType(elem)
	cgoCheckMemmove2(typ, dst, src, 0, typ.Size_)
}

const LogosRegionMaxBytes uintptr = logosRegionMaxBytes

func LogosRegionOpen() (h unsafe.Pointer, gen uintptr) {
	return logosRegionOpen()
}

func LogosRegionClose(h unsafe.Pointer, gen uintptr) {
	logosRegionClose(h, gen)
}

func LogosRegionBump(h unsafe.Pointer, gen uintptr, size uintptr) unsafe.Pointer {
	return logosRegionBump(h, gen, size)
}

func LogosRegionAllocNoscan(h unsafe.Pointer, gen uintptr, size, align uintptr) unsafe.Pointer {
	return logosRegionAllocNoscan(h, gen, size, align)
}

func LogosRegionAllocScan(h unsafe.Pointer, gen uintptr, elem any, size uintptr) unsafe.Pointer {
	return logosRegionAllocScan(h, gen, logosEfaceType(elem), size)
}

func LogosRegionMakeSlice(h unsafe.Pointer, gen uintptr, elem any, len64, cap64 int64) unsafe.Pointer {
	typ := logosEfaceType(elem)
	if typ.Pointers() {
		return logosRegionMakeSliceScan(h, gen, typ, len64, cap64)
	}
	return logosRegionMakeSliceNoscan(h, gen, typ, len64, cap64)
}

func LogosRegionGrowSlice(h unsafe.Pointer, gen uintptr, oldPtr unsafe.Pointer, newLen, oldCap, num int, elem any) (array unsafe.Pointer, capacity int) {
	typ := logosEfaceType(elem)
	var s slice
	if typ.Pointers() {
		s = logosRegionGrowSliceScan(h, gen, oldPtr, newLen, oldCap, num, typ)
	} else {
		s = logosRegionGrowSliceNoscan(h, gen, oldPtr, newLen, oldCap, num, typ)
	}
	return s.array, s.cap
}

func LogosHeapGrowSlice(oldPtr unsafe.Pointer, newLen, oldCap, num int, elem any) (array unsafe.Pointer, capacity int) {
	s := growslice(oldPtr, newLen, oldCap, num, logosEfaceType(elem))
	return s.array, s.cap
}

func LogosRegionBytes() int {
	return logosRegionBytes()
}

func LogosTypePointers(elem any) bool {
	return logosEfaceType(elem).Pointers()
}

func LogosRegionChunks() []uintptr {
	gp := getg()
	n := 0
	for rec := gp.logosRegions; rec != nil; rec = rec.prev {
		for c := rec.chunk; c != nil; c = c.next {
			n++
		}
	}
	out := make([]uintptr, 0, n)
	for rec := gp.logosRegions; rec != nil; rec = rec.prev {
		for c := rec.chunk; c != nil; c = c.next {
			out = append(out, uintptr(unsafe.Pointer(c)))
		}
	}
	return out
}

func LogosRegionChunkTotal() int {
	stw := stopTheWorld(stwForTestCountPagesInUse)
	n := 0
	forEachG(func(gp *g) {
		for rec := gp.logosRegions; rec != nil; rec = rec.prev {
			for c := rec.chunk; c != nil; c = c.next {
				n++
			}
		}
	})
	startTheWorld(stw)
	return n
}

func LogosRegionEmptyChunk(addr uintptr) bool {
	for rec := getg().logosRegions; rec != nil; rec = rec.prev {
		for c := rec.chunk; c != nil; c = c.next {
			if uintptr(unsafe.Pointer(c)) == addr {
				c.popScan(c.scanBase)
				c.noscanCur = c.noscanBase
				return true
			}
		}
	}
	return false
}

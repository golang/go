// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"internal/goexperiment"
	"internal/runtime/atomic"
	"internal/runtime/math"
	"internal/runtime/sys"
	"unsafe"
)

const (
	logosArenaChunkBytes = 64 << 10
	logosArenaChunkPages = logosArenaChunkBytes / pageSize

	logosArenaUnit = 2 * goarch.PtrSize

	logosArenaHeaderBytes = (unsafe.Sizeof(sideStackChunk{}) + logosArenaUnit - 1) &^ (logosArenaUnit - 1)

	logosArenaScanHeaderBytes = unsafe.Sizeof(logosArenaScanHeader{})

	logosArenaDefaultLimit = 1 << 14
)

const (
	_ = uint(4*goarch.PtrSize - unsafe.Offsetof(g{}.logosArena))
	_ = uint(unsafe.Offsetof(g{}.logosArena) - 4*goarch.PtrSize)
	_ = uint(2*goarch.PtrSize - unsafe.Offsetof(sideStackChunk{}.noscanCur))
	_ = uint(unsafe.Offsetof(sideStackChunk{}.noscanCur) - 2*goarch.PtrSize)
	_ = uint(3*goarch.PtrSize - unsafe.Offsetof(sideStackChunk{}.scanCur))
	_ = uint(unsafe.Offsetof(sideStackChunk{}.scanCur) - 3*goarch.PtrSize)
)

type sideStackChunk struct {
	_          sys.NotInHeap
	next       *sideStackChunk
	noscanBase uintptr
	noscanCur  uintptr
	scanCur    uintptr
	scanBase   uintptr
}

type logosArenaScanHeader struct {
	typ  *_type
	size uintptr
}

type logosArenaMark struct {
	chunk  *sideStackChunk
	noscan uintptr
	scan   uintptr
}

var (
	logosArenaChunkTotal atomic.Int64
	logosArenaChunkLimit atomic.Int64
)

func init() {
	logosArenaChunkLimit.Store(logosArenaDefaultLimit)
}

func (c *sideStackChunk) reset() {
	base := uintptr(unsafe.Pointer(c))
	c.next = nil
	c.noscanBase = base + logosArenaHeaderBytes
	c.noscanCur = c.noscanBase
	c.scanBase = base + logosArenaChunkBytes
	c.scanCur = c.scanBase
}

func (c *sideStackChunk) empty() bool {
	return c.noscanCur == c.noscanBase && c.scanCur == c.scanBase
}

func (c *sideStackChunk) tryNoscan(size, align uintptr) unsafe.Pointer {
	if c == nil {
		return nil
	}
	cur := alignUp(c.noscanCur, align)
	if cur+size > c.scanCur {
		return nil
	}
	c.noscanCur = cur + size
	return unsafe.Pointer(cur)
}

func (c *sideStackChunk) tryScan(typ *_type, size uintptr) unsafe.Pointer {
	if c == nil {
		return nil
	}
	stride := alignUp(size, logosArenaUnit) + logosArenaScanHeaderBytes
	if stride > c.scanCur-c.noscanCur {
		return nil
	}
	hdr := c.scanCur - stride
	memclrNoHeapPointers(unsafe.Pointer(hdr), stride)
	*(*logosArenaScanHeader)(unsafe.Pointer(hdr)) = logosArenaScanHeader{typ: typ, size: size}
	c.scanCur = hdr
	return unsafe.Pointer(hdr + logosArenaScanHeaderBytes)
}

func (c *sideStackChunk) popScan(to uintptr) {
	old := c.scanCur
	if to < old || to > c.scanBase {
		throw("logosArena: scan watermark restored out of range")
	}
	c.scanCur = to
	if raceenabled && to > old {
		racefree(unsafe.Pointer(old), to-old)
	}
	memclrNoHeapPointers(unsafe.Pointer(old), to-old)
}

func logosArenaAlign(typ *_type) uintptr {
	if typ == nil || uintptr(typ.Align_) < logosArenaUnit/2 {
		return logosArenaUnit / 2
	}
	return uintptr(typ.Align_)
}

func logosArenaNoscanFits(size, align uintptr) bool {
	return align <= pageSize && size <= logosArenaChunkBytes-alignUp(logosArenaHeaderBytes, align)
}

func logosArenaScanFits(typ *_type, size uintptr) bool {
	return uintptr(typ.Align_) <= logosArenaUnit &&
		alignUp(size, logosArenaUnit) <= logosArenaChunkBytes-logosArenaHeaderBytes-logosArenaScanHeaderBytes
}

func logosArenaUsable(gp *g) bool {
	return gp.m.curg == gp
}

func logosArenaAlloc(typ *_type, size uintptr) unsafe.Pointer {
	if size == 0 {
		return unsafe.Pointer(&zerobase)
	}
	if typ != nil && typ.Pointers() {
		throw("logosArenaAlloc: type with pointers")
	}
	gp := getg()
	align := logosArenaAlign(typ)
	if !logosArenaUsable(gp) || !logosArenaNoscanFits(size, align) {
		return mallocgc(size, typ, true)
	}
	p := gp.logosArena.tryNoscan(size, align)
	if p == nil {
		p = logosArenaNoscanLinked(gp, size, align)
	}
	if p == nil {
		return mallocgc(size, typ, true)
	}
	memclrNoHeapPointers(p, size)
	if raceenabled {
		racemalloc(p, size)
	}
	return p
}

func logosArenaMakeSlice(typ *_type, len64, cap64 int64) unsafe.Pointer {
	len := int(len64)
	if int64(len) != len64 {
		panicmakeslicelen()
	}

	cap := int(cap64)
	if int64(cap) != cap64 {
		panicmakeslicecap()
	}

	mem, overflow := math.MulUintptr(typ.Size_, uintptr(cap))
	if overflow || mem > maxAlloc || len < 0 || len > cap {
		mem, overflow := math.MulUintptr(typ.Size_, uintptr(len))
		if overflow || mem > maxAlloc || len < 0 {
			panicmakeslicelen()
		}
		panicmakeslicecap()
	}

	if typ != nil && typ.Pointers() {
		return logosArenaAllocScan(typ, mem)
	}
	return logosArenaAlloc(typ, mem)
}

func logosArenaNoscanLinked(gp *g, size, align uintptr) unsafe.Pointer {
	c := logosArenaPush(gp)
	if c == nil {
		return nil
	}
	p := c.tryNoscan(size, align)
	if p == nil {
		throw("logosArenaAlloc: fresh chunk rejected a fitting object")
	}
	return p
}

func logosArenaScanLinked(gp *g, typ *_type, size uintptr) unsafe.Pointer {
	c := logosArenaPush(gp)
	if c == nil {
		return nil
	}
	p := c.tryScan(typ, size)
	if p == nil {
		throw("logosArenaAllocScan: fresh chunk rejected a fitting object")
	}
	return p
}

func logosArenaAllocScan(typ *_type, size uintptr) unsafe.Pointer {
	if size == 0 {
		return unsafe.Pointer(&zerobase)
	}
	if typ == nil || !typ.Pointers() {
		throw("logosArenaAllocScan: type without pointers")
	}
	gp := getg()
	if !logosArenaUsable(gp) || !logosArenaScanFits(typ, size) {
		return mallocgc(size, typ, true)
	}
	p := gp.logosArena.tryScan(typ, size)
	if p == nil {
		p = logosArenaScanLinked(gp, typ, size)
	}
	if p == nil {
		return mallocgc(size, typ, true)
	}
	if raceenabled {
		racemalloc(p, size)
	}
	return p
}

func logosArenaSave() logosArenaMark {
	c := getg().logosArena
	if c == nil || (c.next == nil && c.empty()) {
		return logosArenaMark{}
	}
	return logosArenaMark{chunk: c, noscan: c.noscanCur, scan: c.scanCur}
}

func logosArenaRestore(m logosArenaMark) {
	gp := getg()
	c := gp.logosArena
	if c == nil {
		return
	}
	for c != m.chunk && c.next != nil {
		gp.logosArena = c.next
		c.popScan(c.scanBase)
		c.reset()
		logosArenaRelease(c)
		c = gp.logosArena
	}
	if m.chunk == nil {
		m = logosArenaMark{chunk: c, noscan: c.noscanBase, scan: c.scanBase}
	}
	if c != m.chunk || m.noscan > c.noscanCur {
		throw("logosArenaRestore: mark does not match the arena chain")
	}
	c.popScan(m.scan)
	if raceenabled && m.noscan < c.noscanCur {
		racefree(unsafe.Pointer(m.noscan), c.noscanCur-m.noscan)
	}
	c.noscanCur = m.noscan
}

func logosArenaMarkSave() (chunk, noscan, scan uintptr) {
	m := logosArenaSave()
	return uintptr(unsafe.Pointer(m.chunk)), m.noscan, m.scan
}

//go:nocheckptr
func logosArenaMarkRestore(chunk, noscan, scan uintptr) {
	logosArenaRestore(logosArenaMark{chunk: (*sideStackChunk)(unsafe.Pointer(chunk)), noscan: noscan, scan: scan})
}

func logosArenaRetSet(ok bool) {
	getg().logosArenaRetOK = ok
}

func logosArenaRetTake() bool {
	gp := getg()
	ok := gp.logosArenaRetOK
	gp.logosArenaRetOK = false
	return ok
}

//go:nowritebarrier
func logosArenaScanStack(gp *g, gcw *gcWork, state *stackScanState) {
	for c := gp.logosArena; c != nil; c = c.next {
		logosArenaScanChunk(c, gcw, state)
	}
}

//go:nowritebarrier
func logosArenaScanChunk(c *sideStackChunk, gcw *gcWork, state *stackScanState) {
	for hdrAddr := c.scanCur; hdrAddr < c.scanBase; {
		hdr := (*logosArenaScanHeader)(unsafe.Pointer(hdrAddr))
		obj := hdrAddr + logosArenaScanHeaderBytes
		hdrAddr = obj + alignUp(hdr.size, logosArenaUnit)
		if hdr.typ != nil && hdr.size != 0 {
			logosArenaScanObject(hdr.typ, obj, hdr.size, gcw, state)
		}
	}
}

//go:nowritebarrier
func logosArenaScanObject(typ *_type, obj, size uintptr, gcw *gcWork, state *stackScanState) {
	limit := obj + size
	tp := typePointers{elem: obj, addr: obj, mask: readUintptr(getGCMask(typ)), typ: typ}.fastForward(0, limit)
	for {
		var addr uintptr
		if tp, addr = tp.nextFast(); addr == 0 {
			if tp, addr = tp.next(limit); addr == 0 {
				return
			}
		}
		p := *(*uintptr)(unsafe.Pointer(addr))
		if p == 0 {
			continue
		}
		if state != nil && p >= state.stack.lo && p < state.stack.hi {
			state.putPtr(p, false)
			continue
		}
		if tryDeferToSpanScan(p, gcw) {
			continue
		}
		if base, span, objIndex := findObject(p, obj, addr-obj); base != 0 {
			greyobject(base, obj, addr-obj, span, gcw, objIndex)
		}
	}
}

func logosArenaPush(gp *g) *sideStackChunk {
	c := logosArenaTake()
	if c == nil {
		return nil
	}
	if raceenabled {
		racemalloc(unsafe.Pointer(c), logosArenaChunkBytes)
	}
	c.next = gp.logosArena
	gp.logosArena = c
	return c
}

func logosArenaTake() *sideStackChunk {
	mp := acquirem()
	pp := mp.p.ptr()
	c := pp.logosArenaPool
	if c != nil {
		pp.logosArenaPool = c.next
		pp.logosArenaPoolN--
		c.next = nil
	}
	releasem(mp)
	if c == nil {
		c = logosArenaNewChunk()
	} else if asanenabled {
		asanunpoison(unsafe.Pointer(c), logosArenaChunkBytes)
	}
	return c
}

func logosArenaRelease(c *sideStackChunk) {
	if asanenabled {
		asanpoison(unsafe.Pointer(c), logosArenaChunkBytes)
	}
	mp := acquirem()
	pp := mp.p.ptr()
	c.next = pp.logosArenaPool
	pp.logosArenaPool = c
	pp.logosArenaPoolN++
	releasem(mp)
}

func logosArenaNewChunk() *sideStackChunk {
	if logosArenaChunkTotal.Add(1) > logosArenaChunkLimit.Load() {
		logosArenaChunkTotal.Add(-1)
		return nil
	}
	var s *mspan
	systemstack(func() {
		s = mheap_.allocManual(logosArenaChunkPages, spanAllocLogosArena)
	})
	if s == nil {
		logosArenaChunkTotal.Add(-1)
		return nil
	}
	s.isLogosArenaChunk = true
	if s.needzero != 0 {
		memclrNoHeapPointers(unsafe.Pointer(s.base()), logosArenaChunkBytes)
	}
	c := (*sideStackChunk)(unsafe.Pointer(s.base()))
	c.reset()
	if asanenabled {
		asanunpoison(unsafe.Pointer(s.base()), logosArenaChunkBytes)
	}
	return c
}

func logosArenaDrop(gp *g) {
	c := gp.logosArena
	gp.logosArena = nil
	for c != nil {
		next := c.next
		c.popScan(c.scanBase)
		c.reset()
		logosArenaRelease(c)
		c = next
	}
}

func logosArenaPark(gp *g) {
	if goexperiment.LogosArena && gp.logosArena != nil {
		logosArenaParkChain(gp)
	}
	if goexperiment.LogosArena && gp.logosRegions != nil {
		logosRegionPark(gp)
	}
}

func logosArenaParkChain(gp *g) {
	var prev *sideStackChunk
	c := gp.logosArena
	for c != nil {
		next := c.next
		if c.empty() {
			if prev == nil {
				gp.logosArena = next
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

func logosArenaExit(gp *g) {
	if !goexperiment.LogosArena {
		return
	}
	gp.logosArenaRetOK = false
	if gp.logosArena != nil {
		logosArenaDrop(gp)
	}
	if gp.logosRegions != nil || gp.logosRegionFree != nil {
		logosRegionExit(gp)
	}
}

func logosArenaFlushP(pp *p) {
	if goexperiment.LogosArena && pp.logosArenaPool != nil {
		systemstack(func() {
			logosArenaFreePool(pp)
		})
	}
}

//go:systemstack
func logosArenaFreePool(pp *p) {
	c := pp.logosArenaPool
	pp.logosArenaPool = nil
	pp.logosArenaPoolN = 0
	for c != nil {
		next := c.next
		s := spanOf(uintptr(unsafe.Pointer(c)))
		s.isLogosArenaChunk = false
		if asanenabled {
			asanpoison(unsafe.Pointer(c), logosArenaChunkBytes)
		}
		mheap_.freeManual(s, spanAllocLogosArena)
		logosArenaChunkTotal.Add(-1)
		c = next
	}
}

func inLogosArenaChunk(p uintptr) bool {
	s := spanOf(p)
	return s != nil && s.isLogosArenaChunk && s.state.get() == mSpanManual && p >= s.base() && p < s.limit
}

func logosArenaReject(p unsafe.Pointer, api string) {
	if goexperiment.LogosArena && inLogosArenaChunk(uintptr(p)) {
		panic(plainError("runtime: " + api + " on loGo/s arena object"))
	}
}

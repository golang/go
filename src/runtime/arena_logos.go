// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"internal/goexperiment"
	"internal/runtime/atomic"
	"internal/runtime/sys"
	"unsafe"
)

const (
	// logosArenaChunkBytes is the size of one arena chunk (SPEC Sec 3.1).
	logosArenaChunkBytes = 64 << 10
	logosArenaChunkPages = logosArenaChunkBytes / pageSize

	// logosArenaUnit is the size and alignment unit of the scan region and
	// the minimum alignment of the noscan region.
	logosArenaUnit = 2 * goarch.PtrSize

	// logosArenaHeaderBytes is the chunk header size, rounded up to the unit.
	logosArenaHeaderBytes = (unsafe.Sizeof(sideStackChunk{}) + logosArenaUnit - 1) &^ (logosArenaUnit - 1)

	// logosArenaScanHeaderBytes is the size of the type word block that
	// precedes each scan-region object.
	logosArenaScanHeaderBytes = unsafe.Sizeof(logosArenaScanHeader{})

	// logosArenaDefaultLimit caps the chunks that exist at once, in use or
	// cached (1 GiB). Past it an allocation falls back to mallocgc, which is
	// always safe because the heap outlives every frame.
	logosArenaDefaultLimit = 1 << 14
)

// sideStackChunk is one leased arena chunk (PLAN Sec 8.1). Its header sits at
// the start of the chunk, which is a manual span of its own. The noscan
// region grows up from noscanBase to noscanCur, the scan region grows down
// from scanBase to scanCur, and noscanCur <= scanCur always holds. Chunks of
// one goroutine link newest first through next; the same field links the
// chunks cached in a P.
type sideStackChunk struct {
	_          sys.NotInHeap
	next       *sideStackChunk
	noscanBase uintptr
	noscanCur  uintptr
	scanCur    uintptr
	scanBase   uintptr
}

// logosArenaScanHeader precedes every scan-region object: the type word that
// the stack scan reads (AR3) and the allocated byte size. Objects sit densely:
// the next header starts at the object address plus size rounded up to
// logosArenaUnit.
type logosArenaScanHeader struct {
	typ  *_type
	size uintptr
}

// logosArenaMark is a watermark saved at frame entry. The zero mark stands
// for a goroutine that holds no live arena memory.
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

// popScan raises scanCur to to, then zeroes the popped scan memory (AR2,
// #71, #81): the watermark moves first so the collector never scans memory
// that is about to be cleared.
func (c *sideStackChunk) popScan(to uintptr) {
	old := c.scanCur
	if to < old || to > c.scanBase {
		throw("logosArena: scan watermark restored out of range")
	}
	c.scanCur = to
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
		size <= logosArenaChunkBytes-logosArenaHeaderBytes-logosArenaScanHeaderBytes
}

func logosArenaUsable(gp *g) bool {
	return gp.m.curg == gp
}

// logosArenaAlloc returns zeroed, pointer-free memory of size bytes aligned
// for typ from the calling goroutine's arena, leasing a chunk on first use.
// A request that the current chunk cannot hold links a fresh chunk and leaves
// earlier objects in place. When no chunk can be had, or the object cannot
// fit any chunk, it falls back to mallocgc. The memory lives until the
// allocating frame restores its watermark.
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
	return p
}

// logosArenaNoscanLinked links a fresh chunk to gp and carves a noscan
// object from it. It returns nil when no chunk can be had.
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

// logosArenaScanLinked links a fresh chunk to gp and carves a scan object
// from it. It returns nil when no chunk can be had.
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

// logosArenaAllocScan is logosArenaAlloc for objects that contain pointers.
// Until the stack scan reads [scanCur, scanBase) of every chunk (M18-S3,
// AR3), the collector does not see the pointers held in these objects, so
// the compiler must not place pointer-containing objects here before that
// hook lands.
// The object is preceded by a logosArenaScanHeader carrying typ, and the
// memory is zero on return: tryScan clears it, because the span of a scan
// object may overlay noscan bytes that an earlier frame left dirty.
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
	return p
}

// logosArenaSave returns the watermark to restore at frame exit. A goroutine
// whose arena is empty yields the zero mark, because gopark may return its
// chunk before the frame restores.
func logosArenaSave() logosArenaMark {
	c := getg().logosArena
	if c == nil || (c.next == nil && c.empty()) {
		return logosArenaMark{}
	}
	return logosArenaMark{chunk: c, noscan: c.noscanCur, scan: c.scanCur}
}

// logosArenaRestore pops the arena back to m (AR2): chunks linked since the
// save return to the P cache, the scan watermark rises before the popped scan
// memory is zeroed, and the noscan watermark drops. The zero mark empties the
// arena but keeps the goroutine's first chunk until it parks or exits.
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
	c.noscanCur = m.noscan
}

// logosArenaPush leases a chunk and links it as the newest chunk of gp. It
// returns nil when the chunk limit is reached or the heap cannot grow.
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
	}
	return c
}

// logosArenaRelease caches a chunk whose regions are at base and whose scan
// memory is zero in the P of the calling M.
func logosArenaRelease(c *sideStackChunk) {
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
	return c
}

// logosArenaDrop returns every chunk of gp to the P cache, clearing leftover
// scan memory first (AR2).
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

// logosArenaPark is the gopark hook: a goroutine parking with empty regions
// in every chunk returns its chunks (AR4, #4).
func logosArenaPark(gp *g) {
	if goexperiment.LogosArena && gp.logosArena != nil {
		logosArenaParkChain(gp)
	}
}

func logosArenaParkChain(gp *g) {
	for c := gp.logosArena; c != nil; c = c.next {
		if !c.empty() {
			return
		}
	}
	logosArenaDrop(gp)
}

// logosArenaExit is the gdestroy hook: an exiting goroutine returns its
// chunks even when an unwind skipped the frame restores.
func logosArenaExit(gp *g) {
	if goexperiment.LogosArena && gp.logosArena != nil {
		logosArenaDrop(gp)
	}
}

// logosArenaFlushP is the hook for the end of a GC cycle and for P
// destruction: it frees the chunks cached in pp, the way stack caches clear.
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
		mheap_.freeManual(s, spanAllocLogosArena)
		logosArenaChunkTotal.Add(-1)
		c = next
	}
}

// inLogosArenaChunk reports whether p points into an arena chunk. The chunks
// are manual spans, so the span flag is the only record of their kind.
func inLogosArenaChunk(p uintptr) bool {
	s := spanOf(p)
	return s != nil && s.isLogosArenaChunk && s.state.get() == mSpanManual && p >= s.base() && p < s.limit
}

// logosArenaReject panics when p is an arena object and api would attach
// runtime state to it (#106). Arena memory is reused at frame exit, so no
// finalizer or cleanup may reference it.
func logosArenaReject(p unsafe.Pointer, api string) {
	if goexperiment.LogosArena && inLogosArenaChunk(uintptr(p)) {
		panic(plainError("runtime: " + api + " on loGo/s arena object"))
	}
}

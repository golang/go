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

// Generational collection for GOEXPERIMENT=logosgen (M19-S4).
//
// A span that survives logosGenMatureAge cycles is mature and closed to
// allocation, so every object in it is old. A minor cycle marks from the
// roots and from the dirty cards of mature spans, treats every pointer
// into a mature span as live without reading the span's mark bits, and
// sweeps mature spans only far enough to keep their bit arrays alive.
// Every logosGenFullEvery minor cycles, and for every forced or timed
// cycle, a full cycle marks and sweeps everything and returns every span
// to age 1.
//
// The card table holds one byte per logosGenCardBytes of heap. The compiled
// write barrier buffers the address of every pointer slot it stores to,
// tagged with logosGenSlotTag, and the buffer flush turns each entry into
// a dirty card; the bulk barriers and the atomic pointer stores dirty the
// cards of their ranges directly. The invariant between cycles: a card of
// a mature span is dirty if one of its slots may hold a pointer into a
// span that is not mature. Promotion rebuilds the cards of a span from its
// prospective survivors, and a card scan keeps a card dirty while a slot still
// points into a span that has not survived a cycle.

const (
	logosGenCardShift = 9
	logosGenCardBytes = 1 << logosGenCardShift
	logosGenMatureAge = 2
	logosGenFullEvery = 4

	// logosGenSlotTag marks a write barrier buffer entry as the address of
	// a stored-to slot instead of a pointer. Heap addresses never have the
	// top bit set.
	logosGenSlotTag = uintptr(1) << (goarch.PtrSize*8 - 1)

	logosGenHistoryCap = 256

	// logosGenGate passed: `go test -short std` with GOEXPERIMENT=logosgen
	// passes, so a -gc=greentea build with the experiment runs the
	// generational collector. Without the experiment behavior is unchanged
	// and the experiment stays opt-in.
	logosGenGate = true
)

// logosGenScratchLen is the entries of the buffer a store made without a P
// writes into: the most one barrier call asks for. It is zero without the
// experiment.
const logosGenScratchLen = wbMaxEntriesPerCall * goexperiment.LogosGenInt

// logosGenScratch is the scratch write barrier buffer of an M.
type logosGenScratch [logosGenScratchLen]uintptr

// Test modes of logosGen.test.
const (
	logosGenTestNone  = iota
	logosGenTestMinor // the next cycle is minor
	logosGenTestAuto  // the next cycle is decided as a heap-triggered cycle is
)

// logosGenCardTable is the card table of one heap arena.
type logosGenCardTable struct {
	_     sys.NotInHeap
	cards [heapArenaBytes >> logosGenCardShift]uint8
}

// logosGenState is the collector's generational state. active, minor, and
// minors change only with the world stopped.
type logosGenState struct {
	active bool // the process runs the generational collector
	minor  bool // the running or last cycle is minor
	test   uint8
	minors uint32 // minor cycles since the last full cycle

	// matureBytes counts the live bytes of the mature spans. The sweeper
	// adds a span when it becomes mature; a full cycle zeroes it.
	matureBytes atomic.Int64

	histN uint32
	hist  [logosGenHistoryCap]byte
}

var logosGen logosGenState

// wbCards reports whether pointer stores record their slots.
//
//go:nosplit
func wbCards() bool {
	return goexperiment.LogosGen && writeBarrier.cards
}

// wbActive reports whether pointer stores must run the barrier slow path:
// the collector is marking, or the card table is being kept.
//
//go:nosplit
func wbActive() bool {
	return writeBarrier.enabled || wbCards()
}

// logosGenEnable starts card recording when this process runs the
// generational collector: built with the experiment, and run without -gc or
// under -gc=greentea with the gate passed.
func logosGenEnable() {
	if !goexperiment.LogosGen {
		return
	}
	switch logosGCMode {
	case "":
	case "greentea":
		if !logosGenGate {
			return
		}
	default:
		return
	}
	logosGen.active = true
	writeBarrier.cards = true
}

// logosGenInitArena gives a new heap arena its card table.
func logosGenInitArena(ha *heapArena) {
	if !goexperiment.LogosGen {
		return
	}
	ha.logosCards = (*logosGenCardTable)(persistentalloc(unsafe.Sizeof(logosGenCardTable{}), goarch.PtrSize, &memstats.gcMiscSys))
	if ha.logosCards == nil {
		throw("out of memory allocating card table")
	}
}

// logosGenTriggerFor returns the heap size that starts a cycle: the pacer's
// trigger, or 50% growth past marked when that is sooner, but never below
// the heap minimum.
func logosGenTriggerFor(pacer, marked, minimum uint64) uint64 {
	return min(pacer, max(marked+marked/2, minimum))
}

// logosGenHeapTrigger returns the heap size that starts a cycle given the
// pacer's trigger. A heap-triggered cycle is the one a minor cycle serves.
func logosGenHeapTrigger(pacer uint64) uint64 {
	if !goexperiment.LogosGen || !logosGen.active || gcController.gcPercent.Load() < 0 {
		return pacer
	}
	return logosGenTriggerFor(pacer, gcController.heapMarked, gcController.heapMinimum)
}

// logosGenHeapCycleMinor reports whether a heap-triggered cycle is minor:
// fewer than logosGenFullEvery minor cycles have run since the last full
// cycle, and no mark-bit checking is on.
func logosGenHeapCycleMinor(minors uint32) bool {
	return minors < logosGenFullEvery && debug.gccheckmark == 0
}

// logosGenMinorCycle reports whether the cycle t starts is minor. A forced
// cycle and a timed cycle are full, and so is a cycle that detects leaked
// goroutines, which reads the mark bit of every sync object.
func logosGenMinorCycle(t gcTrigger) bool {
	switch {
	case work.goroutineLeak.pending.Load():
		return false
	case logosGen.test == logosGenTestMinor:
		return true
	case t.kind == gcTriggerHeap || logosGen.test == logosGenTestAuto:
		return logosGenHeapCycleMinor(logosGen.minors)
	}
	return false
}

// logosGenSkip reports whether marking and sweeping ignore s: the running
// cycle is minor and s is mature.
//
//go:nosplit
func logosGenSkip(s *mspan) bool {
	return goexperiment.LogosGen && logosGen.minor && s.logosAge >= logosGenMatureAge
}

// logosGenSkipPointer reports whether p points into a span that
// logosGenSkip ignores.
//
//go:nosplit
func logosGenSkipPointer(ha *heapArena, p uintptr) bool {
	if !goexperiment.LogosGen || !logosGen.minor {
		return false
	}
	s := ha.spans[(p/pageSize)%pagesPerArena]
	return s != nil && s.state.get() == mSpanInUse && p >= s.base() && p < s.limit && s.logosAge >= logosGenMatureAge
}

// logosGenClosed reports whether s is mature, and so closed to allocation.
func logosGenClosed(s *mspan) bool {
	return goexperiment.LogosGen && logosGen.active && s.logosAge >= logosGenMatureAge
}

// logosGenCardFor returns the card of the heap address addr, or nil when
// addr is outside every arena.
//
//go:nosplit
//go:nowritebarrierrec
func logosGenCardFor(addr uintptr) *uint8 {
	ri := arenaIndex(addr)
	if arenaL1Bits == 0 {
		if ri.l2() >= uint(len(mheap_.arenas[0])) {
			return nil
		}
	} else if ri.l1() >= uint(len(mheap_.arenas)) {
		return nil
	}
	l2 := mheap_.arenas[ri.l1()]
	if arenaL1Bits != 0 && l2 == nil {
		return nil
	}
	ha := l2[ri.l2()]
	if ha == nil || ha.logosCards == nil {
		return nil
	}
	return &ha.logosCards.cards[(addr%heapArenaBytes)>>logosGenCardShift]
}

// logosGenCardDirty reports whether the card of addr is dirty.
func logosGenCardDirty(addr uintptr) bool {
	c := logosGenCardFor(addr)
	return c != nil && *c != 0
}

// logosGenMarkCard dirties the card of addr when addr lies in a heap span.
//
//go:nosplit
//go:nowritebarrierrec
func logosGenMarkCard(addr uintptr) {
	if spanOfHeap(addr) == nil {
		return
	}
	if c := logosGenCardFor(addr); c != nil {
		*c = 1
	}
}

// logosGenFlushCards dirties the cards of the slot entries in ptrs and
// clears the entries.
//
//go:nosplit
//go:nowritebarrierrec
func logosGenFlushCards(ptrs []uintptr) {
	for i, e := range ptrs {
		if e&logosGenSlotTag == 0 {
			continue
		}
		logosGenMarkCard(e &^ logosGenSlotTag)
		ptrs[i] = 0
	}
}

// logosGenFlush handles the slot entries of a write barrier buffer flush.
// It reports whether the flush is complete: outside the mark phase the
// pointer entries are dropped, because nothing is marking.
//
//go:nosplit
//go:nowritebarrierrec
func logosGenFlush(pp *p, ptrs []uintptr) bool {
	if !goexperiment.LogosGen {
		return false
	}
	logosGenFlushCards(ptrs)
	if writeBarrier.enabled {
		return false
	}
	pp.wbBuf.reset()
	return true
}

// logosGenDrainCards dirties the cards of the slot entries buffered in pp
// without shading anything. The buffer keeps its pointer entries.
//
//go:nosplit
//go:nowritebarrierrec
func logosGenDrainCards(pp *p) {
	if !goexperiment.LogosGen || !logosGen.active {
		return
	}
	start := uintptr(unsafe.Pointer(&pp.wbBuf.buf[0]))
	if pp.wbBuf.next < start {
		return
	}
	n := (pp.wbBuf.next - start) / unsafe.Sizeof(pp.wbBuf.buf[0])
	logosGenFlushCards(pp.wbBuf.buf[:n])
}

// logosGenSlot dirties the card of slot after a pointer store of new to it.
// It needs no P.
//
//go:nosplit
//go:nowritebarrierrec
func logosGenSlot(slot, new uintptr) {
	if !wbCards() || new < minLegalPointer {
		return
	}
	logosGenMarkCard(slot)
}

// logosGenMarkRange dirties every card of the heap range [addr, addr+size).
//
//go:nosplit
//go:nowritebarrierrec
func logosGenMarkRange(addr, size uintptr) {
	for a := addr &^ (logosGenCardBytes - 1); a < addr+size; a += logosGenCardBytes {
		logosGenMarkCard(a)
	}
}

// logosGenGlobalDst reports whether a bulk copy into a global needs no
// barrier: only the card table asked for one, and globals are roots.
//
//go:nosplit
func logosGenGlobalDst() bool {
	return goexperiment.LogosGen && !writeBarrier.enabled
}

// logosGenBulk dirties the cards of a bulk pointer copy from src to dst into
// the heap span s: it may store a pointer anywhere in the range, and the
// card scan finds which slots do. It reports whether the copy needs nothing
// more, which is when no mark phase is running.
//
//go:nosplit
//go:nowritebarrierrec
func logosGenBulk(s *mspan, dst, src, size uintptr) bool {
	if !goexperiment.LogosGen {
		return false
	}
	if src != 0 && !s.spanclass.noscan() {
		logosGenMarkRange(dst, size)
	}
	return !writeBarrier.enabled
}

// logosGenTypeBits is logosGenBulk for typeBitsBulkBarrier, whose
// destination may be any memory: only heap cards are dirtied.
//
//go:nosplit
//go:nowritebarrierrec
func logosGenTypeBits(typ *_type, dst uintptr) bool {
	if !goexperiment.LogosGen {
		return false
	}
	logosGenMarkRange(dst, typ.PtrBytes)
	return !writeBarrier.enabled
}

// logosGenSettle turns the card entries buffered by every P into cards and
// drops the pointer entries, which are stale: no mark phase has run since
// they were buffered. The world is stopped, and the mark phase begins
// right after, so no later store can leave a pointer entry for the flush to
// shade.
func logosGenSettle() {
	if !goexperiment.LogosGen || !logosGen.active {
		return
	}
	for _, pp := range allp {
		logosGenDrainCards(pp)
		pp.wbBuf.reset()
	}
}

// logosGenBeginCycle decides the kind of the cycle starting now. The world
// is stopped and the previous sweep is finished. A full cycle forgets all
// cards, because every span is young after it.
func logosGenBeginCycle(t gcTrigger) {
	if !goexperiment.LogosGen || !logosGen.active {
		return
	}
	logosGen.minor = logosGenMinorCycle(t)
	kind := byte('F')
	if logosGen.minor {
		kind = 'm'
		logosGen.minors++
	} else {
		logosGen.minors = 0
		logosGen.matureBytes.Store(0)
		systemstack(logosGenClearCards)
	}
	logosGen.hist[logosGen.histN%logosGenHistoryCap] = kind
	logosGen.histN++
}

func (g *logosGenState) resetHistory() {
	g.histN = 0
}

// logosGenHistory returns the kinds of the recorded cycles, oldest first.
func logosGenHistory() []byte {
	n := min(logosGen.histN, logosGenHistoryCap)
	out := make([]byte, 0, n)
	for i := logosGen.histN - n; i < logosGen.histN; i++ {
		out = append(out, logosGen.hist[i%logosGenHistoryCap])
	}
	return out
}

// logosGenForEachArena calls f for every heap arena that has a card table,
// with the arena's base address. The world must be stopped.
//
//go:systemstack
func logosGenForEachArena(f func(ha *heapArena, base uintptr)) {
	lock(&mheap_.lock)
	arenas := mheap_.heapArenas
	unlock(&mheap_.lock)
	for _, ai := range arenas {
		ha := mheap_.arenas[ai.l1()][ai.l2()]
		if ha.logosCards != nil {
			f(ha, arenaBase(ai))
		}
	}
}

// logosGenClearCards forgets every card.
//
//go:systemstack
func logosGenClearCards() {
	logosGenForEachArena(func(ha *heapArena, _ uintptr) {
		clear(ha.logosCards.cards[:])
	})
}

// logosGenMatureBytes returns the live bytes the mature spans keep, which a
// minor cycle does not mark.
func logosGenMatureBytes() uint64 {
	if !goexperiment.LogosGen || !logosGen.minor {
		return 0
	}
	return uint64(logosGen.matureBytes.Load())
}

// logosGenMarkCards marks from the dirty cards of the mature spans at the
// start of a minor cycle. The world is stopped and the mark phase is set.
func logosGenMarkCards() {
	if !goexperiment.LogosGen || !logosGen.minor {
		return
	}
	systemstack(func() {
		gcw := &getg().m.p.ptr().gcw
		logosGenForEachArena(func(ha *heapArena, base uintptr) {
			logosGenScanArenaCards(ha, base, gcw)
		})
		gcw.dispose()
	})
}

// logosGenScanArenaCards scans the dirty cards of one arena. A card stays
// dirty while a slot in it points into a span that has not survived a cycle.
func logosGenScanArenaCards(ha *heapArena, base uintptr, gcw *gcWork) {
	cards := &ha.logosCards.cards
	for c := range cards {
		if cards[c] != 0 {
			cards[c] = logosGenScanCard(ha, base+uintptr(c)<<logosGenCardShift, gcw)
		}
	}
}

// logosGenSpanAt returns the in-use heap span containing a, or nil.
func logosGenSpanAt(ha *heapArena, a uintptr) *mspan {
	s := ha.spans[(a/pageSize)%pagesPerArena]
	if s == nil || s.state.get() != mSpanInUse || a < s.base() || a >= s.limit {
		return nil
	}
	return s
}

// logosGenScanCard greys the pointees of the card at lo and returns the new
// card value. A card may overlap more than one span; every overlapping
// mature scannable span is scanned, and the card stays dirty while any slot
// in it points into a span that has not survived a cycle.
func logosGenScanCard(ha *heapArena, lo uintptr, gcw *gcWork) uint8 {
	end := lo + logosGenCardBytes
	young := false
	for a := lo; a < end; {
		s := logosGenSpanAt(ha, a)
		if s == nil {
			a = min(end, (a+pageSize)&^(pageSize-1))
			continue
		}
		hi := min(end, s.limit)
		if s.logosAge >= logosGenMatureAge && !s.spanclass.noscan() && logosGenScanSpanRange(s, a, hi, gcw) {
			young = true
		}
		a = hi
	}
	if young {
		return 1
	}
	return 0
}

// logosGenScanSpanRange greys the pointees of the pointer slots of the
// objects of s overlapping [from, to) and reports whether any points into
// a span that has not survived a cycle.
func logosGenScanSpanRange(s *mspan, from, to uintptr, gcw *gcWork) bool {
	young := false
	for i := s.objIndex(from); i < uintptr(s.nelems); i++ {
		obj := s.base() + i*s.elemsize
		if obj >= to {
			break
		}
		if s.isFree(i) {
			continue
		}
		if logosGenScanObject(s, obj, from, to, gcw) {
			young = true
		}
	}
	return young
}

// logosGenScanObject greys the pointees of the pointer slots of the object
// at obj, within [lo, hi) when the object is larger than a card, and
// reports whether any points into a span that has not survived a cycle.
func logosGenScanObject(s *mspan, obj, lo, hi uintptr, gcw *gcWork) bool {
	from, to := obj, obj+s.elemsize
	if s.elemsize > logosGenCardBytes {
		from, to = max(lo, obj), min(hi, to)
	}
	tp := s.typePointersOfUnchecked(obj)
	if from > obj {
		tp = tp.fastForward(from-tp.addr, to)
	}
	young := false
	for {
		var addr uintptr
		if tp, addr = tp.next(to); addr == 0 {
			return young
		}
		v := *(*uintptr)(unsafe.Pointer(addr))
		if v < minLegalPointer {
			continue
		}
		if t := spanOfHeap(v); t != nil && t.logosAge == 0 {
			young = true
		}
		logosGenGrey(v, gcw)
	}
}

// logosGenGrey marks the object v points into, as scanning a slot does.
func logosGenGrey(v uintptr, gcw *gcWork) {
	if tryDeferToSpanScan(v, gcw) {
		return
	}
	if obj, span, idx := findObject(v, 0, 0); obj != 0 {
		greyobject(obj, 0, 0, span, gcw, idx)
	}
}

// logosGenEndCycle runs at mark termination of a minor cycle, with the
// world stopped and the marks final. It rebuilds the cards of every span
// the coming sweep makes mature, from the span's marked objects and the
// unmarked objects its finalizers will retain.
func logosGenEndCycle() {
	if !goexperiment.LogosGen || !logosGen.minor {
		return
	}
	for _, s := range mheap_.allspans {
		if s.state.get() == mSpanInUse && s.logosAge+1 == logosGenMatureAge && !s.isUserArenaChunk && !s.spanclass.noscan() {
			logosGenPromote(s)
		}
	}
}

// logosGenPromote makes the cards of s describe its prospective survivors:
// marked objects, plus unmarked objects carrying finalizers, which the
// coming sweep retains instead of freeing. A card is dirty when a slot in
// it points into a span that has not survived a cycle. A card shared with
// another span stays dirty: only the owning span may clear it, and this
// span does not own the shared edge.
func logosGenPromote(s *mspan) {
	for a := s.base(); a < s.limit; a += logosGenCardBytes {
		if c := logosGenCardFor(a); c != nil {
			if a+logosGenCardBytes > s.limit {
				*c = 1
			} else {
				*c = 0
			}
		}
	}
	for i := range uintptr(s.nelems) {
		if s.markBitsForIndex(i).isMarked() {
			logosGenPromoteObject(s, s.base()+i*s.elemsize)
		}
	}
	// Unmarked finalizer survivors need cards too; s.limit matches no object base.
	last := s.limit
	for sp := s.specials; sp != nil; sp = sp.next {
		if sp.kind != _KindSpecialFinalizer {
			continue
		}
		base := s.base() + sp.offset/s.elemsize*s.elemsize
		if base == last {
			continue
		}
		last = base
		if s.markBitsForIndex(sp.offset / s.elemsize).isMarked() {
			continue
		}
		logosGenPromoteObject(s, base)
	}
}

// logosGenPromoteObject dirties the card of every slot of the object at obj
// that points into a span that has not survived a cycle.
func logosGenPromoteObject(s *mspan, obj uintptr) {
	tp := s.typePointersOfUnchecked(obj)
	for {
		var addr uintptr
		if tp, addr = tp.next(obj + s.elemsize); addr == 0 {
			return
		}
		v := *(*uintptr)(unsafe.Pointer(addr))
		if v < minLegalPointer {
			continue
		}
		if t := spanOfHeap(v); t != nil && t.logosAge == 0 {
			logosGenMarkCard(addr)
		}
	}
}

// logosGenSweepMature is the sweep of a span a minor cycle skipped: the span
// keeps its objects, and only its bit arrays move to the current epoch,
// because the sweeper recycles the previous epoch. The allocation index
// returns to the start of the span so free checks read the bit arrays,
// which is what a normal sweep leaves behind.
func logosGenSweepMature(s *mspan) {
	mheap_.pagesSwept.Add(int64(s.npages))
	logosGenRehome(s)
	s.freeindex = 0
	s.freeIndexForScan = 0
	s.refillAllocCache(0)
	atomic.Store(&s.sweepgen, mheap_.sweepgen)
}

// logosGenRehome copies the allocation bits of s and gives it fresh mark
// bits in the arenas of the current epoch.
func logosGenRehome(s *mspan) {
	n := (uintptr(s.nelems) + 63) / 64 * 8
	fresh := newMarkBits(uintptr(s.nelems))
	memmove(unsafe.Pointer(fresh.bytep(0)), unsafe.Pointer(s.allocBits.bytep(0)), n)
	s.allocBits = fresh
	s.gcmarkBits = newMarkBits(uintptr(s.nelems))
	if s.pinnerBits != nil {
		s.refreshPinnerBits()
	}
}

// logosGenRelist puts the swept mature span s on its mcentral's full list.
func logosGenRelist(s *mspan) {
	mheap_.central[s.spanclass].mcentral.fullSwept(mheap_.sweepgen).push(s)
}

// logosGenAge counts the cycle s survived. A full cycle returns every span
// to age 1; a minor cycle ages a span until it is mature.
func logosGenAge(s *mspan, nalloc uint16) {
	if !goexperiment.LogosGen || !logosGen.active || s.isUserArenaChunk || nalloc == 0 {
		return
	}
	if !logosGen.minor {
		s.logosAge = 1
		return
	}
	s.logosAge++
	if s.logosAge == logosGenMatureAge {
		logosGen.matureBytes.Add(int64(nalloc) * int64(s.elemsize))
	}
}

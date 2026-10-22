// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.greenteagc && (!js || !wasm)

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/runtime/gc"
	"unsafe"
)

// Test seam for M19-S6: wrappers over the Green Tea span deck for package
// runtime_test. Every wrapper that touches collector state either holds the
// world stopped with the collector idle and restores the state it touched,
// or reads memory the collector does not write.

// LogosSpanDeckStats sums the pop and order-violation counters of every P.
func LogosSpanDeckStats() (pops uint64, inversions uint32) {
	stw := stopTheWorldGC(stwUnknown)
	for _, pp := range allp {
		pops += pp.gcw.spanq.deck.pops
		inversions += pp.gcw.spanq.deck.inversions
	}
	startTheWorldGC(stw)
	return pops, inversions
}

// LogosSpanQueueScript runs a script against a fresh span queue in the
// compiled mode. A nonzero entry puts the span whose base is that address; a
// zero entry pops one span (local queue first, then the queue's own spill
// chain, as tryGetSpan does). The queue is drained after the last entry. It
// reports each popped span base with the sweep counter at that pop, the
// number of puts that were never popped, and the queue's order-violation
// counter.
func LogosSpanQueueScript(script []uintptr) (bases []uintptr, sweeps []uint32, lost int, inversions uint32) {
	bases = make([]uintptr, 0, len(script))
	sweeps = make([]uint32, 0, len(script))
	puts := 0
	q := new(spanQueue)
	stw := stopTheWorldGC(stwUnknown)
	pop := func() {
		s := q.tryGetFast()
		if s == 0 {
			s = q.steal(q)
		}
		if s != 0 {
			bases = append(bases, s.spanBase())
			sweeps = append(sweeps, q.deck.sweeps)
		}
	}
	for _, v := range script {
		if v == 0 {
			pop()
			continue
		}
		q.put(makeObjPtr(v, 5))
		puts++
	}
	for !q.empty() {
		pop()
	}
	q.destroy()
	inversions = q.deck.inversions
	startTheWorldGC(stw)
	return bases, sweeps, puts - len(bases), inversions
}

// LogosMarkProbeResult is what LogosSpanMarkProbe observed.
type LogosMarkProbeResult struct {
	Fresh                bool // the two objects started unmarked, unscanned, in clean spans
	ParentMarkedBefore   bool
	ParentDirtyBefore    bool
	ParentMarked         bool
	ParentDirty          bool
	ChildMarkedAfterMark bool // marking the parent must not have followed its pointer
	Queued               int
	ChildMarkedAfterScan bool
	ChildDirtyAfterScan  bool
}

// LogosSpanMarkProbe marks parent through the collector's own marking entry,
// reads the mark bit and the dirty state, then scans the queued span and reads
// them again. Both objects must live in spans with inline mark bits and child
// must hold no pointers. The world stays stopped throughout and the probe
// clears every bit it set, so no later cycle sees it.
func LogosSpanMarkProbe(parent, child unsafe.Pointer) (r LogosMarkProbeResult) {
	gcw := new(gcWork)
	gcw.ptrBuf = new([pageSize / goarch.PtrSize]uintptr)
	stw := stopTheWorldGC(stwUnknown)
	r = logosMarkProbeLocked(gcw, uintptr(parent), uintptr(child))
	startTheWorldGC(stw)
	return r
}

func logosMarkProbeLocked(gcw *gcWork, pp, cp uintptr) (r LogosMarkProbeResult) {
	ps, cs := spanOf(pp), spanOf(cp)
	if gcphase != _GCoff || !gcUsesSpanInlineMarkBits(ps.elemsize) || !gcUsesSpanInlineMarkBits(cs.elemsize) {
		return r
	}
	pi, ci := ps.objIndex(pp), cs.objIndex(cp)
	pm, cm := ps.markBitsForIndex(pi), cs.markBitsForIndex(ci)
	psc, csc := ps.scannedBitsForIndex(pi), cs.scannedBitsForIndex(ci)
	r.Fresh = !pm.isMarked() && !cm.isMarked() && !psc.isMarked() && !csc.isMarked() &&
		!logosSpanDirty(ps.base()) && !logosSpanDirty(cs.base())
	if !r.Fresh {
		return r
	}
	r.ParentMarkedBefore, r.ParentDirtyBefore = pm.isMarked(), logosSpanDirty(ps.base())
	tryDeferToSpanScan(pp, gcw)
	r.ParentMarked, r.ParentDirty = pm.isMarked(), logosSpanDirty(ps.base())
	r.ChildMarkedAfterMark = cm.isMarked()
	r.Queued = int(gcw.spanq.tail - gcw.spanq.head)
	scanSpan(gcw.spanq.tryGetFast(), gcw)
	r.ChildMarkedAfterScan, r.ChildDirtyAfterScan = cm.isMarked(), logosSpanDirty(cs.base())
	for s := gcw.spanq.tryGetFast(); s != 0; s = gcw.spanq.tryGetFast() {
		scanSpan(s, gcw)
	}
	for _, b := range []markBits{pm, cm, psc, csc} {
		*b.bytep &^= b.mask
	}
	return r
}

// LogosSpanGatherSlots returns the pointer slots the span scan collects for
// the grey objects objs, which must share one span with inline mark bits,
// together with the span's base, size in bytes and object size.
func LogosSpanGatherSlots(objs []unsafe.Pointer) (slots []uintptr, base, size, elemsize uintptr) {
	s := spanOf(uintptr(objs[0]))
	var grey gc.ObjMask
	for _, o := range objs {
		i := s.objIndex(uintptr(o))
		grey[i/(goarch.PtrSize*8)] |= 1 << (i % (goarch.PtrSize * 8))
	}
	gcw := new(gcWork)
	gcw.ptrBuf = new([pageSize / goarch.PtrSize]uintptr)
	n := logosGatherObjects(s.base(), s.elemsize, uint16(s.nelems), gcw, &grey)
	return append([]uintptr(nil), gcw.ptrBuf[:n]...), s.base(), s.npages * pageSize, s.elemsize
}

// LogosMapKeyHasher returns the hash function the map type of m uses for keys.
func LogosMapKeyHasher(m any) func(unsafe.Pointer, uintptr) uintptr {
	return (*abi.MapType)(unsafe.Pointer(efaceOf(&m)._type)).Hasher
}

// logosBitRef names one mark or scan bit by address. The address is a
// uintptr on purpose: it points into the inline mark bits that end a span's
// page, past its last object, and a pointer there would make the collector
// mark an object slot that does not exist.
type logosBitRef struct {
	addr uintptr
	mask uint8
}

func (b logosBitRef) set()   { *(*uint8)(unsafe.Pointer(b.addr)) |= b.mask }
func (b logosBitRef) clear() { *(*uint8)(unsafe.Pointer(b.addr)) &^= b.mask }

// LogosSpanOrderRig is a set of small heap objects prepared so that
// BenchmarkLogosSpanOrder can scan their spans in a chosen order with the
// collector's own span scan, and then restore every bit it touched.
type LogosSpanOrderRig struct {
	marks, scans []logosBitRef
	bases        []uintptr
	owned        []uintptr
	gcw          *gcWork
}

// NewLogosSpanOrderRig prepares the objects objs, which must point only to one
// another and live in clean spans with inline mark bits. It returns nil when
// any of them does not.
func NewLogosSpanOrderRig(objs []unsafe.Pointer) *LogosSpanOrderRig {
	r := &LogosSpanOrderRig{gcw: new(gcWork)}
	r.gcw.ptrBuf = new([pageSize / goarch.PtrSize]uintptr)
	seen := make(map[uintptr]bool)
	for _, o := range objs {
		s := spanOf(uintptr(o))
		if s == nil || !gcUsesSpanInlineMarkBits(s.elemsize) {
			return nil
		}
		i := s.objIndex(uintptr(o))
		m, sc := s.markBitsForIndex(i), s.scannedBitsForIndex(i)
		if m.isMarked() || sc.isMarked() || logosSpanDirty(s.base()) {
			return nil
		}
		r.marks = append(r.marks, logosBitRef{uintptr(unsafe.Pointer(m.bytep)), m.mask})
		r.scans = append(r.scans, logosBitRef{uintptr(unsafe.Pointer(sc.bytep)), sc.mask})
		if !seen[s.base()] {
			seen[s.base()] = true
			r.bases = append(r.bases, s.base())
			r.owned = append(r.owned, uintptr(unsafe.Pointer(&spanInlineMarkBitsFromBase(s.base()).owned)))
		}
	}
	return r
}

// Bases returns the base address of each span of the rig, in no particular
// order.
func (r *LogosSpanOrderRig) Bases() []uintptr { return r.bases }

// Arm marks every rig object and makes every rig span dirty, as a mark phase
// would after reaching all of them. No collection may run until Disarm.
func (r *LogosSpanOrderRig) Arm() {
	for _, m := range r.marks {
		m.set()
	}
	for _, o := range r.owned {
		*(*spanScanOwnership)(unsafe.Pointer(o)) = spanScanManyMark
	}
}

// Scan scans the rig spans listed in order, in that order, with scanSpan.
func (r *LogosSpanOrderRig) Scan(order []uintptr) {
	for _, base := range order {
		scanSpan(makeObjPtr(base, 0), r.gcw)
	}
}

// Disarm clears every mark, scan and dirty bit Arm and Scan set.
func (r *LogosSpanOrderRig) Disarm() {
	for i := range r.marks {
		r.marks[i].clear()
		r.scans[i].clear()
	}
	for _, o := range r.owned {
		*(*spanScanOwnership)(unsafe.Pointer(o)) = spanScanUnowned
	}
}

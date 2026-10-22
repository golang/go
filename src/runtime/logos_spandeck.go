// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.greenteagc && (!js || !wasm)

package runtime

import (
	"internal/goarch"
	"internal/goexperiment"
	"internal/runtime/gc"
	"internal/runtime/sys"
	"unsafe"
)

// logosSpanDeckOn selects the span deck (M19-S6, SPEC Sec 19.1), the
// address-ordered form of the Green Tea span queue. Marking a pointer into a
// small-object span already sets the object's mark bit and the span's owned
// byte (logosSpanDirty) without reading the object (tryDeferToSpanScan). The
// deck changes what happens to a dirty span afterwards: the P-local ring pops
// spans in ascending address order, one sweep at a time, and the span scan
// prefetches the mark bits of the spans its pointers lead to.
//
// The ring stays a spanQueue ring so spilling, stealing and flushing keep
// their code. In deck mode its live region [head, tail) is sorted in
// descending order: spilling takes the highest spans, which are the last ones
// the owner would reach, and a pop removes the entry just above the cursor.
//
// The setting is a compile-time invariant. It is on only when the build sets
// GOEXPERIMENT=logosspandeck; there is no runtime toggle.
const logosSpanDeckOn = goexperiment.LogosSpanDeck

// logosSpanPrefetchDepth is how many pointer targets the scan prefetches
// ahead of the one it is marking.
const logosSpanPrefetchDepth = 12

// logosSpanDeck is the per-queue order state. A sweep is a run of pops that
// ascend; it ends when no queued span lies above the cursor.
type logosSpanDeck struct {
	cursor     uintptr // lowest base the next pop may take in this sweep
	last       uintptr // base of the previous pop in this sweep
	pops       uint64
	sweeps     uint32
	inversions uint32 // pops that broke the order inside a sweep; always zero
}

// logosSpanDirty reports whether the span with inline mark bits at base has
// grey objects: it is queued for scanning or being scanned.
func logosSpanDirty(base uintptr) bool {
	return spanInlineMarkBitsFromBase(base).owned.load() != spanScanUnowned
}

// land records a pop of s.
func (d *logosSpanDeck) land(s objptr) {
	base := s.spanBase()
	if base < d.last {
		d.inversions++
	}
	d.last = base
	d.cursor = base + gc.PageSize
	d.pops++
}

// logosSpanAtLeast returns how many entries of the descending ring are at
// least key.
func logosSpanAtLeast(ring []objptr, key uintptr) int {
	lo, hi := 0, len(ring)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if uintptr(ring[mid]) >= key {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// logosSpanSort sorts ring in descending order.
func logosSpanSort(ring []objptr) {
	for i := 1; i < len(ring); i++ {
		s := ring[i]
		j := i
		for ; j > 0 && ring[j-1] < s; j-- {
			ring[j] = ring[j-1]
		}
		ring[j] = s
	}
}

// logosSpanAscend reverses a descending run in place, so a batch published to
// the shared chain ascends and a thief that takes it in order sweeps upward.
// A run that is already ascending is left alone, which makes a retried
// publication safe.
func logosSpanAscend(run []objptr) {
	if len(run) < 2 || run[0] < run[len(run)-1] {
		return
	}
	for i, j := 0, len(run)-1; i < j; i, j = i+1, j-1 {
		run[i], run[j] = run[j], run[i]
	}
}

// logosDeckPack moves the live ring region to the start of the ring and
// returns its length.
func (q *spanQueue) logosDeckPack() int {
	n := q.tail - q.head
	if n == 0 {
		q.head, q.tail = 0, 0
		return 0
	}
	if q.head != 0 {
		copy(q.ring[:n], q.ring[q.head:q.tail])
		q.head, q.tail = 0, n
	}
	return int(n)
}

// logosDeckInsert puts s into its place in the ring, or reports that the ring
// is full.
func (q *spanQueue) logosDeckInsert(s objptr) bool {
	n := q.logosDeckPack()
	if n == len(q.ring) {
		return false
	}
	i := logosSpanAtLeast(q.ring[:n], uintptr(s)+1)
	copy(q.ring[i+1:n+1], q.ring[i:n])
	q.ring[i] = s
	q.tail = uint32(n + 1)
	return true
}

// logosDeckPop removes the lowest span at or above the cursor, or starts a
// new sweep from the lowest span when none is left above it.
func (q *spanQueue) logosDeckPop() objptr {
	n := q.logosDeckPack()
	if n == 0 {
		return 0
	}
	k := logosSpanAtLeast(q.ring[:n], q.deck.cursor)
	if k == 0 {
		q.deck.cursor, q.deck.last = 0, 0
		q.deck.sweeps++
		k = n
	}
	s := q.ring[k-1]
	copy(q.ring[k-1:n-1], q.ring[k:n])
	q.tail = uint32(n - 1)
	q.deck.land(s)
	return s
}

// logosMarkOffset is where the mark bits and the size class sit in a span:
// the second cache line of its inline mark bits.
const logosMarkOffset = gc.PageSize - unsafe.Sizeof(spanInlineMarkBits{}) + unsafe.Offsetof(spanInlineMarkBits{}.marks)

// logosMarkLine returns the address of the cache line that holds the mark
// bits and the size class of the span p points into. It is plain address
// arithmetic: a nil-checked field address would load from the line it is
// about to prefetch.
func logosMarkLine(p uintptr) uintptr {
	return alignDown(p, gc.PageSize) + logosMarkOffset
}

// logosGatherSlots appends the pointer slots of the object at b to
// gcw.ptrBuf from index nptrs and prefetches each slot. It returns the new
// count.
func logosGatherSlots(gcw *gcWork, hbits *byte, base, b, objSize uintptr, nptrs int) int {
	ptrBits := extractHeapBitsSmall(hbits, base, b, objSize)
	gcw.heapScanWork += int64(sys.Len64(uint64(ptrBits)) * goarch.PtrSize)
	for ; ptrBits != 0; ptrBits &= ptrBits - 1 {
		addr := b + uintptr(sys.TrailingZeros64(uint64(ptrBits)))*goarch.PtrSize
		sys.Prefetch(addr)
		gcw.ptrBuf[nptrs] = addr
		nptrs++
	}
	return nptrs
}

// logosGatherObjects walks the grey objects of the single-page span at base
// in address order and gathers their pointer slots. It returns the number of
// slots in gcw.ptrBuf.
func logosGatherObjects(base, objSize uintptr, elems uint16, gcw *gcWork, grey *gc.ObjMask) int {
	hbitsBase, _ := spanHeapBitsRange(base, gc.PageSize, objSize)
	hbits := (*byte)(unsafe.Pointer(hbitsBase))
	nptrs := 0
	for i, bits := range grey {
		if i*(goarch.PtrSize*8) > int(elems) {
			break
		}
		for ; bits != 0; bits &= bits - 1 {
			j := uintptr(sys.TrailingZeros64(uint64(bits)))
			b := base + (uintptr(i)*goarch.PtrSize*8+j)*objSize
			nptrs = logosGatherSlots(gcw, hbits, base, b, objSize, nptrs)
		}
	}
	return nptrs
}

// logosFlushSlots loads the pointers held in the first nptrs slots of
// gcw.ptrBuf and marks the ones that are not nil.
func logosFlushSlots(gcw *gcWork, nptrs int) {
	buf := gcw.ptrBuf[:nptrs]
	m := 0
	for _, slot := range buf {
		if p := *(*uintptr)(unsafe.Pointer(slot)); p != 0 {
			buf[m] = p
			m++
		}
	}
	logosMarkPtrs(gcw, buf[:m])
}

// logosMarkPtrs marks each pointer in ptrs, prefetching the mark bits of the
// pointer logosSpanPrefetchDepth places ahead.
func logosMarkPtrs(gcw *gcWork, ptrs []uintptr) {
	for _, p := range ptrs[:min(len(ptrs), logosSpanPrefetchDepth)] {
		sys.Prefetch(logosMarkLine(p))
	}
	for i, p := range ptrs {
		if j := i + logosSpanPrefetchDepth; j < len(ptrs) {
			sys.Prefetch(logosMarkLine(ptrs[j]))
		}
		if !tryDeferToSpanScan(p, gcw) {
			if obj, span, objIndex := findObject(p, 0, 0); obj != 0 {
				greyobject(obj, 0, 0, span, gcw, objIndex)
			}
		}
	}
}

// logosScanObjects scans every grey object of one span in a single
// ascending pass.
func logosScanObjects(base, objSize uintptr, elems uint16, gcw *gcWork, grey *gc.ObjMask) {
	logosFlushSlots(gcw, logosGatherObjects(base, objSize, elems, gcw, grey))
}

// logosScanObject scans the one grey object at b.
func logosScanObject(base, b, objSize uintptr, gcw *gcWork) {
	hbitsBase, _ := spanHeapBitsRange(base, gc.PageSize, objSize)
	hbits := (*byte)(unsafe.Pointer(hbitsBase))
	logosFlushSlots(gcw, logosGatherSlots(gcw, hbits, base, b, objSize, 0))
}

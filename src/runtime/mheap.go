// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Page heap.
//
// See malloc.go for overview.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// Main malloc heap.
// The heap itself is the "free[]" and "large" arrays,
// but all the other global data is here too.
type mheap struct {
	lock      mutex
	free      [_MaxMHeapList]mSpanList // free lists of given length
	freelarge mSpanList                // free lists length >= _MaxMHeapList
	busy      [_MaxMHeapList]mSpanList // busy lists of large objects of given length
	busylarge mSpanList                // busy lists of large objects length >= _MaxMHeapList
	allspans  **mspan                  // all spans out there
	gcspans   **mspan                  // copy of allspans referenced by gc marker or sweeper
	nspan     uint32
	sweepgen  uint32 // sweep generation, see comment in mspan
	sweepdone uint32 // all spans are swept
	// span lookup
	spans        **mspan
	spans_mapped uintptr

	// Proportional sweep
	pagesInUse        uint64  // pages of spans in stats _MSpanInUse; R/W with mheap.lock
	spanBytesAlloc    uint64  // bytes of spans allocated this cycle; updated atomically
	pagesSwept        uint64  // pages swept this cycle; updated atomically
	sweepPagesPerByte float64 // proportional sweep ratio; written with lock, read without
	// TODO(austin): pagesInUse should be a uintptr, but the 386
	// compiler can't 8-byte align fields.

	// Malloc stats.
	largefree  uint64                  // bytes freed for large objects (>maxsmallsize)
	nlargefree uint64                  // number of frees for large objects (>maxsmallsize)
	nsmallfree [_NumSizeClasses]uint64 // number of frees for small objects (<=maxsmallsize)

	// range of addresses we might see in the heap
	bitmap         uintptr
	bitmap_mapped  uintptr
	arena_start    uintptr
	arena_used     uintptr // always mHeap_Map{Bits,Spans} before updating
	arena_end      uintptr
	arena_reserved bool

	// central free lists for small size classes.
	// the padding makes sure that the MCentrals are
	// spaced CacheLineSize bytes apart, so that each MCentral.lock
	// gets its own cache line.
	central [_NumSizeClasses]struct {
		mcentral mcentral
		pad      [sys.CacheLineSize]byte
	}

	spanalloc             fixalloc // allocator for span*
	cachealloc            fixalloc // allocator for mcache*
	specialfinalizeralloc fixalloc // allocator for specialfinalizer*
	specialprofilealloc   fixalloc // allocator for specialprofile*
	speciallock           mutex    // lock for special record allocators.
}

var mheap_ mheap

// An MSpan is a run of pages.
//
// When a MSpan is in the heap free list, state == MSpanFree
// and heapmap(s->start) == span, heapmap(s->start+s->npages-1) == span.
//
// When a MSpan is allocated, state == MSpanInUse or MSpanStack
// and heapmap(i) == span for all s->start <= i < s->start+s->npages.

// Every MSpan is in one doubly-linked list,
// either one of the MHeap's free lists or one of the
// MCentral's span lists.

// An MSpan representing actual memory has state _MSpanInUse,
// _MSpanStack, or _MSpanFree. Transitions between these states are
// constrained as follows:
//
// * A span may transition from free to in-use or stack during any GC
//   phase.
//
// * During sweeping (gcphase == _GCoff), a span may transition from
//   in-use to free (as a result of sweeping) or stack to free (as a
//   result of stacks being freed).
//
// * During GC (gcphase != _GCoff), a span *must not* transition from
//   stack or in-use to free. Because concurrent GC may read a pointer
//   and then look up its span, the span state must be monotonic.
const (
	_MSpanInUse = iota // allocated for garbage collected heap
	_MSpanStack        // allocated for use by stack allocator
	_MSpanFree
	_MSpanDead
)

// mSpanList heads a linked list of spans.
//
// Linked list structure is based on BSD's "tail queue" data structure.
type mSpanList struct {
	first *mspan  // first span in list, or nil if none
	last  **mspan // last span's next field, or first if none
}

type mspan struct {
	next *mspan     // next span in list, or nil if none
	prev **mspan    // previous span's next field, or list head's first field if none
	list *mSpanList // For debugging. TODO: Remove.

	start    pageID    // starting page number
	npages   uintptr   // number of pages in span
	freelist gclinkptr // list of free objects
	// sweep generation:
	// if sweepgen == h->sweepgen - 2, the span needs sweeping
	// if sweepgen == h->sweepgen - 1, the span is currently being swept
	// if sweepgen == h->sweepgen, the span is swept and ready to use
	// h->sweepgen is incremented by 2 after every GC

	sweepgen    uint32
	divMul      uint32   // for divide by elemsize - divMagic.mul
	ref         uint16   // capacity - number of objects in freelist
	sizeclass   uint8    // size class
	incache     bool     // being used by an mcache
	state       uint8    // mspaninuse etc
	needzero    uint8    // needs to be zeroed before allocation
	divShift    uint8    // for divide by elemsize - divMagic.shift
	divShift2   uint8    // for divide by elemsize - divMagic.shift2
	elemsize    uintptr  // computed from sizeclass or from npages
	unusedsince int64    // first time spotted by gc in mspanfree state
	npreleased  uintptr  // number of pages released to the os
	limit       uintptr  // end of data in span
	speciallock mutex    // guards specials list
	specials    *special // linked list of special records sorted by offset.
	baseMask    uintptr  // if non-0, elemsize is a power of 2, & this will get object allocation base
}

func (s *mspan) base() uintptr {
	return uintptr(s.start << _PageShift)
}

func (s *mspan) layout() (size, n, total uintptr) {
	total = s.npages << _PageShift
	size = s.elemsize
	if size > 0 {
		n = total / size
	}
	return
}

var h_allspans []*mspan // TODO: make this h.allspans once mheap can be defined in Go

// h_spans is a lookup table to map virtual address page IDs to *mspan.
// For allocated spans, their pages map to the span itself.
// For free spans, only the lowest and highest pages map to the span itself.  Internal
// pages map to an arbitrary span.
// For pages that have never been allocated, h_spans entries are nil.
var h_spans []*mspan // TODO: make this h.spans once mheap can be defined in Go

func recordspan(vh unsafe.Pointer, p unsafe.Pointer) {
	h := (*mheap)(vh)
	s := (*mspan)(p)
	if len(h_allspans) >= cap(h_allspans) {
		n := 64 * 1024 / sys.PtrSize
		if n < cap(h_allspans)*3/2 {
			n = cap(h_allspans) * 3 / 2
		}
		var new []*mspan
		sp := (*slice)(unsafe.Pointer(&new))
		sp.array = sysAlloc(uintptr(n)*sys.PtrSize, &memstats.other_sys)
		if sp.array == nil {
			throw("runtime: cannot allocate memory")
		}
		sp.len = len(h_allspans)
		sp.cap = n
		if len(h_allspans) > 0 {
			copy(new, h_allspans)
			// Don't free the old array if it's referenced by sweep.
			// See the comment in mgc.go.
			if h.allspans != mheap_.gcspans {
				sysFree(unsafe.Pointer(h.allspans), uintptr(cap(h_allspans))*sys.PtrSize, &memstats.other_sys)
			}
		}
		h_allspans = new
		h.allspans = (**mspan)(unsafe.Pointer(sp.array))
	}
	h_allspans = append(h_allspans, s)
	h.nspan = uint32(len(h_allspans))
}

// inheap reports whether b is a pointer into a (potentially dead) heap object.
// It returns false for pointers into stack spans.
// Non-preemptible because it is used by write barriers.
//go:nowritebarrier
//go:nosplit
func inheap(b uintptr) bool {
	if b == 0 || b < mheap_.arena_start || b >= mheap_.arena_used {
		return false
	}
	// Not a beginning of a block, consult span table to find the block beginning.
	k := b >> _PageShift
	x := k
	x -= mheap_.arena_start >> _PageShift
	s := h_spans[x]
	if s == nil || pageID(k) < s.start || b >= s.limit || s.state != mSpanInUse {
		return false
	}
	return true
}

// TODO: spanOf and spanOfUnchecked are open-coded in a lot of places.
// Use the functions instead.

// spanOf returns the span of p. If p does not point into the heap or
// no span contains p, spanOf returns nil.
func spanOf(p uintptr) *mspan {
	if p == 0 || p < mheap_.arena_start || p >= mheap_.arena_used {
		return nil
	}
	return spanOfUnchecked(p)
}

// spanOfUnchecked is equivalent to spanOf, but the caller must ensure
// that p points into the heap (that is, mheap_.arena_start <= p <
// mheap_.arena_used).
func spanOfUnchecked(p uintptr) *mspan {
	return h_spans[(p-mheap_.arena_start)>>_PageShift]
}

func mlookup(v uintptr, base *uintptr, size *uintptr, sp **mspan) int32 {
	_g_ := getg()

	_g_.m.mcache.local_nlookup++
	if sys.PtrSize == 4 && _g_.m.mcache.local_nlookup >= 1<<30 {
		// purge cache stats to prevent overflow
		lock(&mheap_.lock)
		purgecachedstats(_g_.m.mcache)
		unlock(&mheap_.lock)
	}

	s := mheap_.lookupMaybe(unsafe.Pointer(v))
	if sp != nil {
		*sp = s
	}
	if s == nil {
		if base != nil {
			*base = 0
		}
		if size != nil {
			*size = 0
		}
		return 0
	}

	p := uintptr(s.start) << _PageShift
	if s.sizeclass == 0 {
		// Large object.
		if base != nil {
			*base = p
		}
		if size != nil {
			*size = s.npages << _PageShift
		}
		return 1
	}

	n := s.elemsize
	if base != nil {
		i := (uintptr(v) - uintptr(p)) / n
		*base = p + i*n
	}
	if size != nil {
		*size = n
	}

	return 1
}

// Initialize the heap.
func (h *mheap) init(spans_size uintptr) {
	h.spanalloc.init(unsafe.Sizeof(mspan{}), recordspan, unsafe.Pointer(h), &memstats.mspan_sys)
	h.cachealloc.init(unsafe.Sizeof(mcache{}), nil, nil, &memstats.mcache_sys)
	h.specialfinalizeralloc.init(unsafe.Sizeof(specialfinalizer{}), nil, nil, &memstats.other_sys)
	h.specialprofilealloc.init(unsafe.Sizeof(specialprofile{}), nil, nil, &memstats.other_sys)

	// h->mapcache needs no init
	for i := range h.free {
		h.free[i].init()
		h.busy[i].init()
	}

	h.freelarge.init()
	h.busylarge.init()
	for i := range h.central {
		h.central[i].mcentral.init(int32(i))
	}

	sp := (*slice)(unsafe.Pointer(&h_spans))
	sp.array = unsafe.Pointer(h.spans)
	sp.len = int(spans_size / sys.PtrSize)
	sp.cap = int(spans_size / sys.PtrSize)
}

// mHeap_MapSpans makes sure that the spans are mapped
// up to the new value of arena_used.
//
// It must be called with the expected new value of arena_used,
// *before* h.arena_used has been updated.
// Waiting to update arena_used until after the memory has been mapped
// avoids faults when other threads try access the bitmap immediately
// after observing the change to arena_used.
func (h *mheap) mapSpans(arena_used uintptr) {
	// Map spans array, PageSize at a time.
	n := arena_used
	n -= h.arena_start
	n = n / _PageSize * sys.PtrSize
	n = round(n, sys.PhysPageSize)
	if h.spans_mapped >= n {
		return
	}
	sysMap(add(unsafe.Pointer(h.spans), h.spans_mapped), n-h.spans_mapped, h.arena_reserved, &memstats.other_sys)
	h.spans_mapped = n
}

// Sweeps spans in list until reclaims at least npages into heap.
// Returns the actual number of pages reclaimed.
func (h *mheap) reclaimList(list *mSpanList, npages uintptr) uintptr {
	n := uintptr(0)
	sg := mheap_.sweepgen
retry:
	for s := list.first; s != nil; s = s.next {
		if s.sweepgen == sg-2 && atomic.Cas(&s.sweepgen, sg-2, sg-1) {
			list.remove(s)
			// swept spans are at the end of the list
			list.insertBack(s)
			unlock(&h.lock)
			snpages := s.npages
			if s.sweep(false) {
				n += snpages
			}
			lock(&h.lock)
			if n >= npages {
				return n
			}
			// the span could have been moved elsewhere
			goto retry
		}
		if s.sweepgen == sg-1 {
			// the span is being sweept by background sweeper, skip
			continue
		}
		// already swept empty span,
		// all subsequent ones must also be either swept or in process of sweeping
		break
	}
	return n
}

// Sweeps and reclaims at least npage pages into heap.
// Called before allocating npage pages.
func (h *mheap) reclaim(npage uintptr) {
	// First try to sweep busy spans with large objects of size >= npage,
	// this has good chances of reclaiming the necessary space.
	for i := int(npage); i < len(h.busy); i++ {
		if h.reclaimList(&h.busy[i], npage) != 0 {
			return // Bingo!
		}
	}

	// Then -- even larger objects.
	if h.reclaimList(&h.busylarge, npage) != 0 {
		return // Bingo!
	}

	// Now try smaller objects.
	// One such object is not enough, so we need to reclaim several of them.
	reclaimed := uintptr(0)
	for i := 0; i < int(npage) && i < len(h.busy); i++ {
		reclaimed += h.reclaimList(&h.busy[i], npage-reclaimed)
		if reclaimed >= npage {
			return
		}
	}

	// Now sweep everything that is not yet swept.
	unlock(&h.lock)
	for {
		n := sweepone()
		if n == ^uintptr(0) { // all spans are swept
			break
		}
		reclaimed += n
		if reclaimed >= npage {
			break
		}
	}
	lock(&h.lock)
}

// Allocate a new span of npage pages from the heap for GC'd memory
// and record its size class in the HeapMap and HeapMapCache.
func (h *mheap) alloc_m(npage uintptr, sizeclass int32, large bool) *mspan {
	_g_ := getg()
	if _g_ != _g_.m.g0 {
		throw("_mheap_alloc not on g0 stack")
	}
	lock(&h.lock)

	// To prevent excessive heap growth, before allocating n pages
	// we need to sweep and reclaim at least n pages.
	if h.sweepdone == 0 {
		// TODO(austin): This tends to sweep a large number of
		// spans in order to find a few completely free spans
		// (for example, in the garbage benchmark, this sweeps
		// ~30x the number of pages its trying to allocate).
		// If GC kept a bit for whether there were any marks
		// in a span, we could release these free spans
		// at the end of GC and eliminate this entirely.
		h.reclaim(npage)
	}

	// transfer stats from cache to global
	memstats.heap_scan += uint64(_g_.m.mcache.local_scan)
	_g_.m.mcache.local_scan = 0
	memstats.tinyallocs += uint64(_g_.m.mcache.local_tinyallocs)
	_g_.m.mcache.local_tinyallocs = 0

	s := h.allocSpanLocked(npage)
	if s != nil {
		// Record span info, because gc needs to be
		// able to map interior pointer to containing span.
		atomic.Store(&s.sweepgen, h.sweepgen)
		s.state = _MSpanInUse
		s.freelist = 0
		s.ref = 0
		s.sizeclass = uint8(sizeclass)
		if sizeclass == 0 {
			s.elemsize = s.npages << _PageShift
			s.divShift = 0
			s.divMul = 0
			s.divShift2 = 0
			s.baseMask = 0
		} else {
			s.elemsize = uintptr(class_to_size[sizeclass])
			m := &class_to_divmagic[sizeclass]
			s.divShift = m.shift
			s.divMul = m.mul
			s.divShift2 = m.shift2
			s.baseMask = m.baseMask
		}

		// update stats, sweep lists
		h.pagesInUse += uint64(npage)
		if large {
			memstats.heap_objects++
			atomic.Xadd64(&memstats.heap_live, int64(npage<<_PageShift))
			// Swept spans are at the end of lists.
			if s.npages < uintptr(len(h.free)) {
				h.busy[s.npages].insertBack(s)
			} else {
				h.busylarge.insertBack(s)
			}
		}
	}
	// heap_scan and heap_live were updated.
	if gcBlackenEnabled != 0 {
		gcController.revise()
	}

	if trace.enabled {
		traceHeapAlloc()
	}

	// h_spans is accessed concurrently without synchronization
	// from other threads. Hence, there must be a store/store
	// barrier here to ensure the writes to h_spans above happen
	// before the caller can publish a pointer p to an object
	// allocated from s. As soon as this happens, the garbage
	// collector running on another processor could read p and
	// look up s in h_spans. The unlock acts as the barrier to
	// order these writes. On the read side, the data dependency
	// between p and the index in h_spans orders the reads.
	unlock(&h.lock)
	return s
}

func (h *mheap) alloc(npage uintptr, sizeclass int32, large bool, needzero bool) *mspan {
	// Don't do any operations that lock the heap on the G stack.
	// It might trigger stack growth, and the stack growth code needs
	// to be able to allocate heap.
	var s *mspan
	systemstack(func() {
		s = h.alloc_m(npage, sizeclass, large)
	})

	if s != nil {
		if needzero && s.needzero != 0 {
			memclr(unsafe.Pointer(s.start<<_PageShift), s.npages<<_PageShift)
		}
		s.needzero = 0
	}
	return s
}

func (h *mheap) allocStack(npage uintptr) *mspan {
	_g_ := getg()
	if _g_ != _g_.m.g0 {
		throw("mheap_allocstack not on g0 stack")
	}
	lock(&h.lock)
	s := h.allocSpanLocked(npage)
	if s != nil {
		s.state = _MSpanStack
		s.freelist = 0
		s.ref = 0
		memstats.stacks_inuse += uint64(s.npages << _PageShift)
	}

	// This unlock acts as a release barrier. See mHeap_Alloc_m.
	unlock(&h.lock)
	return s
}

// Allocates a span of the given size.  h must be locked.
// The returned span has been removed from the
// free list, but its state is still MSpanFree.
func (h *mheap) allocSpanLocked(npage uintptr) *mspan {
	var list *mSpanList
	var s *mspan

	// Try in fixed-size lists up to max.
	for i := int(npage); i < len(h.free); i++ {
		list = &h.free[i]
		if !list.isEmpty() {
			s = list.first
			goto HaveSpan
		}
	}

	// Best fit in list of large spans.
	list = &h.freelarge
	s = h.allocLarge(npage)
	if s == nil {
		if !h.grow(npage) {
			return nil
		}
		s = h.allocLarge(npage)
		if s == nil {
			return nil
		}
	}

HaveSpan:
	// Mark span in use.
	if s.state != _MSpanFree {
		throw("MHeap_AllocLocked - MSpan not free")
	}
	if s.npages < npage {
		throw("MHeap_AllocLocked - bad npages")
	}
	list.remove(s)
	if s.inList() {
		throw("still in list")
	}
	if s.npreleased > 0 {
		sysUsed(unsafe.Pointer(s.start<<_PageShift), s.npages<<_PageShift)
		memstats.heap_released -= uint64(s.npreleased << _PageShift)
		s.npreleased = 0
	}

	if s.npages > npage {
		// Trim extra and put it back in the heap.
		t := (*mspan)(h.spanalloc.alloc())
		t.init(s.start+pageID(npage), s.npages-npage)
		s.npages = npage
		p := uintptr(t.start)
		p -= (h.arena_start >> _PageShift)
		if p > 0 {
			h_spans[p-1] = s
		}
		h_spans[p] = t
		h_spans[p+t.npages-1] = t
		t.needzero = s.needzero
		s.state = _MSpanStack // prevent coalescing with s
		t.state = _MSpanStack
		h.freeSpanLocked(t, false, false, s.unusedsince)
		s.state = _MSpanFree
	}
	s.unusedsince = 0

	p := uintptr(s.start)
	p -= (h.arena_start >> _PageShift)
	for n := uintptr(0); n < npage; n++ {
		h_spans[p+n] = s
	}

	memstats.heap_inuse += uint64(npage << _PageShift)
	memstats.heap_idle -= uint64(npage << _PageShift)

	//println("spanalloc", hex(s.start<<_PageShift))
	if s.inList() {
		throw("still in list")
	}
	return s
}

// Allocate a span of exactly npage pages from the list of large spans.
func (h *mheap) allocLarge(npage uintptr) *mspan {
	return bestFit(&h.freelarge, npage, nil)
}

// Search list for smallest span with >= npage pages.
// If there are multiple smallest spans, take the one
// with the earliest starting address.
func bestFit(list *mSpanList, npage uintptr, best *mspan) *mspan {
	for s := list.first; s != nil; s = s.next {
		if s.npages < npage {
			continue
		}
		if best == nil || s.npages < best.npages || (s.npages == best.npages && s.start < best.start) {
			best = s
		}
	}
	return best
}

// Try to add at least npage pages of memory to the heap,
// returning whether it worked.
//
// h must be locked.
func (h *mheap) grow(npage uintptr) bool {
	// Ask for a big chunk, to reduce the number of mappings
	// the operating system needs to track; also amortizes
	// the overhead of an operating system mapping.
	// Allocate a multiple of 64kB.
	npage = round(npage, (64<<10)/_PageSize)
	ask := npage << _PageShift
	if ask < _HeapAllocChunk {
		ask = _HeapAllocChunk
	}

	v := h.sysAlloc(ask)
	if v == nil {
		if ask > npage<<_PageShift {
			ask = npage << _PageShift
			v = h.sysAlloc(ask)
		}
		if v == nil {
			print("runtime: out of memory: cannot allocate ", ask, "-byte block (", memstats.heap_sys, " in use)\n")
			return false
		}
	}

	// Create a fake "in use" span and free it, so that the
	// right coalescing happens.
	s := (*mspan)(h.spanalloc.alloc())
	s.init(pageID(uintptr(v)>>_PageShift), ask>>_PageShift)
	p := uintptr(s.start)
	p -= (h.arena_start >> _PageShift)
	for i := p; i < p+s.npages; i++ {
		h_spans[i] = s
	}
	atomic.Store(&s.sweepgen, h.sweepgen)
	s.state = _MSpanInUse
	h.pagesInUse += uint64(npage)
	h.freeSpanLocked(s, false, true, 0)
	return true
}

// Look up the span at the given address.
// Address is guaranteed to be in map
// and is guaranteed to be start or end of span.
func (h *mheap) lookup(v unsafe.Pointer) *mspan {
	p := uintptr(v)
	p -= h.arena_start
	return h_spans[p>>_PageShift]
}

// Look up the span at the given address.
// Address is *not* guaranteed to be in map
// and may be anywhere in the span.
// Map entries for the middle of a span are only
// valid for allocated spans.  Free spans may have
// other garbage in their middles, so we have to
// check for that.
func (h *mheap) lookupMaybe(v unsafe.Pointer) *mspan {
	if uintptr(v) < h.arena_start || uintptr(v) >= h.arena_used {
		return nil
	}
	p := uintptr(v) >> _PageShift
	q := p
	q -= h.arena_start >> _PageShift
	s := h_spans[q]
	if s == nil || p < uintptr(s.start) || uintptr(v) >= uintptr(unsafe.Pointer(s.limit)) || s.state != _MSpanInUse {
		return nil
	}
	return s
}

// Free the span back into the heap.
func (h *mheap) freeSpan(s *mspan, acct int32) {
	systemstack(func() {
		mp := getg().m
		lock(&h.lock)
		memstats.heap_scan += uint64(mp.mcache.local_scan)
		mp.mcache.local_scan = 0
		memstats.tinyallocs += uint64(mp.mcache.local_tinyallocs)
		mp.mcache.local_tinyallocs = 0
		if acct != 0 {
			memstats.heap_objects--
		}
		if gcBlackenEnabled != 0 {
			// heap_scan changed.
			gcController.revise()
		}
		h.freeSpanLocked(s, true, true, 0)
		unlock(&h.lock)
	})
}

func (h *mheap) freeStack(s *mspan) {
	_g_ := getg()
	if _g_ != _g_.m.g0 {
		throw("mheap_freestack not on g0 stack")
	}
	s.needzero = 1
	lock(&h.lock)
	memstats.stacks_inuse -= uint64(s.npages << _PageShift)
	h.freeSpanLocked(s, true, true, 0)
	unlock(&h.lock)
}

// s must be on a busy list (h.busy or h.busylarge) or unlinked.
func (h *mheap) freeSpanLocked(s *mspan, acctinuse, acctidle bool, unusedsince int64) {
	switch s.state {
	case _MSpanStack:
		if s.ref != 0 {
			throw("MHeap_FreeSpanLocked - invalid stack free")
		}
	case _MSpanInUse:
		if s.ref != 0 || s.sweepgen != h.sweepgen {
			print("MHeap_FreeSpanLocked - span ", s, " ptr ", hex(s.start<<_PageShift), " ref ", s.ref, " sweepgen ", s.sweepgen, "/", h.sweepgen, "\n")
			throw("MHeap_FreeSpanLocked - invalid free")
		}
		h.pagesInUse -= uint64(s.npages)
	default:
		throw("MHeap_FreeSpanLocked - invalid span state")
	}

	if acctinuse {
		memstats.heap_inuse -= uint64(s.npages << _PageShift)
	}
	if acctidle {
		memstats.heap_idle += uint64(s.npages << _PageShift)
	}
	s.state = _MSpanFree
	if s.inList() {
		h.busyList(s.npages).remove(s)
	}

	// Stamp newly unused spans. The scavenger will use that
	// info to potentially give back some pages to the OS.
	s.unusedsince = unusedsince
	if unusedsince == 0 {
		s.unusedsince = nanotime()
	}
	s.npreleased = 0

	// Coalesce with earlier, later spans.
	p := uintptr(s.start)
	p -= h.arena_start >> _PageShift
	if p > 0 {
		t := h_spans[p-1]
		if t != nil && t.state == _MSpanFree {
			s.start = t.start
			s.npages += t.npages
			s.npreleased = t.npreleased // absorb released pages
			s.needzero |= t.needzero
			p -= t.npages
			h_spans[p] = s
			h.freeList(t.npages).remove(t)
			t.state = _MSpanDead
			h.spanalloc.free(unsafe.Pointer(t))
		}
	}
	if (p+s.npages)*sys.PtrSize < h.spans_mapped {
		t := h_spans[p+s.npages]
		if t != nil && t.state == _MSpanFree {
			s.npages += t.npages
			s.npreleased += t.npreleased
			s.needzero |= t.needzero
			h_spans[p+s.npages-1] = s
			h.freeList(t.npages).remove(t)
			t.state = _MSpanDead
			h.spanalloc.free(unsafe.Pointer(t))
		}
	}

	// Insert s into appropriate list.
	h.freeList(s.npages).insert(s)
}

func (h *mheap) freeList(npages uintptr) *mSpanList {
	if npages < uintptr(len(h.free)) {
		return &h.free[npages]
	}
	return &h.freelarge
}

func (h *mheap) busyList(npages uintptr) *mSpanList {
	if npages < uintptr(len(h.free)) {
		return &h.busy[npages]
	}
	return &h.busylarge
}

func scavengelist(list *mSpanList, now, limit uint64) uintptr {
	if sys.PhysPageSize > _PageSize {
		// golang.org/issue/9993
		// If the physical page size of the machine is larger than
		// our logical heap page size the kernel may round up the
		// amount to be freed to its page size and corrupt the heap
		// pages surrounding the unused block.
		return 0
	}

	if list.isEmpty() {
		return 0
	}

	var sumreleased uintptr
	for s := list.first; s != nil; s = s.next {
		if (now-uint64(s.unusedsince)) > limit && s.npreleased != s.npages {
			released := (s.npages - s.npreleased) << _PageShift
			memstats.heap_released += uint64(released)
			sumreleased += released
			s.npreleased = s.npages
			sysUnused(unsafe.Pointer(s.start<<_PageShift), s.npages<<_PageShift)
		}
	}
	return sumreleased
}

func (h *mheap) scavenge(k int32, now, limit uint64) {
	lock(&h.lock)
	var sumreleased uintptr
	for i := 0; i < len(h.free); i++ {
		sumreleased += scavengelist(&h.free[i], now, limit)
	}
	sumreleased += scavengelist(&h.freelarge, now, limit)
	unlock(&h.lock)

	if debug.gctrace > 0 {
		if sumreleased > 0 {
			print("scvg", k, ": ", sumreleased>>20, " MB released\n")
		}
		// TODO(dvyukov): these stats are incorrect as we don't subtract stack usage from heap.
		// But we can't call ReadMemStats on g0 holding locks.
		print("scvg", k, ": inuse: ", memstats.heap_inuse>>20, ", idle: ", memstats.heap_idle>>20, ", sys: ", memstats.heap_sys>>20, ", released: ", memstats.heap_released>>20, ", consumed: ", (memstats.heap_sys-memstats.heap_released)>>20, " (MB)\n")
	}
}

//go:linkname runtime_debug_freeOSMemory runtime/debug.freeOSMemory
func runtime_debug_freeOSMemory() {
	gcStart(gcForceBlockMode, false)
	systemstack(func() { mheap_.scavenge(-1, ^uint64(0), 0) })
}

// Initialize a new span with the given start and npages.
func (span *mspan) init(start pageID, npages uintptr) {
	span.next = nil
	span.prev = nil
	span.list = nil
	span.start = start
	span.npages = npages
	span.freelist = 0
	span.ref = 0
	span.sizeclass = 0
	span.incache = false
	span.elemsize = 0
	span.state = _MSpanDead
	span.unusedsince = 0
	span.npreleased = 0
	span.speciallock.key = 0
	span.specials = nil
	span.needzero = 0
}

func (span *mspan) inList() bool {
	return span.prev != nil
}

// Initialize an empty doubly-linked list.
func (list *mSpanList) init() {
	list.first = nil
	list.last = &list.first
}

func (list *mSpanList) remove(span *mspan) {
	if span.prev == nil || span.list != list {
		println("failed MSpanList_Remove", span, span.prev, span.list, list)
		throw("MSpanList_Remove")
	}
	if span.next != nil {
		span.next.prev = span.prev
	} else {
		// TODO: After we remove the span.list != list check above,
		// we could at least still check list.last == &span.next here.
		list.last = span.prev
	}
	*span.prev = span.next
	span.next = nil
	span.prev = nil
	span.list = nil
}

func (list *mSpanList) isEmpty() bool {
	return list.first == nil
}

func (list *mSpanList) insert(span *mspan) {
	if span.next != nil || span.prev != nil || span.list != nil {
		println("failed MSpanList_Insert", span, span.next, span.prev, span.list)
		throw("MSpanList_Insert")
	}
	span.next = list.first
	if list.first != nil {
		list.first.prev = &span.next
	} else {
		list.last = &span.next
	}
	list.first = span
	span.prev = &list.first
	span.list = list
}

func (list *mSpanList) insertBack(span *mspan) {
	if span.next != nil || span.prev != nil || span.list != nil {
		println("failed MSpanList_InsertBack", span, span.next, span.prev, span.list)
		throw("MSpanList_InsertBack")
	}
	span.next = nil
	span.prev = list.last
	*list.last = span
	list.last = &span.next
	span.list = list
}

const (
	_KindSpecialFinalizer = 1
	_KindSpecialProfile   = 2
	// Note: The finalizer special must be first because if we're freeing
	// an object, a finalizer special will cause the freeing operation
	// to abort, and we want to keep the other special records around
	// if that happens.
)

type special struct {
	next   *special // linked list in span
	offset uint16   // span offset of object
	kind   byte     // kind of special
}

// Adds the special record s to the list of special records for
// the object p.  All fields of s should be filled in except for
// offset & next, which this routine will fill in.
// Returns true if the special was successfully added, false otherwise.
// (The add will fail only if a record with the same p and s->kind
//  already exists.)
func addspecial(p unsafe.Pointer, s *special) bool {
	span := mheap_.lookupMaybe(p)
	if span == nil {
		throw("addspecial on invalid pointer")
	}

	// Ensure that the span is swept.
	// Sweeping accesses the specials list w/o locks, so we have
	// to synchronize with it. And it's just much safer.
	mp := acquirem()
	span.ensureSwept()

	offset := uintptr(p) - uintptr(span.start<<_PageShift)
	kind := s.kind

	lock(&span.speciallock)

	// Find splice point, check for existing record.
	t := &span.specials
	for {
		x := *t
		if x == nil {
			break
		}
		if offset == uintptr(x.offset) && kind == x.kind {
			unlock(&span.speciallock)
			releasem(mp)
			return false // already exists
		}
		if offset < uintptr(x.offset) || (offset == uintptr(x.offset) && kind < x.kind) {
			break
		}
		t = &x.next
	}

	// Splice in record, fill in offset.
	s.offset = uint16(offset)
	s.next = *t
	*t = s
	unlock(&span.speciallock)
	releasem(mp)

	return true
}

// Removes the Special record of the given kind for the object p.
// Returns the record if the record existed, nil otherwise.
// The caller must FixAlloc_Free the result.
func removespecial(p unsafe.Pointer, kind uint8) *special {
	span := mheap_.lookupMaybe(p)
	if span == nil {
		throw("removespecial on invalid pointer")
	}

	// Ensure that the span is swept.
	// Sweeping accesses the specials list w/o locks, so we have
	// to synchronize with it. And it's just much safer.
	mp := acquirem()
	span.ensureSwept()

	offset := uintptr(p) - uintptr(span.start<<_PageShift)

	lock(&span.speciallock)
	t := &span.specials
	for {
		s := *t
		if s == nil {
			break
		}
		// This function is used for finalizers only, so we don't check for
		// "interior" specials (p must be exactly equal to s->offset).
		if offset == uintptr(s.offset) && kind == s.kind {
			*t = s.next
			unlock(&span.speciallock)
			releasem(mp)
			return s
		}
		t = &s.next
	}
	unlock(&span.speciallock)
	releasem(mp)
	return nil
}

// The described object has a finalizer set for it.
type specialfinalizer struct {
	special special
	fn      *funcval
	nret    uintptr
	fint    *_type
	ot      *ptrtype
}

// Adds a finalizer to the object p.  Returns true if it succeeded.
func addfinalizer(p unsafe.Pointer, f *funcval, nret uintptr, fint *_type, ot *ptrtype) bool {
	lock(&mheap_.speciallock)
	s := (*specialfinalizer)(mheap_.specialfinalizeralloc.alloc())
	unlock(&mheap_.speciallock)
	s.special.kind = _KindSpecialFinalizer
	s.fn = f
	s.nret = nret
	s.fint = fint
	s.ot = ot
	if addspecial(p, &s.special) {
		// This is responsible for maintaining the same
		// GC-related invariants as markrootSpans in any
		// situation where it's possible that markrootSpans
		// has already run but mark termination hasn't yet.
		if gcphase != _GCoff {
			_, base, _ := findObject(p)
			mp := acquirem()
			gcw := &mp.p.ptr().gcw
			// Mark everything reachable from the object
			// so it's retained for the finalizer.
			scanobject(uintptr(base), gcw)
			// Mark the finalizer itself, since the
			// special isn't part of the GC'd heap.
			scanblock(uintptr(unsafe.Pointer(&s.fn)), sys.PtrSize, &oneptrmask[0], gcw)
			if gcBlackenPromptly {
				gcw.dispose()
			}
			releasem(mp)
		}
		return true
	}

	// There was an old finalizer
	lock(&mheap_.speciallock)
	mheap_.specialfinalizeralloc.free(unsafe.Pointer(s))
	unlock(&mheap_.speciallock)
	return false
}

// Removes the finalizer (if any) from the object p.
func removefinalizer(p unsafe.Pointer) {
	s := (*specialfinalizer)(unsafe.Pointer(removespecial(p, _KindSpecialFinalizer)))
	if s == nil {
		return // there wasn't a finalizer to remove
	}
	lock(&mheap_.speciallock)
	mheap_.specialfinalizeralloc.free(unsafe.Pointer(s))
	unlock(&mheap_.speciallock)
}

// The described object is being heap profiled.
type specialprofile struct {
	special special
	b       *bucket
}

// Set the heap profile bucket associated with addr to b.
func setprofilebucket(p unsafe.Pointer, b *bucket) {
	lock(&mheap_.speciallock)
	s := (*specialprofile)(mheap_.specialprofilealloc.alloc())
	unlock(&mheap_.speciallock)
	s.special.kind = _KindSpecialProfile
	s.b = b
	if !addspecial(p, &s.special) {
		throw("setprofilebucket: profile already set")
	}
}

// Do whatever cleanup needs to be done to deallocate s.  It has
// already been unlinked from the MSpan specials list.
func freespecial(s *special, p unsafe.Pointer, size uintptr) {
	switch s.kind {
	case _KindSpecialFinalizer:
		sf := (*specialfinalizer)(unsafe.Pointer(s))
		queuefinalizer(p, sf.fn, sf.nret, sf.fint, sf.ot)
		lock(&mheap_.speciallock)
		mheap_.specialfinalizeralloc.free(unsafe.Pointer(sf))
		unlock(&mheap_.speciallock)
	case _KindSpecialProfile:
		sp := (*specialprofile)(unsafe.Pointer(s))
		mProf_Free(sp.b, size)
		lock(&mheap_.speciallock)
		mheap_.specialprofilealloc.free(unsafe.Pointer(sp))
		unlock(&mheap_.speciallock)
	default:
		throw("bad special kind")
		panic("not reached")
	}
}

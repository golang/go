// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Page heap.
//
// See malloc.h for overview.
//
// When a MSpan is in the heap free list, state == MSpanFree
// and heapmap(s->start) == span, heapmap(s->start+s->npages-1) == span.
//
// When a MSpan is allocated, state == MSpanInUse or MSpanStack
// and heapmap(i) == span for all s->start <= i < s->start+s->npages.

package runtime

import "unsafe"

var h_allspans []*mspan // TODO: make this h.allspans once mheap can be defined in Go
var h_spans []*mspan    // TODO: make this h.spans once mheap can be defined in Go

func recordspan(vh unsafe.Pointer, p unsafe.Pointer) {
	h := (*mheap)(vh)
	s := (*mspan)(p)
	if len(h_allspans) >= cap(h_allspans) {
		n := 64 * 1024 / ptrSize
		if n < cap(h_allspans)*3/2 {
			n = cap(h_allspans) * 3 / 2
		}
		var new []*mspan
		sp := (*slice)(unsafe.Pointer(&new))
		sp.array = (*byte)(sysAlloc(uintptr(n)*ptrSize, &memstats.other_sys))
		if sp.array == nil {
			throw("runtime: cannot allocate memory")
		}
		sp.len = uint(len(h_allspans))
		sp.cap = uint(n)
		if len(h_allspans) > 0 {
			copy(new, h_allspans)
			// Don't free the old array if it's referenced by sweep.
			// See the comment in mgc0.c.
			if h.allspans != mheap_.gcspans {
				sysFree(unsafe.Pointer(h.allspans), uintptr(cap(h_allspans))*ptrSize, &memstats.other_sys)
			}
		}
		h_allspans = new
		h.allspans = (**mspan)(unsafe.Pointer(sp.array))
	}
	h_allspans = append(h_allspans, s)
	h.nspan = uint32(len(h_allspans))
}

// Initialize the heap.
func mHeap_Init(h *mheap, spans_size uintptr) {
	fixAlloc_Init(&h.spanalloc, unsafe.Sizeof(mspan{}), recordspan, unsafe.Pointer(h), &memstats.mspan_sys)
	fixAlloc_Init(&h.cachealloc, unsafe.Sizeof(mcache{}), nil, nil, &memstats.mcache_sys)
	fixAlloc_Init(&h.specialfinalizeralloc, unsafe.Sizeof(specialfinalizer{}), nil, nil, &memstats.other_sys)
	fixAlloc_Init(&h.specialprofilealloc, unsafe.Sizeof(specialprofile{}), nil, nil, &memstats.other_sys)

	// h->mapcache needs no init
	for i := range h.free {
		mSpanList_Init(&h.free[i])
		mSpanList_Init(&h.busy[i])
	}

	mSpanList_Init(&h.freelarge)
	mSpanList_Init(&h.busylarge)
	for i := range h.central {
		mCentral_Init(&h.central[i].mcentral, int32(i))
	}

	sp := (*slice)(unsafe.Pointer(&h_spans))
	sp.array = (*byte)(unsafe.Pointer(h.spans))
	sp.len = uint(spans_size / ptrSize)
	sp.cap = uint(spans_size / ptrSize)
}

func mHeap_MapSpans(h *mheap) {
	// Map spans array, PageSize at a time.
	n := uintptr(unsafe.Pointer(h.arena_used))
	n -= uintptr(unsafe.Pointer(h.arena_start))
	n = n / _PageSize * ptrSize
	n = round(n, _PhysPageSize)
	if h.spans_mapped >= n {
		return
	}
	sysMap(add(unsafe.Pointer(h.spans), h.spans_mapped), n-h.spans_mapped, h.arena_reserved, &memstats.other_sys)
	h.spans_mapped = n
}

// Sweeps spans in list until reclaims at least npages into heap.
// Returns the actual number of pages reclaimed.
func mHeap_ReclaimList(h *mheap, list *mspan, npages uintptr) uintptr {
	n := uintptr(0)
	sg := mheap_.sweepgen
retry:
	for s := list.next; s != list; s = s.next {
		if s.sweepgen == sg-2 && cas(&s.sweepgen, sg-2, sg-1) {
			mSpanList_Remove(s)
			// swept spans are at the end of the list
			mSpanList_InsertBack(list, s)
			unlock(&h.lock)
			if mSpan_Sweep(s, false) {
				// TODO(rsc,dvyukov): This is probably wrong.
				// It is undercounting the number of pages reclaimed.
				// See golang.org/issue/9048.
				// Note that if we want to add the true count of s's pages,
				// we must record that before calling mSpan_Sweep,
				// because if mSpan_Sweep returns true the span has
				// been
				n++
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
func mHeap_Reclaim(h *mheap, npage uintptr) {
	// First try to sweep busy spans with large objects of size >= npage,
	// this has good chances of reclaiming the necessary space.
	for i := int(npage); i < len(h.busy); i++ {
		if mHeap_ReclaimList(h, &h.busy[i], npage) != 0 {
			return // Bingo!
		}
	}

	// Then -- even larger objects.
	if mHeap_ReclaimList(h, &h.busylarge, npage) != 0 {
		return // Bingo!
	}

	// Now try smaller objects.
	// One such object is not enough, so we need to reclaim several of them.
	reclaimed := uintptr(0)
	for i := 0; i < int(npage) && i < len(h.busy); i++ {
		reclaimed += mHeap_ReclaimList(h, &h.busy[i], npage-reclaimed)
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
func mHeap_Alloc_m(h *mheap, npage uintptr, sizeclass int32, large bool) *mspan {
	_g_ := getg()
	if _g_ != _g_.m.g0 {
		throw("_mheap_alloc not on g0 stack")
	}
	lock(&h.lock)

	// To prevent excessive heap growth, before allocating n pages
	// we need to sweep and reclaim at least n pages.
	if h.sweepdone == 0 {
		mHeap_Reclaim(h, npage)
	}

	// transfer stats from cache to global
	memstats.heap_alloc += uint64(_g_.m.mcache.local_cachealloc)
	_g_.m.mcache.local_cachealloc = 0
	memstats.tinyallocs += uint64(_g_.m.mcache.local_tinyallocs)
	_g_.m.mcache.local_tinyallocs = 0

	s := mHeap_AllocSpanLocked(h, npage)
	if s != nil {
		// Record span info, because gc needs to be
		// able to map interior pointer to containing span.
		atomicstore(&s.sweepgen, h.sweepgen)
		s.state = _MSpanInUse
		s.freelist = 0
		s.ref = 0
		s.sizeclass = uint8(sizeclass)
		if sizeclass == 0 {
			s.elemsize = s.npages << _PageShift
		} else {
			s.elemsize = uintptr(class_to_size[sizeclass])
		}

		// update stats, sweep lists
		if large {
			memstats.heap_objects++
			memstats.heap_alloc += uint64(npage << _PageShift)
			// Swept spans are at the end of lists.
			if s.npages < uintptr(len(h.free)) {
				mSpanList_InsertBack(&h.busy[s.npages], s)
			} else {
				mSpanList_InsertBack(&h.busylarge, s)
			}
		}
	}
	unlock(&h.lock)
	return s
}

func mHeap_Alloc(h *mheap, npage uintptr, sizeclass int32, large bool, needzero bool) *mspan {
	// Don't do any operations that lock the heap on the G stack.
	// It might trigger stack growth, and the stack growth code needs
	// to be able to allocate heap.
	var s *mspan
	systemstack(func() {
		s = mHeap_Alloc_m(h, npage, sizeclass, large)
	})

	if s != nil {
		if needzero && s.needzero != 0 {
			memclr(unsafe.Pointer(s.start<<_PageShift), s.npages<<_PageShift)
		}
		s.needzero = 0
	}
	return s
}

func mHeap_AllocStack(h *mheap, npage uintptr) *mspan {
	_g_ := getg()
	if _g_ != _g_.m.g0 {
		throw("mheap_allocstack not on g0 stack")
	}
	lock(&h.lock)
	s := mHeap_AllocSpanLocked(h, npage)
	if s != nil {
		s.state = _MSpanStack
		s.freelist = 0
		s.ref = 0
		memstats.stacks_inuse += uint64(s.npages << _PageShift)
	}
	unlock(&h.lock)
	return s
}

// Allocates a span of the given size.  h must be locked.
// The returned span has been removed from the
// free list, but its state is still MSpanFree.
func mHeap_AllocSpanLocked(h *mheap, npage uintptr) *mspan {
	var s *mspan

	// Try in fixed-size lists up to max.
	for i := int(npage); i < len(h.free); i++ {
		if !mSpanList_IsEmpty(&h.free[i]) {
			s = h.free[i].next
			goto HaveSpan
		}
	}

	// Best fit in list of large spans.
	s = mHeap_AllocLarge(h, npage)
	if s == nil {
		if !mHeap_Grow(h, npage) {
			return nil
		}
		s = mHeap_AllocLarge(h, npage)
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
	mSpanList_Remove(s)
	if s.next != nil || s.prev != nil {
		throw("still in list")
	}
	if s.npreleased > 0 {
		sysUsed((unsafe.Pointer)(s.start<<_PageShift), s.npages<<_PageShift)
		memstats.heap_released -= uint64(s.npreleased << _PageShift)
		s.npreleased = 0
	}

	if s.npages > npage {
		// Trim extra and put it back in the heap.
		t := (*mspan)(fixAlloc_Alloc(&h.spanalloc))
		mSpan_Init(t, s.start+pageID(npage), s.npages-npage)
		s.npages = npage
		p := uintptr(t.start)
		p -= (uintptr(unsafe.Pointer(h.arena_start)) >> _PageShift)
		if p > 0 {
			h_spans[p-1] = s
		}
		h_spans[p] = t
		h_spans[p+t.npages-1] = t
		t.needzero = s.needzero
		s.state = _MSpanStack // prevent coalescing with s
		t.state = _MSpanStack
		mHeap_FreeSpanLocked(h, t, false, false)
		t.unusedsince = s.unusedsince // preserve age (TODO: wrong: t is possibly merged and/or deallocated at this point)
		s.state = _MSpanFree
	}
	s.unusedsince = 0

	p := uintptr(s.start)
	p -= (uintptr(unsafe.Pointer(h.arena_start)) >> _PageShift)
	for n := uintptr(0); n < npage; n++ {
		h_spans[p+n] = s
	}

	memstats.heap_inuse += uint64(npage << _PageShift)
	memstats.heap_idle -= uint64(npage << _PageShift)

	//println("spanalloc", hex(s.start<<_PageShift))
	if s.next != nil || s.prev != nil {
		throw("still in list")
	}
	return s
}

// Allocate a span of exactly npage pages from the list of large spans.
func mHeap_AllocLarge(h *mheap, npage uintptr) *mspan {
	return bestFit(&h.freelarge, npage, nil)
}

// Search list for smallest span with >= npage pages.
// If there are multiple smallest spans, take the one
// with the earliest starting address.
func bestFit(list *mspan, npage uintptr, best *mspan) *mspan {
	for s := list.next; s != list; s = s.next {
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
func mHeap_Grow(h *mheap, npage uintptr) bool {
	// Ask for a big chunk, to reduce the number of mappings
	// the operating system needs to track; also amortizes
	// the overhead of an operating system mapping.
	// Allocate a multiple of 64kB.
	npage = round(npage, (64<<10)/_PageSize)
	ask := npage << _PageShift
	if ask < _HeapAllocChunk {
		ask = _HeapAllocChunk
	}

	v := mHeap_SysAlloc(h, ask)
	if v == nil {
		if ask > npage<<_PageShift {
			ask = npage << _PageShift
			v = mHeap_SysAlloc(h, ask)
		}
		if v == nil {
			print("runtime: out of memory: cannot allocate ", ask, "-byte block (", memstats.heap_sys, " in use)\n")
			return false
		}
	}

	// Create a fake "in use" span and free it, so that the
	// right coalescing happens.
	s := (*mspan)(fixAlloc_Alloc(&h.spanalloc))
	mSpan_Init(s, pageID(uintptr(v)>>_PageShift), ask>>_PageShift)
	p := uintptr(s.start)
	p -= (uintptr(unsafe.Pointer(h.arena_start)) >> _PageShift)
	h_spans[p] = s
	h_spans[p+s.npages-1] = s
	atomicstore(&s.sweepgen, h.sweepgen)
	s.state = _MSpanInUse
	mHeap_FreeSpanLocked(h, s, false, true)
	return true
}

// Look up the span at the given address.
// Address is guaranteed to be in map
// and is guaranteed to be start or end of span.
func mHeap_Lookup(h *mheap, v unsafe.Pointer) *mspan {
	p := uintptr(v)
	p -= uintptr(unsafe.Pointer(h.arena_start))
	return h_spans[p>>_PageShift]
}

// Look up the span at the given address.
// Address is *not* guaranteed to be in map
// and may be anywhere in the span.
// Map entries for the middle of a span are only
// valid for allocated spans.  Free spans may have
// other garbage in their middles, so we have to
// check for that.
func mHeap_LookupMaybe(h *mheap, v unsafe.Pointer) *mspan {
	if uintptr(v) < uintptr(unsafe.Pointer(h.arena_start)) || uintptr(v) >= uintptr(unsafe.Pointer(h.arena_used)) {
		return nil
	}
	p := uintptr(v) >> _PageShift
	q := p
	q -= uintptr(unsafe.Pointer(h.arena_start)) >> _PageShift
	s := h_spans[q]
	if s == nil || p < uintptr(s.start) || uintptr(v) >= uintptr(unsafe.Pointer(s.limit)) || s.state != _MSpanInUse {
		return nil
	}
	return s
}

// Free the span back into the heap.
func mHeap_Free(h *mheap, s *mspan, acct int32) {
	systemstack(func() {
		mp := getg().m
		lock(&h.lock)
		memstats.heap_alloc += uint64(mp.mcache.local_cachealloc)
		mp.mcache.local_cachealloc = 0
		memstats.tinyallocs += uint64(mp.mcache.local_tinyallocs)
		mp.mcache.local_tinyallocs = 0
		if acct != 0 {
			memstats.heap_alloc -= uint64(s.npages << _PageShift)
			memstats.heap_objects--
		}
		mHeap_FreeSpanLocked(h, s, true, true)
		unlock(&h.lock)
	})
}

func mHeap_FreeStack(h *mheap, s *mspan) {
	_g_ := getg()
	if _g_ != _g_.m.g0 {
		throw("mheap_freestack not on g0 stack")
	}
	s.needzero = 1
	lock(&h.lock)
	memstats.stacks_inuse -= uint64(s.npages << _PageShift)
	mHeap_FreeSpanLocked(h, s, true, true)
	unlock(&h.lock)
}

func mHeap_FreeSpanLocked(h *mheap, s *mspan, acctinuse, acctidle bool) {
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
	mSpanList_Remove(s)

	// Stamp newly unused spans. The scavenger will use that
	// info to potentially give back some pages to the OS.
	s.unusedsince = nanotime()
	s.npreleased = 0

	// Coalesce with earlier, later spans.
	p := uintptr(s.start)
	p -= uintptr(unsafe.Pointer(h.arena_start)) >> _PageShift
	if p > 0 {
		t := h_spans[p-1]
		if t != nil && t.state != _MSpanInUse && t.state != _MSpanStack {
			s.start = t.start
			s.npages += t.npages
			s.npreleased = t.npreleased // absorb released pages
			s.needzero |= t.needzero
			p -= t.npages
			h_spans[p] = s
			mSpanList_Remove(t)
			t.state = _MSpanDead
			fixAlloc_Free(&h.spanalloc, (unsafe.Pointer)(t))
		}
	}
	if (p+s.npages)*ptrSize < h.spans_mapped {
		t := h_spans[p+s.npages]
		if t != nil && t.state != _MSpanInUse && t.state != _MSpanStack {
			s.npages += t.npages
			s.npreleased += t.npreleased
			s.needzero |= t.needzero
			h_spans[p+s.npages-1] = s
			mSpanList_Remove(t)
			t.state = _MSpanDead
			fixAlloc_Free(&h.spanalloc, (unsafe.Pointer)(t))
		}
	}

	// Insert s into appropriate list.
	if s.npages < uintptr(len(h.free)) {
		mSpanList_Insert(&h.free[s.npages], s)
	} else {
		mSpanList_Insert(&h.freelarge, s)
	}
}

func scavengelist(list *mspan, now, limit uint64) uintptr {
	if mSpanList_IsEmpty(list) {
		return 0
	}

	var sumreleased uintptr
	for s := list.next; s != list; s = s.next {
		if (now-uint64(s.unusedsince)) > limit && s.npreleased != s.npages {
			released := (s.npages - s.npreleased) << _PageShift
			memstats.heap_released += uint64(released)
			sumreleased += released
			s.npreleased = s.npages
			sysUnused((unsafe.Pointer)(s.start<<_PageShift), s.npages<<_PageShift)
		}
	}
	return sumreleased
}

func mHeap_Scavenge(k int32, now, limit uint64) {
	h := &mheap_
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

func scavenge_m() {
	mHeap_Scavenge(-1, ^uint64(0), 0)
}

// Initialize a new span with the given start and npages.
func mSpan_Init(span *mspan, start pageID, npages uintptr) {
	span.next = nil
	span.prev = nil
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

// Initialize an empty doubly-linked list.
func mSpanList_Init(list *mspan) {
	list.state = _MSpanListHead
	list.next = list
	list.prev = list
}

func mSpanList_Remove(span *mspan) {
	if span.prev == nil && span.next == nil {
		return
	}
	span.prev.next = span.next
	span.next.prev = span.prev
	span.prev = nil
	span.next = nil
}

func mSpanList_IsEmpty(list *mspan) bool {
	return list.next == list
}

func mSpanList_Insert(list *mspan, span *mspan) {
	if span.next != nil || span.prev != nil {
		println("failed MSpanList_Insert", span, span.next, span.prev)
		throw("MSpanList_Insert")
	}
	span.next = list.next
	span.prev = list
	span.next.prev = span
	span.prev.next = span
}

func mSpanList_InsertBack(list *mspan, span *mspan) {
	if span.next != nil || span.prev != nil {
		println("failed MSpanList_InsertBack", span, span.next, span.prev)
		throw("MSpanList_InsertBack")
	}
	span.next = list
	span.prev = list.prev
	span.next.prev = span
	span.prev.next = span
}

// Adds the special record s to the list of special records for
// the object p.  All fields of s should be filled in except for
// offset & next, which this routine will fill in.
// Returns true if the special was successfully added, false otherwise.
// (The add will fail only if a record with the same p and s->kind
//  already exists.)
func addspecial(p unsafe.Pointer, s *special) bool {
	span := mHeap_LookupMaybe(&mheap_, p)
	if span == nil {
		throw("addspecial on invalid pointer")
	}

	// Ensure that the span is swept.
	// GC accesses specials list w/o locks. And it's just much safer.
	mp := acquirem()
	mSpan_EnsureSwept(span)

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
	span := mHeap_LookupMaybe(&mheap_, p)
	if span == nil {
		throw("removespecial on invalid pointer")
	}

	// Ensure that the span is swept.
	// GC accesses specials list w/o locks. And it's just much safer.
	mp := acquirem()
	mSpan_EnsureSwept(span)

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

// Adds a finalizer to the object p.  Returns true if it succeeded.
func addfinalizer(p unsafe.Pointer, f *funcval, nret uintptr, fint *_type, ot *ptrtype) bool {
	lock(&mheap_.speciallock)
	s := (*specialfinalizer)(fixAlloc_Alloc(&mheap_.specialfinalizeralloc))
	unlock(&mheap_.speciallock)
	s.special.kind = _KindSpecialFinalizer
	s.fn = f
	s.nret = nret
	s.fint = fint
	s.ot = ot
	if addspecial(p, &s.special) {
		return true
	}

	// There was an old finalizer
	lock(&mheap_.speciallock)
	fixAlloc_Free(&mheap_.specialfinalizeralloc, (unsafe.Pointer)(s))
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
	fixAlloc_Free(&mheap_.specialfinalizeralloc, (unsafe.Pointer)(s))
	unlock(&mheap_.speciallock)
}

// Set the heap profile bucket associated with addr to b.
func setprofilebucket(p unsafe.Pointer, b *bucket) {
	lock(&mheap_.speciallock)
	s := (*specialprofile)(fixAlloc_Alloc(&mheap_.specialprofilealloc))
	unlock(&mheap_.speciallock)
	s.special.kind = _KindSpecialProfile
	s.b = b
	if !addspecial(p, &s.special) {
		throw("setprofilebucket: profile already set")
	}
}

// Do whatever cleanup needs to be done to deallocate s.  It has
// already been unlinked from the MSpan specials list.
// Returns true if we should keep working on deallocating p.
func freespecial(s *special, p unsafe.Pointer, size uintptr, freed bool) bool {
	switch s.kind {
	case _KindSpecialFinalizer:
		sf := (*specialfinalizer)(unsafe.Pointer(s))
		queuefinalizer(p, sf.fn, sf.nret, sf.fint, sf.ot)
		lock(&mheap_.speciallock)
		fixAlloc_Free(&mheap_.specialfinalizeralloc, (unsafe.Pointer)(sf))
		unlock(&mheap_.speciallock)
		return false // don't free p until finalizer is done
	case _KindSpecialProfile:
		sp := (*specialprofile)(unsafe.Pointer(s))
		mProf_Free(sp.b, size, freed)
		lock(&mheap_.speciallock)
		fixAlloc_Free(&mheap_.specialprofilealloc, (unsafe.Pointer)(sp))
		unlock(&mheap_.speciallock)
		return true
	default:
		throw("bad special kind")
		panic("not reached")
	}
}

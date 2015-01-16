// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Central free lists.
//
// See malloc.h for an overview.
//
// The MCentral doesn't actually contain the list of free objects; the MSpan does.
// Each MCentral is two lists of MSpans: those with free objects (c->nonempty)
// and those that are completely allocated (c->empty).

package runtime

// Initialize a single central free list.
func mCentral_Init(c *mcentral, sizeclass int32) {
	c.sizeclass = sizeclass
	mSpanList_Init(&c.nonempty)
	mSpanList_Init(&c.empty)
}

// Allocate a span to use in an MCache.
func mCentral_CacheSpan(c *mcentral) *mspan {
	lock(&c.lock)
	sg := mheap_.sweepgen
retry:
	var s *mspan
	for s = c.nonempty.next; s != &c.nonempty; s = s.next {
		if s.sweepgen == sg-2 && cas(&s.sweepgen, sg-2, sg-1) {
			mSpanList_Remove(s)
			mSpanList_InsertBack(&c.empty, s)
			unlock(&c.lock)
			mSpan_Sweep(s, true)
			goto havespan
		}
		if s.sweepgen == sg-1 {
			// the span is being swept by background sweeper, skip
			continue
		}
		// we have a nonempty span that does not require sweeping, allocate from it
		mSpanList_Remove(s)
		mSpanList_InsertBack(&c.empty, s)
		unlock(&c.lock)
		goto havespan
	}

	for s = c.empty.next; s != &c.empty; s = s.next {
		if s.sweepgen == sg-2 && cas(&s.sweepgen, sg-2, sg-1) {
			// we have an empty span that requires sweeping,
			// sweep it and see if we can free some space in it
			mSpanList_Remove(s)
			// swept spans are at the end of the list
			mSpanList_InsertBack(&c.empty, s)
			unlock(&c.lock)
			mSpan_Sweep(s, true)
			if s.freelist.ptr() != nil {
				goto havespan
			}
			lock(&c.lock)
			// the span is still empty after sweep
			// it is already in the empty list, so just retry
			goto retry
		}
		if s.sweepgen == sg-1 {
			// the span is being swept by background sweeper, skip
			continue
		}
		// already swept empty span,
		// all subsequent ones must also be either swept or in process of sweeping
		break
	}
	unlock(&c.lock)

	// Replenish central list if empty.
	s = mCentral_Grow(c)
	if s == nil {
		return nil
	}
	lock(&c.lock)
	mSpanList_InsertBack(&c.empty, s)
	unlock(&c.lock)

	// At this point s is a non-empty span, queued at the end of the empty list,
	// c is unlocked.
havespan:
	cap := int32((s.npages << _PageShift) / s.elemsize)
	n := cap - int32(s.ref)
	if n == 0 {
		throw("empty span")
	}
	if s.freelist.ptr() == nil {
		throw("freelist empty")
	}
	s.incache = true
	return s
}

// Return span from an MCache.
func mCentral_UncacheSpan(c *mcentral, s *mspan) {
	lock(&c.lock)

	s.incache = false

	if s.ref == 0 {
		throw("uncaching full span")
	}

	cap := int32((s.npages << _PageShift) / s.elemsize)
	n := cap - int32(s.ref)
	if n > 0 {
		mSpanList_Remove(s)
		mSpanList_Insert(&c.nonempty, s)
	}
	unlock(&c.lock)
}

// Free n objects from a span s back into the central free list c.
// Called during sweep.
// Returns true if the span was returned to heap.  Sets sweepgen to
// the latest generation.
// If preserve=true, don't return the span to heap nor relink in MCentral lists;
// caller takes care of it.
func mCentral_FreeSpan(c *mcentral, s *mspan, n int32, start gclinkptr, end gclinkptr, preserve bool) bool {
	if s.incache {
		throw("freespan into cached span")
	}

	// Add the objects back to s's free list.
	wasempty := s.freelist.ptr() == nil
	end.ptr().next = s.freelist
	s.freelist = start
	s.ref -= uint16(n)

	if preserve {
		// preserve is set only when called from MCentral_CacheSpan above,
		// the span must be in the empty list.
		if s.next == nil {
			throw("can't preserve unlinked span")
		}
		atomicstore(&s.sweepgen, mheap_.sweepgen)
		return false
	}

	lock(&c.lock)

	// Move to nonempty if necessary.
	if wasempty {
		mSpanList_Remove(s)
		mSpanList_Insert(&c.nonempty, s)
	}

	// delay updating sweepgen until here.  This is the signal that
	// the span may be used in an MCache, so it must come after the
	// linked list operations above (actually, just after the
	// lock of c above.)
	atomicstore(&s.sweepgen, mheap_.sweepgen)

	if s.ref != 0 {
		unlock(&c.lock)
		return false
	}

	// s is completely freed, return it to the heap.
	mSpanList_Remove(s)
	s.needzero = 1
	s.freelist = 0
	unlock(&c.lock)
	heapBitsForSpan(s.base()).clearSpan(s.layout())
	mHeap_Free(&mheap_, s, 0)
	return true
}

// Fetch a new span from the heap and carve into objects for the free list.
func mCentral_Grow(c *mcentral) *mspan {
	npages := uintptr(class_to_allocnpages[c.sizeclass])
	size := uintptr(class_to_size[c.sizeclass])
	n := (npages << _PageShift) / size

	s := mHeap_Alloc(&mheap_, npages, c.sizeclass, false, true)
	if s == nil {
		return nil
	}

	p := uintptr(s.start << _PageShift)
	s.limit = p + size*n
	head := gclinkptr(p)
	tail := gclinkptr(p)
	// i==0 iteration already done
	for i := uintptr(1); i < n; i++ {
		p += size
		tail.ptr().next = gclinkptr(p)
		tail = gclinkptr(p)
	}
	if s.freelist.ptr() != nil {
		throw("freelist not empty")
	}
	tail.ptr().next = 0
	s.freelist = head
	heapBitsForSpan(s.base()).initSpan(s.layout())
	return s
}

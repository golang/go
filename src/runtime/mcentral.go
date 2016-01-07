// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Central free lists.
//
// See malloc.go for an overview.
//
// The MCentral doesn't actually contain the list of free objects; the MSpan does.
// Each MCentral is two lists of MSpans: those with free objects (c->nonempty)
// and those that are completely allocated (c->empty).

package runtime

import "runtime/internal/atomic"

// Central list of free objects of a given size.
type mcentral struct {
	lock      mutex
	sizeclass int32
	nonempty  mSpanList // list of spans with a free object
	empty     mSpanList // list of spans with no free objects (or cached in an mcache)
}

// Initialize a single central free list.
func (c *mcentral) init(sizeclass int32) {
	c.sizeclass = sizeclass
	c.nonempty.init()
	c.empty.init()
}

// Allocate a span to use in an MCache.
func (c *mcentral) cacheSpan() *mspan {
	// Deduct credit for this span allocation and sweep if necessary.
	spanBytes := uintptr(class_to_allocnpages[c.sizeclass]) * _PageSize
	deductSweepCredit(spanBytes, 0)

	lock(&c.lock)
	sg := mheap_.sweepgen
retry:
	var s *mspan
	for s = c.nonempty.first; s != nil; s = s.next {
		if s.sweepgen == sg-2 && atomic.Cas(&s.sweepgen, sg-2, sg-1) {
			c.nonempty.remove(s)
			c.empty.insertBack(s)
			unlock(&c.lock)
			s.sweep(true)
			goto havespan
		}
		if s.sweepgen == sg-1 {
			// the span is being swept by background sweeper, skip
			continue
		}
		// we have a nonempty span that does not require sweeping, allocate from it
		c.nonempty.remove(s)
		c.empty.insertBack(s)
		unlock(&c.lock)
		goto havespan
	}

	for s = c.empty.first; s != nil; s = s.next {
		if s.sweepgen == sg-2 && atomic.Cas(&s.sweepgen, sg-2, sg-1) {
			// we have an empty span that requires sweeping,
			// sweep it and see if we can free some space in it
			c.empty.remove(s)
			// swept spans are at the end of the list
			c.empty.insertBack(s)
			unlock(&c.lock)
			s.sweep(true)
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
	s = c.grow()
	if s == nil {
		return nil
	}
	lock(&c.lock)
	c.empty.insertBack(s)
	unlock(&c.lock)

	// At this point s is a non-empty span, queued at the end of the empty list,
	// c is unlocked.
havespan:
	cap := int32((s.npages << _PageShift) / s.elemsize)
	n := cap - int32(s.ref)
	if n == 0 {
		throw("empty span")
	}
	usedBytes := uintptr(s.ref) * s.elemsize
	if usedBytes > 0 {
		reimburseSweepCredit(usedBytes)
	}
	atomic.Xadd64(&memstats.heap_live, int64(spanBytes)-int64(usedBytes))
	if trace.enabled {
		// heap_live changed.
		traceHeapAlloc()
	}
	if gcBlackenEnabled != 0 {
		// heap_live changed.
		gcController.revise()
	}
	if s.freelist.ptr() == nil {
		throw("freelist empty")
	}
	s.incache = true
	return s
}

// Return span from an MCache.
func (c *mcentral) uncacheSpan(s *mspan) {
	lock(&c.lock)

	s.incache = false

	if s.ref == 0 {
		throw("uncaching full span")
	}

	cap := int32((s.npages << _PageShift) / s.elemsize)
	n := cap - int32(s.ref)
	if n > 0 {
		c.empty.remove(s)
		c.nonempty.insert(s)
		// mCentral_CacheSpan conservatively counted
		// unallocated slots in heap_live. Undo this.
		atomic.Xadd64(&memstats.heap_live, -int64(n)*int64(s.elemsize))
	}
	unlock(&c.lock)
}

// Free n objects from a span s back into the central free list c.
// Called during sweep.
// Returns true if the span was returned to heap.  Sets sweepgen to
// the latest generation.
// If preserve=true, don't return the span to heap nor relink in MCentral lists;
// caller takes care of it.
func (c *mcentral) freeSpan(s *mspan, n int32, start gclinkptr, end gclinkptr, preserve bool) bool {
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
		if !s.inList() {
			throw("can't preserve unlinked span")
		}
		atomic.Store(&s.sweepgen, mheap_.sweepgen)
		return false
	}

	lock(&c.lock)

	// Move to nonempty if necessary.
	if wasempty {
		c.empty.remove(s)
		c.nonempty.insert(s)
	}

	// delay updating sweepgen until here.  This is the signal that
	// the span may be used in an MCache, so it must come after the
	// linked list operations above (actually, just after the
	// lock of c above.)
	atomic.Store(&s.sweepgen, mheap_.sweepgen)

	if s.ref != 0 {
		unlock(&c.lock)
		return false
	}

	// s is completely freed, return it to the heap.
	c.nonempty.remove(s)
	s.needzero = 1
	s.freelist = 0
	unlock(&c.lock)
	heapBitsForSpan(s.base()).initSpan(s.layout())
	mheap_.freeSpan(s, 0)
	return true
}

// Fetch a new span from the heap and carve into objects for the free list.
func (c *mcentral) grow() *mspan {
	npages := uintptr(class_to_allocnpages[c.sizeclass])
	size := uintptr(class_to_size[c.sizeclass])
	n := (npages << _PageShift) / size

	s := mheap_.alloc(npages, c.sizeclass, false, true)
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

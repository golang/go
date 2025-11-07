// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
	"internal/runtime/gc"
	"internal/runtime/sys"
	"unsafe"
)

// Per-thread (in Go, per-P) cache for small objects.
// This includes a small object cache and local allocation stats.
// No locking needed because it is per-thread (per-P).
//
// mcaches are allocated from non-GC'd memory, so any heap pointers
// must be specially handled.
type mcache struct {
	_ sys.NotInHeap

	// The following members are accessed on every malloc,
	// so they are grouped here for better caching.
	nextSample  int64   // trigger heap sample after allocating this many bytes
	memProfRate int     // cached mem profile rate, used to detect changes
	scanAlloc   uintptr // bytes of scannable heap allocated

	// Allocator cache for tiny objects w/o pointers.
	// See "Tiny allocator" comment in malloc.go.

	// tiny points to the beginning of the current tiny block, or
	// nil if there is no current tiny block.
	//
	// tiny is a heap pointer. Since mcache is in non-GC'd memory,
	// we handle it by clearing it in releaseAll during mark
	// termination.
	//
	// tinyAllocs is the number of tiny allocations performed
	// by the P that owns this mcache.
	tiny       uintptr
	tinyoffset uintptr
	tinyAllocs uintptr

	// The rest is not accessed on every malloc.

	alloc [numSpanClasses]*mspan // spans to allocate from, indexed by spanClass

	stackcache [_NumStackOrders]stackfreelist

	// flushGen indicates the sweepgen during which this mcache
	// was last flushed. If flushGen != mheap_.sweepgen, the spans
	// in this mcache are stale and need to be flushed so they
	// can be swept. This is done in acquirep.
	flushGen atomic.Uint32
}

// A gclink is a node in a linked list of blocks, like mlink,
// but it is opaque to the garbage collector.
// The GC does not trace the pointers during collection,
// and the compiler does not emit write barriers for assignments
// of gclinkptr values. Code should store references to gclinks
// as gclinkptr, not as *gclink.
type gclink struct {
	next gclinkptr
}

// A gclinkptr is a pointer to a gclink, but it is opaque
// to the garbage collector.
type gclinkptr uintptr

// ptr returns the *gclink form of p.
// The result should be used for accessing fields, not stored
// in other data structures.
func (p gclinkptr) ptr() *gclink {
	return (*gclink)(unsafe.Pointer(p))
}

type stackfreelist struct {
	list gclinkptr // linked list of free stacks
	size uintptr   // total size of stacks in list
}

// dummy mspan that contains no free objects.
var emptymspan mspan

func allocmcache() *mcache {
	var c *mcache
	systemstack(func() {
		lock(&mheap_.lock)
		c = (*mcache)(mheap_.cachealloc.alloc())
		c.flushGen.Store(mheap_.sweepgen)
		unlock(&mheap_.lock)
	})
	for i := range c.alloc {
		c.alloc[i] = &emptymspan
	}
	c.nextSample = nextSample()
	return c
}

// freemcache releases resources associated with this
// mcache and puts the object onto a free list.
//
// In some cases there is no way to simply release
// resources, such as statistics, so donate them to
// a different mcache (the recipient).
func freemcache(c *mcache) {
	systemstack(func() {
		c.releaseAll()
		stackcache_clear(c)

		// NOTE(rsc,rlh): If gcworkbuffree comes back, we need to coordinate
		// with the stealing of gcworkbufs during garbage collection to avoid
		// a race where the workbuf is double-freed.
		// gcworkbuffree(c.gcworkbuf)

		lock(&mheap_.lock)
		mheap_.cachealloc.free(unsafe.Pointer(c))
		unlock(&mheap_.lock)
	})
}

// getMCache is a convenience function which tries to obtain an mcache.
//
// Returns nil if we're not bootstrapping or we don't have a P. The caller's
// P must not change, so we must be in a non-preemptible state.
func getMCache(mp *m) *mcache {
	// Grab the mcache, since that's where stats live.
	pp := mp.p.ptr()
	var c *mcache
	if pp == nil {
		// We will be called without a P while bootstrapping,
		// in which case we use mcache0, which is set in mallocinit.
		// mcache0 is cleared when bootstrapping is complete,
		// by procresize.
		c = mcache0
	} else {
		c = pp.mcache
	}
	return c
}

// refill acquires a new span of span class spc for c. This span will
// have at least one free object. The current span in c must be full.
//
// Must run in a non-preemptible context since otherwise the owner of
// c could change.
func (c *mcache) refill(spc spanClass) {
	// Return the current cached span to the central lists.
	s := c.alloc[spc]

	if s.allocCount != s.nelems {
		throw("refill of span with free space remaining")
	}
	if s != &emptymspan {
		// Mark this span as no longer cached.
		if s.sweepgen != mheap_.sweepgen+3 {
			throw("bad sweepgen in refill")
		}
		mheap_.central[spc].mcentral.uncacheSpan(s)

		// Count up how many slots were used and record it.
		stats := memstats.heapStats.acquire()
		slotsUsed := int64(s.allocCount) - int64(s.allocCountBeforeCache)
		atomic.Xadd64(&stats.smallAllocCount[spc.sizeclass()], slotsUsed)

		// Flush tinyAllocs.
		if spc == tinySpanClass {
			atomic.Xadd64(&stats.tinyAllocCount, int64(c.tinyAllocs))
			c.tinyAllocs = 0
		}
		memstats.heapStats.release()

		// Count the allocs in inconsistent, internal stats.
		bytesAllocated := slotsUsed * int64(s.elemsize)
		gcController.totalAlloc.Add(bytesAllocated)

		// Clear the second allocCount just to be safe.
		s.allocCountBeforeCache = 0
	}

	// Get a new cached span from the central lists.
	s = mheap_.central[spc].mcentral.cacheSpan()
	if s == nil {
		throw("out of memory")
	}

	if s.allocCount == s.nelems {
		throw("span has no free space")
	}

	// Indicate that this span is cached and prevent asynchronous
	// sweeping in the next sweep phase.
	s.sweepgen = mheap_.sweepgen + 3

	// Store the current alloc count for accounting later.
	s.allocCountBeforeCache = s.allocCount

	// Update heapLive and flush scanAlloc.
	//
	// We have not yet allocated anything new into the span, but we
	// assume that all of its slots will get used, so this makes
	// heapLive an overestimate.
	//
	// When the span gets uncached, we'll fix up this overestimate
	// if necessary (see releaseAll).
	//
	// We pick an overestimate here because an underestimate leads
	// the pacer to believe that it's in better shape than it is,
	// which appears to lead to more memory used. See #53738 for
	// more details.
	usedBytes := uintptr(s.allocCount) * s.elemsize
	gcController.update(int64(s.npages*pageSize)-int64(usedBytes), int64(c.scanAlloc))
	c.scanAlloc = 0

	c.alloc[spc] = s
}

// allocLarge allocates a span for a large object.
func (c *mcache) allocLarge(size uintptr, noscan bool) *mspan {
	if size+pageSize < size {
		throw("out of memory")
	}
	npages := size >> gc.PageShift
	if size&pageMask != 0 {
		npages++
	}

	// Deduct credit for this span allocation and sweep if
	// necessary. mHeap_Alloc will also sweep npages, so this only
	// pays the debt down to npage pages.
	deductSweepCredit(npages*pageSize, npages)

	spc := makeSpanClass(0, noscan)
	s := mheap_.alloc(npages, spc)
	if s == nil {
		throw("out of memory")
	}

	// Count the alloc in consistent, external stats.
	stats := memstats.heapStats.acquire()
	atomic.Xadd64(&stats.largeAlloc, int64(npages*pageSize))
	atomic.Xadd64(&stats.largeAllocCount, 1)
	memstats.heapStats.release()

	// Count the alloc in inconsistent, internal stats.
	gcController.totalAlloc.Add(int64(npages * pageSize))

	// Update heapLive.
	gcController.update(int64(s.npages*pageSize), 0)

	// Put the large span in the mcentral swept list so that it's
	// visible to the background sweeper.
	mheap_.central[spc].mcentral.fullSwept(mheap_.sweepgen).push(s)

	// Adjust s.limit down to the object-containing part of the span.
	//
	// This is just to create a slightly tighter bound on the limit.
	// It's totally OK if the garbage collector, in particular
	// conservative scanning, can temporarily observes an inflated
	// limit. It will simply mark the whole object or just skip it
	// since we're in the mark phase anyway.
	s.limit = s.base() + size
	s.initHeapBits()
	return s
}

func (c *mcache) releaseAll() {
	// Take this opportunity to flush scanAlloc.
	scanAlloc := int64(c.scanAlloc)
	c.scanAlloc = 0

	sg := mheap_.sweepgen
	dHeapLive := int64(0)
	for i := range c.alloc {
		s := c.alloc[i]
		if s != &emptymspan {
			slotsUsed := int64(s.allocCount) - int64(s.allocCountBeforeCache)
			s.allocCountBeforeCache = 0

			// Adjust smallAllocCount for whatever was allocated.
			stats := memstats.heapStats.acquire()
			atomic.Xadd64(&stats.smallAllocCount[spanClass(i).sizeclass()], slotsUsed)
			memstats.heapStats.release()

			// Adjust the actual allocs in inconsistent, internal stats.
			// We assumed earlier that the full span gets allocated.
			gcController.totalAlloc.Add(slotsUsed * int64(s.elemsize))

			if s.sweepgen != sg+1 {
				// refill conservatively counted unallocated slots in gcController.heapLive.
				// Undo this.
				//
				// If this span was cached before sweep, then gcController.heapLive was totally
				// recomputed since caching this span, so we don't do this for stale spans.
				dHeapLive -= int64(s.nelems-s.allocCount) * int64(s.elemsize)
			}

			// Release the span to the mcentral.
			mheap_.central[i].mcentral.uncacheSpan(s)
			c.alloc[i] = &emptymspan
		}
	}
	// Clear tinyalloc pool.
	c.tiny = 0
	c.tinyoffset = 0

	// Flush tinyAllocs.
	stats := memstats.heapStats.acquire()
	atomic.Xadd64(&stats.tinyAllocCount, int64(c.tinyAllocs))
	c.tinyAllocs = 0
	memstats.heapStats.release()

	// Update heapLive and heapScan.
	gcController.update(dHeapLive, scanAlloc)
}

// prepareForSweep flushes c if the system has entered a new sweep phase
// since c was populated. This must happen between the sweep phase
// starting and the first allocation from c.
func (c *mcache) prepareForSweep() {
	// Alternatively, instead of making sure we do this on every P
	// between starting the world and allocating on that P, we
	// could leave allocate-black on, allow allocation to continue
	// as usual, use a ragged barrier at the beginning of sweep to
	// ensure all cached spans are swept, and then disable
	// allocate-black. However, with this approach it's difficult
	// to avoid spilling mark bits into the *next* GC cycle.
	sg := mheap_.sweepgen
	flushGen := c.flushGen.Load()
	if flushGen == sg {
		return
	} else if flushGen != sg-2 {
		println("bad flushGen", flushGen, "in prepareForSweep; sweepgen", sg)
		throw("bad flushGen")
	}
	c.releaseAll()
	stackcache_clear(c)
	c.flushGen.Store(mheap_.sweepgen) // Synchronizes with gcStart
}

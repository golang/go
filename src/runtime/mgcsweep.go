// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: sweeping

// The sweeper consists of two different algorithms:
//
// * The object reclaimer finds and frees unmarked slots in spans. It
//   can free a whole span if none of the objects are marked, but that
//   isn't its goal. This can be driven either synchronously by
//   mcentral.cacheSpan for mcentral spans, or asynchronously by
//   sweepone, which looks at all the mcentral lists.
//
// * The span reclaimer looks for spans that contain no marked objects
//   and frees whole spans. This is a separate algorithm because
//   freeing whole spans is the hardest task for the object reclaimer,
//   but is critical when allocating new spans. The entry point for
//   this is mheap_.reclaim and it's driven by a sequential scan of
//   the page marks bitmap in the heap arenas.
//
// Both algorithms ultimately call mspan.sweep, which sweeps a single
// heap span.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

var sweep sweepdata

// State of background sweep.
type sweepdata struct {
	lock    mutex
	g       *g
	parked  bool
	started bool

	nbgsweep    uint32
	npausesweep uint32

	// centralIndex is the current unswept span class.
	// It represents an index into the mcentral span
	// sets. Accessed and updated via its load and
	// update methods. Not protected by a lock.
	//
	// Reset at mark termination.
	// Used by mheap.nextSpanForSweep.
	centralIndex sweepClass
}

// sweepClass is a spanClass and one bit to represent whether we're currently
// sweeping partial or full spans.
type sweepClass uint32

const (
	numSweepClasses            = numSpanClasses * 2
	sweepClassDone  sweepClass = sweepClass(^uint32(0))
)

func (s *sweepClass) load() sweepClass {
	return sweepClass(atomic.Load((*uint32)(s)))
}

func (s *sweepClass) update(sNew sweepClass) {
	// Only update *s if its current value is less than sNew,
	// since *s increases monotonically.
	sOld := s.load()
	for sOld < sNew && !atomic.Cas((*uint32)(s), uint32(sOld), uint32(sNew)) {
		sOld = s.load()
	}
	// TODO(mknyszek): This isn't the only place we have
	// an atomic monotonically increasing counter. It would
	// be nice to have an "atomic max" which is just implemented
	// as the above on most architectures. Some architectures
	// like RISC-V however have native support for an atomic max.
}

func (s *sweepClass) clear() {
	atomic.Store((*uint32)(s), 0)
}

// split returns the underlying span class as well as
// whether we're interested in the full or partial
// unswept lists for that class, indicated as a boolean
// (true means "full").
func (s sweepClass) split() (spc spanClass, full bool) {
	return spanClass(s >> 1), s&1 == 0
}

// nextSpanForSweep finds and pops the next span for sweeping from the
// central sweep buffers. It returns ownership of the span to the caller.
// Returns nil if no such span exists.
func (h *mheap) nextSpanForSweep() *mspan {
	sg := h.sweepgen
	for sc := sweep.centralIndex.load(); sc < numSweepClasses; sc++ {
		spc, full := sc.split()
		c := &h.central[spc].mcentral
		var s *mspan
		if full {
			s = c.fullUnswept(sg).pop()
		} else {
			s = c.partialUnswept(sg).pop()
		}
		if s != nil {
			// Write down that we found something so future sweepers
			// can start from here.
			sweep.centralIndex.update(sc)
			return s
		}
	}
	// Write down that we found nothing.
	sweep.centralIndex.update(sweepClassDone)
	return nil
}

// finishsweep_m ensures that all spans are swept.
//
// The world must be stopped. This ensures there are no sweeps in
// progress.
//
//go:nowritebarrier
func finishsweep_m() {
	assertWorldStopped()

	// Sweeping must be complete before marking commences, so
	// sweep any unswept spans. If this is a concurrent GC, there
	// shouldn't be any spans left to sweep, so this should finish
	// instantly. If GC was forced before the concurrent sweep
	// finished, there may be spans to sweep.
	for sweepone() != ^uintptr(0) {
		sweep.npausesweep++
	}

	// Reset all the unswept buffers, which should be empty.
	// Do this in sweep termination as opposed to mark termination
	// so that we can catch unswept spans and reclaim blocks as
	// soon as possible.
	sg := mheap_.sweepgen
	for i := range mheap_.central {
		c := &mheap_.central[i].mcentral
		c.partialUnswept(sg).reset()
		c.fullUnswept(sg).reset()
	}

	// Sweeping is done, so if the scavenger isn't already awake,
	// wake it up. There's definitely work for it to do at this
	// point.
	wakeScavenger()

	nextMarkBitArenaEpoch()
}

func bgsweep(c chan int) {
	sweep.g = getg()

	lockInit(&sweep.lock, lockRankSweep)
	lock(&sweep.lock)
	sweep.parked = true
	c <- 1
	goparkunlock(&sweep.lock, waitReasonGCSweepWait, traceEvGoBlock, 1)

	for {
		for sweepone() != ^uintptr(0) {
			sweep.nbgsweep++
			Gosched()
		}
		for freeSomeWbufs(true) {
			Gosched()
		}
		lock(&sweep.lock)
		if !isSweepDone() {
			// This can happen if a GC runs between
			// gosweepone returning ^0 above
			// and the lock being acquired.
			unlock(&sweep.lock)
			continue
		}
		sweep.parked = true
		goparkunlock(&sweep.lock, waitReasonGCSweepWait, traceEvGoBlock, 1)
	}
}

// sweepone sweeps some unswept heap span and returns the number of pages returned
// to the heap, or ^uintptr(0) if there was nothing to sweep.
func sweepone() uintptr {
	_g_ := getg()
	sweepRatio := mheap_.sweepPagesPerByte // For debugging

	// increment locks to ensure that the goroutine is not preempted
	// in the middle of sweep thus leaving the span in an inconsistent state for next GC
	_g_.m.locks++
	if atomic.Load(&mheap_.sweepdone) != 0 {
		_g_.m.locks--
		return ^uintptr(0)
	}
	atomic.Xadd(&mheap_.sweepers, +1)

	// Find a span to sweep.
	var s *mspan
	sg := mheap_.sweepgen
	for {
		s = mheap_.nextSpanForSweep()
		if s == nil {
			atomic.Store(&mheap_.sweepdone, 1)
			break
		}
		if state := s.state.get(); state != mSpanInUse {
			// This can happen if direct sweeping already
			// swept this span, but in that case the sweep
			// generation should always be up-to-date.
			if !(s.sweepgen == sg || s.sweepgen == sg+3) {
				print("runtime: bad span s.state=", state, " s.sweepgen=", s.sweepgen, " sweepgen=", sg, "\n")
				throw("non in-use span in unswept list")
			}
			continue
		}
		if s.sweepgen == sg-2 && atomic.Cas(&s.sweepgen, sg-2, sg-1) {
			break
		}
	}

	// Sweep the span we found.
	npages := ^uintptr(0)
	if s != nil {
		npages = s.npages
		if s.sweep(false) {
			// Whole span was freed. Count it toward the
			// page reclaimer credit since these pages can
			// now be used for span allocation.
			atomic.Xadduintptr(&mheap_.reclaimCredit, npages)
		} else {
			// Span is still in-use, so this returned no
			// pages to the heap and the span needs to
			// move to the swept in-use list.
			npages = 0
		}
	}

	// Decrement the number of active sweepers and if this is the
	// last one print trace information.
	if atomic.Xadd(&mheap_.sweepers, -1) == 0 && atomic.Load(&mheap_.sweepdone) != 0 {
		// Since the sweeper is done, move the scavenge gen forward (signalling
		// that there's new work to do) and wake the scavenger.
		//
		// The scavenger is signaled by the last sweeper because once
		// sweeping is done, we will definitely have useful work for
		// the scavenger to do, since the scavenger only runs over the
		// heap once per GC cyle. This update is not done during sweep
		// termination because in some cases there may be a long delay
		// between sweep done and sweep termination (e.g. not enough
		// allocations to trigger a GC) which would be nice to fill in
		// with scavenging work.
		systemstack(func() {
			lock(&mheap_.lock)
			mheap_.pages.scavengeStartGen()
			unlock(&mheap_.lock)
		})
		// Since we might sweep in an allocation path, it's not possible
		// for us to wake the scavenger directly via wakeScavenger, since
		// it could allocate. Ask sysmon to do it for us instead.
		readyForScavenger()

		if debug.gcpacertrace > 0 {
			print("pacer: sweep done at heap size ", memstats.heap_live>>20, "MB; allocated ", (memstats.heap_live-mheap_.sweepHeapLiveBasis)>>20, "MB during sweep; swept ", mheap_.pagesSwept, " pages at ", sweepRatio, " pages/byte\n")
		}
	}
	_g_.m.locks--
	return npages
}

// isSweepDone reports whether all spans are swept or currently being swept.
//
// Note that this condition may transition from false to true at any
// time as the sweeper runs. It may transition from true to false if a
// GC runs; to prevent that the caller must be non-preemptible or must
// somehow block GC progress.
func isSweepDone() bool {
	return mheap_.sweepdone != 0
}

// Returns only when span s has been swept.
//go:nowritebarrier
func (s *mspan) ensureSwept() {
	// Caller must disable preemption.
	// Otherwise when this function returns the span can become unswept again
	// (if GC is triggered on another goroutine).
	_g_ := getg()
	if _g_.m.locks == 0 && _g_.m.mallocing == 0 && _g_ != _g_.m.g0 {
		throw("mspan.ensureSwept: m is not locked")
	}

	sg := mheap_.sweepgen
	spangen := atomic.Load(&s.sweepgen)
	if spangen == sg || spangen == sg+3 {
		return
	}
	// The caller must be sure that the span is a mSpanInUse span.
	if atomic.Cas(&s.sweepgen, sg-2, sg-1) {
		s.sweep(false)
		return
	}
	// unfortunate condition, and we don't have efficient means to wait
	for {
		spangen := atomic.Load(&s.sweepgen)
		if spangen == sg || spangen == sg+3 {
			break
		}
		osyield()
	}
}

// Sweep frees or collects finalizers for blocks not marked in the mark phase.
// It clears the mark bits in preparation for the next GC round.
// Returns true if the span was returned to heap.
// If preserve=true, don't return it to heap nor relink in mcentral lists;
// caller takes care of it.
func (s *mspan) sweep(preserve bool) bool {
	// It's critical that we enter this function with preemption disabled,
	// GC must not start while we are in the middle of this function.
	_g_ := getg()
	if _g_.m.locks == 0 && _g_.m.mallocing == 0 && _g_ != _g_.m.g0 {
		throw("mspan.sweep: m is not locked")
	}
	sweepgen := mheap_.sweepgen
	if state := s.state.get(); state != mSpanInUse || s.sweepgen != sweepgen-1 {
		print("mspan.sweep: state=", state, " sweepgen=", s.sweepgen, " mheap.sweepgen=", sweepgen, "\n")
		throw("mspan.sweep: bad span state")
	}

	if trace.enabled {
		traceGCSweepSpan(s.npages * _PageSize)
	}

	atomic.Xadd64(&mheap_.pagesSwept, int64(s.npages))

	spc := s.spanclass
	size := s.elemsize

	// The allocBits indicate which unmarked objects don't need to be
	// processed since they were free at the end of the last GC cycle
	// and were not allocated since then.
	// If the allocBits index is >= s.freeindex and the bit
	// is not marked then the object remains unallocated
	// since the last GC.
	// This situation is analogous to being on a freelist.

	// Unlink & free special records for any objects we're about to free.
	// Two complications here:
	// 1. An object can have both finalizer and profile special records.
	//    In such case we need to queue finalizer for execution,
	//    mark the object as live and preserve the profile special.
	// 2. A tiny object can have several finalizers setup for different offsets.
	//    If such object is not marked, we need to queue all finalizers at once.
	// Both 1 and 2 are possible at the same time.
	hadSpecials := s.specials != nil
	specialp := &s.specials
	special := *specialp
	for special != nil {
		// A finalizer can be set for an inner byte of an object, find object beginning.
		objIndex := uintptr(special.offset) / size
		p := s.base() + objIndex*size
		mbits := s.markBitsForIndex(objIndex)
		if !mbits.isMarked() {
			// This object is not marked and has at least one special record.
			// Pass 1: see if it has at least one finalizer.
			hasFin := false
			endOffset := p - s.base() + size
			for tmp := special; tmp != nil && uintptr(tmp.offset) < endOffset; tmp = tmp.next {
				if tmp.kind == _KindSpecialFinalizer {
					// Stop freeing of object if it has a finalizer.
					mbits.setMarkedNonAtomic()
					hasFin = true
					break
				}
			}
			// Pass 2: queue all finalizers _or_ handle profile record.
			for special != nil && uintptr(special.offset) < endOffset {
				// Find the exact byte for which the special was setup
				// (as opposed to object beginning).
				p := s.base() + uintptr(special.offset)
				if special.kind == _KindSpecialFinalizer || !hasFin {
					// Splice out special record.
					y := special
					special = special.next
					*specialp = special
					freespecial(y, unsafe.Pointer(p), size)
				} else {
					// This is profile record, but the object has finalizers (so kept alive).
					// Keep special record.
					specialp = &special.next
					special = *specialp
				}
			}
		} else {
			// object is still live: keep special record
			specialp = &special.next
			special = *specialp
		}
	}
	if hadSpecials && s.specials == nil {
		spanHasNoSpecials(s)
	}

	if debug.allocfreetrace != 0 || debug.clobberfree != 0 || raceenabled || msanenabled {
		// Find all newly freed objects. This doesn't have to
		// efficient; allocfreetrace has massive overhead.
		mbits := s.markBitsForBase()
		abits := s.allocBitsForIndex(0)
		for i := uintptr(0); i < s.nelems; i++ {
			if !mbits.isMarked() && (abits.index < s.freeindex || abits.isMarked()) {
				x := s.base() + i*s.elemsize
				if debug.allocfreetrace != 0 {
					tracefree(unsafe.Pointer(x), size)
				}
				if debug.clobberfree != 0 {
					clobberfree(unsafe.Pointer(x), size)
				}
				if raceenabled {
					racefree(unsafe.Pointer(x), size)
				}
				if msanenabled {
					msanfree(unsafe.Pointer(x), size)
				}
			}
			mbits.advance()
			abits.advance()
		}
	}

	// Check for zombie objects.
	if s.freeindex < s.nelems {
		// Everything < freeindex is allocated and hence
		// cannot be zombies.
		//
		// Check the first bitmap byte, where we have to be
		// careful with freeindex.
		obj := s.freeindex
		if (*s.gcmarkBits.bytep(obj / 8)&^*s.allocBits.bytep(obj / 8))>>(obj%8) != 0 {
			s.reportZombies()
		}
		// Check remaining bytes.
		for i := obj/8 + 1; i < divRoundUp(s.nelems, 8); i++ {
			if *s.gcmarkBits.bytep(i)&^*s.allocBits.bytep(i) != 0 {
				s.reportZombies()
			}
		}
	}

	// Count the number of free objects in this span.
	nalloc := uint16(s.countAlloc())
	nfreed := s.allocCount - nalloc
	if nalloc > s.allocCount {
		// The zombie check above should have caught this in
		// more detail.
		print("runtime: nelems=", s.nelems, " nalloc=", nalloc, " previous allocCount=", s.allocCount, " nfreed=", nfreed, "\n")
		throw("sweep increased allocation count")
	}

	s.allocCount = nalloc
	s.freeindex = 0 // reset allocation index to start of span.
	if trace.enabled {
		getg().m.p.ptr().traceReclaimed += uintptr(nfreed) * s.elemsize
	}

	// gcmarkBits becomes the allocBits.
	// get a fresh cleared gcmarkBits in preparation for next GC
	s.allocBits = s.gcmarkBits
	s.gcmarkBits = newMarkBits(s.nelems)

	// Initialize alloc bits cache.
	s.refillAllocCache(0)

	// The span must be in our exclusive ownership until we update sweepgen,
	// check for potential races.
	if state := s.state.get(); state != mSpanInUse || s.sweepgen != sweepgen-1 {
		print("mspan.sweep: state=", state, " sweepgen=", s.sweepgen, " mheap.sweepgen=", sweepgen, "\n")
		throw("mspan.sweep: bad span state after sweep")
	}
	if s.sweepgen == sweepgen+1 || s.sweepgen == sweepgen+3 {
		throw("swept cached span")
	}

	// We need to set s.sweepgen = h.sweepgen only when all blocks are swept,
	// because of the potential for a concurrent free/SetFinalizer.
	//
	// But we need to set it before we make the span available for allocation
	// (return it to heap or mcentral), because allocation code assumes that a
	// span is already swept if available for allocation.
	//
	// Serialization point.
	// At this point the mark bits are cleared and allocation ready
	// to go so release the span.
	atomic.Store(&s.sweepgen, sweepgen)

	if spc.sizeclass() != 0 {
		// Handle spans for small objects.
		if nfreed > 0 {
			// Only mark the span as needing zeroing if we've freed any
			// objects, because a fresh span that had been allocated into,
			// wasn't totally filled, but then swept, still has all of its
			// free slots zeroed.
			s.needzero = 1
			stats := memstats.heapStats.acquire()
			atomic.Xadduintptr(&stats.smallFreeCount[spc.sizeclass()], uintptr(nfreed))
			memstats.heapStats.release()
		}
		if !preserve {
			// The caller may not have removed this span from whatever
			// unswept set its on but taken ownership of the span for
			// sweeping by updating sweepgen. If this span still is in
			// an unswept set, then the mcentral will pop it off the
			// set, check its sweepgen, and ignore it.
			if nalloc == 0 {
				// Free totally free span directly back to the heap.
				mheap_.freeSpan(s)
				return true
			}
			// Return span back to the right mcentral list.
			if uintptr(nalloc) == s.nelems {
				mheap_.central[spc].mcentral.fullSwept(sweepgen).push(s)
			} else {
				mheap_.central[spc].mcentral.partialSwept(sweepgen).push(s)
			}
		}
	} else if !preserve {
		// Handle spans for large objects.
		if nfreed != 0 {
			// Free large object span to heap.

			// NOTE(rsc,dvyukov): The original implementation of efence
			// in CL 22060046 used sysFree instead of sysFault, so that
			// the operating system would eventually give the memory
			// back to us again, so that an efence program could run
			// longer without running out of memory. Unfortunately,
			// calling sysFree here without any kind of adjustment of the
			// heap data structures means that when the memory does
			// come back to us, we have the wrong metadata for it, either in
			// the mspan structures or in the garbage collection bitmap.
			// Using sysFault here means that the program will run out of
			// memory fairly quickly in efence mode, but at least it won't
			// have mysterious crashes due to confused memory reuse.
			// It should be possible to switch back to sysFree if we also
			// implement and then call some kind of mheap.deleteSpan.
			if debug.efence > 0 {
				s.limit = 0 // prevent mlookup from finding this span
				sysFault(unsafe.Pointer(s.base()), size)
			} else {
				mheap_.freeSpan(s)
			}
			stats := memstats.heapStats.acquire()
			atomic.Xadduintptr(&stats.largeFreeCount, 1)
			atomic.Xadduintptr(&stats.largeFree, size)
			memstats.heapStats.release()
			return true
		}

		// Add a large span directly onto the full+swept list.
		mheap_.central[spc].mcentral.fullSwept(sweepgen).push(s)
	}
	return false
}

// reportZombies reports any marked but free objects in s and throws.
//
// This generally means one of the following:
//
// 1. User code converted a pointer to a uintptr and then back
// unsafely, and a GC ran while the uintptr was the only reference to
// an object.
//
// 2. User code (or a compiler bug) constructed a bad pointer that
// points to a free slot, often a past-the-end pointer.
//
// 3. The GC two cycles ago missed a pointer and freed a live object,
// but it was still live in the last cycle, so this GC cycle found a
// pointer to that object and marked it.
func (s *mspan) reportZombies() {
	printlock()
	print("runtime: marked free object in span ", s, ", elemsize=", s.elemsize, " freeindex=", s.freeindex, " (bad use of unsafe.Pointer? try -d=checkptr)\n")
	mbits := s.markBitsForBase()
	abits := s.allocBitsForIndex(0)
	for i := uintptr(0); i < s.nelems; i++ {
		addr := s.base() + i*s.elemsize
		print(hex(addr))
		alloc := i < s.freeindex || abits.isMarked()
		if alloc {
			print(" alloc")
		} else {
			print(" free ")
		}
		if mbits.isMarked() {
			print(" marked  ")
		} else {
			print(" unmarked")
		}
		zombie := mbits.isMarked() && !alloc
		if zombie {
			print(" zombie")
		}
		print("\n")
		if zombie {
			length := s.elemsize
			if length > 1024 {
				length = 1024
			}
			hexdumpWords(addr, addr+length, nil)
		}
		mbits.advance()
		abits.advance()
	}
	throw("found pointer to free object")
}

// deductSweepCredit deducts sweep credit for allocating a span of
// size spanBytes. This must be performed *before* the span is
// allocated to ensure the system has enough credit. If necessary, it
// performs sweeping to prevent going in to debt. If the caller will
// also sweep pages (e.g., for a large allocation), it can pass a
// non-zero callerSweepPages to leave that many pages unswept.
//
// deductSweepCredit makes a worst-case assumption that all spanBytes
// bytes of the ultimately allocated span will be available for object
// allocation.
//
// deductSweepCredit is the core of the "proportional sweep" system.
// It uses statistics gathered by the garbage collector to perform
// enough sweeping so that all pages are swept during the concurrent
// sweep phase between GC cycles.
//
// mheap_ must NOT be locked.
func deductSweepCredit(spanBytes uintptr, callerSweepPages uintptr) {
	if mheap_.sweepPagesPerByte == 0 {
		// Proportional sweep is done or disabled.
		return
	}

	if trace.enabled {
		traceGCSweepStart()
	}

retry:
	sweptBasis := atomic.Load64(&mheap_.pagesSweptBasis)

	// Fix debt if necessary.
	newHeapLive := uintptr(atomic.Load64(&memstats.heap_live)-mheap_.sweepHeapLiveBasis) + spanBytes
	pagesTarget := int64(mheap_.sweepPagesPerByte*float64(newHeapLive)) - int64(callerSweepPages)
	for pagesTarget > int64(atomic.Load64(&mheap_.pagesSwept)-sweptBasis) {
		if sweepone() == ^uintptr(0) {
			mheap_.sweepPagesPerByte = 0
			break
		}
		if atomic.Load64(&mheap_.pagesSweptBasis) != sweptBasis {
			// Sweep pacing changed. Recompute debt.
			goto retry
		}
	}

	if trace.enabled {
		traceGCSweepDone()
	}
}

// clobberfree sets the memory content at x to bad content, for debugging
// purposes.
func clobberfree(x unsafe.Pointer, size uintptr) {
	// size (span.elemsize) is always a multiple of 4.
	for i := uintptr(0); i < size; i += 4 {
		*(*uint32)(add(x, i)) = 0xdeadbeef
	}
}

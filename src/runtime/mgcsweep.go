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

	// active tracks outstanding sweepers and the sweep
	// termination condition.
	active activeSweep

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

const sweepDrainedMask = 1 << 31

// activeSweep is a type that captures whether sweeping
// is done, and whether there are any outstanding sweepers.
//
// Every potential sweeper must call begin() before they look
// for work, and end() after they've finished sweeping.
type activeSweep struct {
	// state is divided into two parts.
	//
	// The top bit (masked by sweepDrainedMask) is a boolean
	// value indicating whether all the sweep work has been
	// drained from the queue.
	//
	// The rest of the bits are a counter, indicating the
	// number of outstanding concurrent sweepers.
	state atomic.Uint32
}

// begin registers a new sweeper. Returns a sweepLocker
// for acquiring spans for sweeping. Any outstanding sweeper blocks
// sweep termination.
//
// If the sweepLocker is invalid, the caller can be sure that all
// outstanding sweep work has been drained, so there is nothing left
// to sweep. Note that there may be sweepers currently running, so
// this does not indicate that all sweeping has completed.
//
// Even if the sweepLocker is invalid, its sweepGen is always valid.
func (a *activeSweep) begin() sweepLocker {
	for {
		state := a.state.Load()
		if state&sweepDrainedMask != 0 {
			return sweepLocker{mheap_.sweepgen, false}
		}
		if a.state.CompareAndSwap(state, state+1) {
			return sweepLocker{mheap_.sweepgen, true}
		}
	}
}

// end deregisters a sweeper. Must be called once for each time
// begin is called if the sweepLocker is valid.
func (a *activeSweep) end(sl sweepLocker) {
	if sl.sweepGen != mheap_.sweepgen {
		throw("sweeper left outstanding across sweep generations")
	}
	for {
		state := a.state.Load()
		if (state&^sweepDrainedMask)-1 >= sweepDrainedMask {
			throw("mismatched begin/end of activeSweep")
		}
		if a.state.CompareAndSwap(state, state-1) {
			if state != sweepDrainedMask {
				return
			}
			if debug.gcpacertrace > 0 {
				print("pacer: sweep done at heap size ", gcController.heapLive>>20, "MB; allocated ", (gcController.heapLive-mheap_.sweepHeapLiveBasis)>>20, "MB during sweep; swept ", mheap_.pagesSwept.Load(), " pages at ", mheap_.sweepPagesPerByte, " pages/byte\n")
			}
			return
		}
	}
}

// markDrained marks the active sweep cycle as having drained
// all remaining work. This is safe to be called concurrently
// with all other methods of activeSweep, though may race.
//
// Returns true if this call was the one that actually performed
// the mark.
func (a *activeSweep) markDrained() bool {
	for {
		state := a.state.Load()
		if state&sweepDrainedMask != 0 {
			return false
		}
		if a.state.CompareAndSwap(state, state|sweepDrainedMask) {
			return true
		}
	}
}

// sweepers returns the current number of active sweepers.
func (a *activeSweep) sweepers() uint32 {
	return a.state.Load() &^ sweepDrainedMask
}

// isDone returns true if all sweep work has been drained and no more
// outstanding sweepers exist. That is, when the sweep phase is
// completely done.
func (a *activeSweep) isDone() bool {
	return a.state.Load() == sweepDrainedMask
}

// reset sets up the activeSweep for the next sweep cycle.
//
// The world must be stopped.
func (a *activeSweep) reset() {
	assertWorldStopped()
	a.state.Store(0)
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

	// Make sure there aren't any outstanding sweepers left.
	// At this point, with the world stopped, it means one of two
	// things. Either we were able to preempt a sweeper, or that
	// a sweeper didn't call sweep.active.end when it should have.
	// Both cases indicate a bug, so throw.
	if sweep.active.sweepers() != 0 {
		throw("active sweepers found at start of mark phase")
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

// sweepLocker acquires sweep ownership of spans.
type sweepLocker struct {
	// sweepGen is the sweep generation of the heap.
	sweepGen uint32
	valid    bool
}

// sweepLocked represents sweep ownership of a span.
type sweepLocked struct {
	*mspan
}

// tryAcquire attempts to acquire sweep ownership of span s. If it
// successfully acquires ownership, it blocks sweep completion.
func (l *sweepLocker) tryAcquire(s *mspan) (sweepLocked, bool) {
	if !l.valid {
		throw("use of invalid sweepLocker")
	}
	// Check before attempting to CAS.
	if atomic.Load(&s.sweepgen) != l.sweepGen-2 {
		return sweepLocked{}, false
	}
	// Attempt to acquire sweep ownership of s.
	if !atomic.Cas(&s.sweepgen, l.sweepGen-2, l.sweepGen-1) {
		return sweepLocked{}, false
	}
	return sweepLocked{s}, true
}

// sweepone sweeps some unswept heap span and returns the number of pages returned
// to the heap, or ^uintptr(0) if there was nothing to sweep.
func sweepone() uintptr {
	gp := getg()

	// Increment locks to ensure that the goroutine is not preempted
	// in the middle of sweep thus leaving the span in an inconsistent state for next GC
	gp.m.locks++

	// TODO(austin): sweepone is almost always called in a loop;
	// lift the sweepLocker into its callers.
	sl := sweep.active.begin()
	if !sl.valid {
		gp.m.locks--
		return ^uintptr(0)
	}

	// Find a span to sweep.
	npages := ^uintptr(0)
	var noMoreWork bool
	for {
		s := mheap_.nextSpanForSweep()
		if s == nil {
			noMoreWork = sweep.active.markDrained()
			break
		}
		if state := s.state.get(); state != mSpanInUse {
			// This can happen if direct sweeping already
			// swept this span, but in that case the sweep
			// generation should always be up-to-date.
			if !(s.sweepgen == sl.sweepGen || s.sweepgen == sl.sweepGen+3) {
				print("runtime: bad span s.state=", state, " s.sweepgen=", s.sweepgen, " sweepgen=", sl.sweepGen, "\n")
				throw("non in-use span in unswept list")
			}
			continue
		}
		if s, ok := sl.tryAcquire(s); ok {
			// Sweep the span we found.
			npages = s.npages
			if s.sweep(false) {
				// Whole span was freed. Count it toward the
				// page reclaimer credit since these pages can
				// now be used for span allocation.
				mheap_.reclaimCredit.Add(npages)
			} else {
				// Span is still in-use, so this returned no
				// pages to the heap and the span needs to
				// move to the swept in-use list.
				npages = 0
			}
			break
		}
	}
	sweep.active.end(sl)

	if noMoreWork {
		// The sweep list is empty. There may still be
		// concurrent sweeps running, but we're at least very
		// close to done sweeping.

		// Move the scavenge gen forward (signalling
		// that there's new work to do) and wake the scavenger.
		//
		// The scavenger is signaled by the last sweeper because once
		// sweeping is done, we will definitely have useful work for
		// the scavenger to do, since the scavenger only runs over the
		// heap once per GC cycle. This update is not done during sweep
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
	}

	gp.m.locks--
	return npages
}

// isSweepDone reports whether all spans are swept.
//
// Note that this condition may transition from false to true at any
// time as the sweeper runs. It may transition from true to false if a
// GC runs; to prevent that the caller must be non-preemptible or must
// somehow block GC progress.
func isSweepDone() bool {
	return sweep.active.isDone()
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

	// If this operation fails, then that means that there are
	// no more spans to be swept. In this case, either s has already
	// been swept, or is about to be acquired for sweeping and swept.
	sl := sweep.active.begin()
	if sl.valid {
		// The caller must be sure that the span is a mSpanInUse span.
		if s, ok := sl.tryAcquire(s); ok {
			s.sweep(false)
			sweep.active.end(sl)
			return
		}
		sweep.active.end(sl)
	}

	// Unfortunately we can't sweep the span ourselves. Somebody else
	// got to it first. We don't have efficient means to wait, but that's
	// OK, it will be swept fairly soon.
	for {
		spangen := atomic.Load(&s.sweepgen)
		if spangen == sl.sweepGen || spangen == sl.sweepGen+3 {
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
func (sl *sweepLocked) sweep(preserve bool) bool {
	// It's critical that we enter this function with preemption disabled,
	// GC must not start while we are in the middle of this function.
	_g_ := getg()
	if _g_.m.locks == 0 && _g_.m.mallocing == 0 && _g_ != _g_.m.g0 {
		throw("mspan.sweep: m is not locked")
	}

	s := sl.mspan
	if !preserve {
		// We'll release ownership of this span. Nil it out to
		// prevent the caller from accidentally using it.
		sl.mspan = nil
	}

	sweepgen := mheap_.sweepgen
	if state := s.state.get(); state != mSpanInUse || s.sweepgen != sweepgen-1 {
		print("mspan.sweep: state=", state, " sweepgen=", s.sweepgen, " mheap.sweepgen=", sweepgen, "\n")
		throw("mspan.sweep: bad span state")
	}

	if trace.enabled {
		traceGCSweepSpan(s.npages * _PageSize)
	}

	mheap_.pagesSwept.Add(int64(s.npages))

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
	siter := newSpecialsIter(s)
	for siter.valid() {
		// A finalizer can be set for an inner byte of an object, find object beginning.
		objIndex := uintptr(siter.s.offset) / size
		p := s.base() + objIndex*size
		mbits := s.markBitsForIndex(objIndex)
		if !mbits.isMarked() {
			// This object is not marked and has at least one special record.
			// Pass 1: see if it has at least one finalizer.
			hasFin := false
			endOffset := p - s.base() + size
			for tmp := siter.s; tmp != nil && uintptr(tmp.offset) < endOffset; tmp = tmp.next {
				if tmp.kind == _KindSpecialFinalizer {
					// Stop freeing of object if it has a finalizer.
					mbits.setMarkedNonAtomic()
					hasFin = true
					break
				}
			}
			// Pass 2: queue all finalizers _or_ handle profile record.
			for siter.valid() && uintptr(siter.s.offset) < endOffset {
				// Find the exact byte for which the special was setup
				// (as opposed to object beginning).
				special := siter.s
				p := s.base() + uintptr(special.offset)
				if special.kind == _KindSpecialFinalizer || !hasFin {
					siter.unlinkAndNext()
					freeSpecial(special, unsafe.Pointer(p), size)
				} else {
					// The object has finalizers, so we're keeping it alive.
					// All other specials only apply when an object is freed,
					// so just keep the special record.
					siter.next()
				}
			}
		} else {
			// object is still live
			if siter.s.kind == _KindSpecialReachable {
				special := siter.unlinkAndNext()
				(*specialReachable)(unsafe.Pointer(special)).reachable = true
				freeSpecial(special, unsafe.Pointer(p), size)
			} else {
				// keep special record
				siter.next()
			}
		}
	}
	if hadSpecials && s.specials == nil {
		spanHasNoSpecials(s)
	}

	if debug.allocfreetrace != 0 || debug.clobberfree != 0 || raceenabled || msanenabled || asanenabled {
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
				if asanenabled {
					asanpoison(unsafe.Pointer(x), size)
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
	sweptBasis := mheap_.pagesSweptBasis.Load()

	// Fix debt if necessary.
	newHeapLive := uintptr(atomic.Load64(&gcController.heapLive)-mheap_.sweepHeapLiveBasis) + spanBytes
	pagesTarget := int64(mheap_.sweepPagesPerByte*float64(newHeapLive)) - int64(callerSweepPages)
	for pagesTarget > int64(mheap_.pagesSwept.Load()-sweptBasis) {
		if sweepone() == ^uintptr(0) {
			mheap_.sweepPagesPerByte = 0
			break
		}
		if mheap_.pagesSweptBasis.Load() != sweptBasis {
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

// gcPaceSweeper updates the sweeper's pacing parameters.
//
// Must be called whenever the GC's pacing is updated.
//
// The world must be stopped, or mheap_.lock must be held.
func gcPaceSweeper(trigger uint64) {
	assertWorldStoppedOrLockHeld(&mheap_.lock)

	// Update sweep pacing.
	if isSweepDone() {
		mheap_.sweepPagesPerByte = 0
	} else {
		// Concurrent sweep needs to sweep all of the in-use
		// pages by the time the allocated heap reaches the GC
		// trigger. Compute the ratio of in-use pages to sweep
		// per byte allocated, accounting for the fact that
		// some might already be swept.
		heapLiveBasis := atomic.Load64(&gcController.heapLive)
		heapDistance := int64(trigger) - int64(heapLiveBasis)
		// Add a little margin so rounding errors and
		// concurrent sweep are less likely to leave pages
		// unswept when GC starts.
		heapDistance -= 1024 * 1024
		if heapDistance < _PageSize {
			// Avoid setting the sweep ratio extremely high
			heapDistance = _PageSize
		}
		pagesSwept := mheap_.pagesSwept.Load()
		pagesInUse := mheap_.pagesInUse.Load()
		sweepDistancePages := int64(pagesInUse) - int64(pagesSwept)
		if sweepDistancePages <= 0 {
			mheap_.sweepPagesPerByte = 0
		} else {
			mheap_.sweepPagesPerByte = float64(sweepDistancePages) / float64(heapDistance)
			mheap_.sweepHeapLiveBasis = heapLiveBasis
			// Write pagesSweptBasis last, since this
			// signals concurrent sweeps to recompute
			// their debt.
			mheap_.pagesSweptBasis.Store(pagesSwept)
		}
	}
}

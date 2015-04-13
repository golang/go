// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: sweeping

package runtime

import "unsafe"

var sweep sweepdata

// State of background sweep.
type sweepdata struct {
	lock    mutex
	g       *g
	parked  bool
	started bool

	spanidx uint32 // background sweeper position

	nbgsweep    uint32
	npausesweep uint32
}

//go:nowritebarrier
func finishsweep_m() {
	// The world is stopped so we should be able to complete the sweeps
	// quickly.
	for sweepone() != ^uintptr(0) {
		sweep.npausesweep++
	}

	// There may be some other spans being swept concurrently that
	// we need to wait for. If finishsweep_m is done with the world stopped
	// this code is not required.
	sg := mheap_.sweepgen
	for _, s := range work.spans {
		if s.sweepgen != sg && s.state == _MSpanInUse {
			mSpan_EnsureSwept(s)
		}
	}
}

func bgsweep(c chan int) {
	sweep.g = getg()

	lock(&sweep.lock)
	sweep.parked = true
	c <- 1
	goparkunlock(&sweep.lock, "GC sweep wait", traceEvGoBlock, 1)

	for {
		for gosweepone() != ^uintptr(0) {
			sweep.nbgsweep++
			Gosched()
		}
		lock(&sweep.lock)
		if !gosweepdone() {
			// This can happen if a GC runs between
			// gosweepone returning ^0 above
			// and the lock being acquired.
			unlock(&sweep.lock)
			continue
		}
		sweep.parked = true
		goparkunlock(&sweep.lock, "GC sweep wait", traceEvGoBlock, 1)
	}
}

// sweeps one span
// returns number of pages returned to heap, or ^uintptr(0) if there is nothing to sweep
//go:nowritebarrier
func sweepone() uintptr {
	_g_ := getg()

	// increment locks to ensure that the goroutine is not preempted
	// in the middle of sweep thus leaving the span in an inconsistent state for next GC
	_g_.m.locks++
	sg := mheap_.sweepgen
	for {
		idx := xadd(&sweep.spanidx, 1) - 1
		if idx >= uint32(len(work.spans)) {
			mheap_.sweepdone = 1
			_g_.m.locks--
			return ^uintptr(0)
		}
		s := work.spans[idx]
		if s.state != mSpanInUse {
			s.sweepgen = sg
			continue
		}
		if s.sweepgen != sg-2 || !cas(&s.sweepgen, sg-2, sg-1) {
			continue
		}
		npages := s.npages
		if !mSpan_Sweep(s, false) {
			npages = 0
		}
		_g_.m.locks--
		return npages
	}
}

//go:nowritebarrier
func gosweepone() uintptr {
	var ret uintptr
	systemstack(func() {
		ret = sweepone()
	})
	return ret
}

//go:nowritebarrier
func gosweepdone() bool {
	return mheap_.sweepdone != 0
}

// Returns only when span s has been swept.
//go:nowritebarrier
func mSpan_EnsureSwept(s *mspan) {
	// Caller must disable preemption.
	// Otherwise when this function returns the span can become unswept again
	// (if GC is triggered on another goroutine).
	_g_ := getg()
	if _g_.m.locks == 0 && _g_.m.mallocing == 0 && _g_ != _g_.m.g0 {
		throw("MSpan_EnsureSwept: m is not locked")
	}

	sg := mheap_.sweepgen
	if atomicload(&s.sweepgen) == sg {
		return
	}
	// The caller must be sure that the span is a MSpanInUse span.
	if cas(&s.sweepgen, sg-2, sg-1) {
		mSpan_Sweep(s, false)
		return
	}
	// unfortunate condition, and we don't have efficient means to wait
	for atomicload(&s.sweepgen) != sg {
		osyield()
	}
}

// Sweep frees or collects finalizers for blocks not marked in the mark phase.
// It clears the mark bits in preparation for the next GC round.
// Returns true if the span was returned to heap.
// If preserve=true, don't return it to heap nor relink in MCentral lists;
// caller takes care of it.
//TODO go:nowritebarrier
func mSpan_Sweep(s *mspan, preserve bool) bool {
	// It's critical that we enter this function with preemption disabled,
	// GC must not start while we are in the middle of this function.
	_g_ := getg()
	if _g_.m.locks == 0 && _g_.m.mallocing == 0 && _g_ != _g_.m.g0 {
		throw("MSpan_Sweep: m is not locked")
	}
	sweepgen := mheap_.sweepgen
	if s.state != mSpanInUse || s.sweepgen != sweepgen-1 {
		print("MSpan_Sweep: state=", s.state, " sweepgen=", s.sweepgen, " mheap.sweepgen=", sweepgen, "\n")
		throw("MSpan_Sweep: bad span state")
	}

	if trace.enabled {
		traceGCSweepStart()
	}

	cl := s.sizeclass
	size := s.elemsize
	res := false
	nfree := 0

	var head, end gclinkptr

	c := _g_.m.mcache
	sweepgenset := false

	// Mark any free objects in this span so we don't collect them.
	for link := s.freelist; link.ptr() != nil; link = link.ptr().next {
		heapBitsForAddr(uintptr(link)).setMarkedNonAtomic()
	}

	// Unlink & free special records for any objects we're about to free.
	specialp := &s.specials
	special := *specialp
	for special != nil {
		// A finalizer can be set for an inner byte of an object, find object beginning.
		p := uintptr(s.start<<_PageShift) + uintptr(special.offset)/size*size
		hbits := heapBitsForAddr(p)
		if !hbits.isMarked() {
			// Find the exact byte for which the special was setup
			// (as opposed to object beginning).
			p := uintptr(s.start<<_PageShift) + uintptr(special.offset)
			// about to free object: splice out special record
			y := special
			special = special.next
			*specialp = special
			if !freespecial(y, unsafe.Pointer(p), size, false) {
				// stop freeing of object if it has a finalizer
				hbits.setMarkedNonAtomic()
			}
		} else {
			// object is still live: keep special record
			specialp = &special.next
			special = *specialp
		}
	}

	// Sweep through n objects of given size starting at p.
	// This thread owns the span now, so it can manipulate
	// the block bitmap without atomic operations.

	size, n, _ := s.layout()
	heapBitsSweepSpan(s.base(), size, n, func(p uintptr) {
		// At this point we know that we are looking at garbage object
		// that needs to be collected.
		if debug.allocfreetrace != 0 {
			tracefree(unsafe.Pointer(p), size)
		}

		// Reset to allocated+noscan.
		if cl == 0 {
			// Free large span.
			if preserve {
				throw("can't preserve large span")
			}
			heapBitsForSpan(p).initSpan(s.layout())
			s.needzero = 1

			// important to set sweepgen before returning it to heap
			atomicstore(&s.sweepgen, sweepgen)
			sweepgenset = true

			// NOTE(rsc,dvyukov): The original implementation of efence
			// in CL 22060046 used SysFree instead of SysFault, so that
			// the operating system would eventually give the memory
			// back to us again, so that an efence program could run
			// longer without running out of memory. Unfortunately,
			// calling SysFree here without any kind of adjustment of the
			// heap data structures means that when the memory does
			// come back to us, we have the wrong metadata for it, either in
			// the MSpan structures or in the garbage collection bitmap.
			// Using SysFault here means that the program will run out of
			// memory fairly quickly in efence mode, but at least it won't
			// have mysterious crashes due to confused memory reuse.
			// It should be possible to switch back to SysFree if we also
			// implement and then call some kind of MHeap_DeleteSpan.
			if debug.efence > 0 {
				s.limit = 0 // prevent mlookup from finding this span
				sysFault(unsafe.Pointer(p), size)
			} else {
				mHeap_Free(&mheap_, s, 1)
			}
			c.local_nlargefree++
			c.local_largefree += size
			res = true
		} else {
			// Free small object.
			if size > 2*ptrSize {
				*(*uintptr)(unsafe.Pointer(p + ptrSize)) = uintptrMask & 0xdeaddeaddeaddead // mark as "needs to be zeroed"
			} else if size > ptrSize {
				*(*uintptr)(unsafe.Pointer(p + ptrSize)) = 0
			}
			if head.ptr() == nil {
				head = gclinkptr(p)
			} else {
				end.ptr().next = gclinkptr(p)
			}
			end = gclinkptr(p)
			end.ptr().next = gclinkptr(0x0bade5)
			nfree++
		}
	})

	// We need to set s.sweepgen = h.sweepgen only when all blocks are swept,
	// because of the potential for a concurrent free/SetFinalizer.
	// But we need to set it before we make the span available for allocation
	// (return it to heap or mcentral), because allocation code assumes that a
	// span is already swept if available for allocation.
	if !sweepgenset && nfree == 0 {
		// The span must be in our exclusive ownership until we update sweepgen,
		// check for potential races.
		if s.state != mSpanInUse || s.sweepgen != sweepgen-1 {
			print("MSpan_Sweep: state=", s.state, " sweepgen=", s.sweepgen, " mheap.sweepgen=", sweepgen, "\n")
			throw("MSpan_Sweep: bad span state after sweep")
		}
		atomicstore(&s.sweepgen, sweepgen)
	}
	if nfree > 0 {
		c.local_nsmallfree[cl] += uintptr(nfree)
		res = mCentral_FreeSpan(&mheap_.central[cl].mcentral, s, int32(nfree), head, end, preserve)
		// MCentral_FreeSpan updates sweepgen
	}
	if trace.enabled {
		traceGCSweepDone()
	}
	return res
}

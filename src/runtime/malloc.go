// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	debugMalloc = false

	flagNoScan = _FlagNoScan
	flagNoZero = _FlagNoZero

	maxTinySize   = _TinySize
	tinySizeClass = _TinySizeClass
	maxSmallSize  = _MaxSmallSize

	pageShift = _PageShift
	pageSize  = _PageSize
	pageMask  = _PageMask

	mSpanInUse = _MSpanInUse

	concurrentSweep = _ConcurrentSweep
)

// Page number (address>>pageShift)
type pageID uintptr

// base address for all 0-byte allocations
var zerobase uintptr

// Trigger the concurrent GC when 1/triggerratio memory is available to allocate.
// Adjust this ratio as part of a scheme to ensure that mutators have enough
// memory to allocate in durring a concurrent GC cycle.
var triggerratio = int64(8)

// Determine whether to initiate a GC.
// If the GC is already working no need to trigger another one.
// This should establish a feedback loop where if the GC does not
// have sufficient time to complete then more memory will be
// requested from the OS increasing heap size thus allow future
// GCs more time to complete.
// memstat.heap_alloc and memstat.next_gc reads have benign races
// A false negative simple does not start a GC, a false positive
// will start a GC needlessly. Neither have correctness issues.
func shouldtriggergc() bool {
	return triggerratio*(int64(memstats.next_gc)-int64(memstats.heap_alloc)) <= int64(memstats.next_gc) && atomicloaduint(&bggc.working) == 0
}

// Allocate an object of size bytes.
// Small objects are allocated from the per-P cache's free lists.
// Large objects (> 32 kB) are allocated straight from the heap.
func mallocgc(size uintptr, typ *_type, flags uint32) unsafe.Pointer {
	shouldhelpgc := false
	if size == 0 {
		return unsafe.Pointer(&zerobase)
	}
	dataSize := size

	if flags&flagNoScan == 0 && typ == nil {
		throw("malloc missing type")
	}

	// Set mp.mallocing to keep from being preempted by GC.
	mp := acquirem()
	if mp.mallocing != 0 {
		throw("malloc deadlock")
	}
	mp.mallocing = 1

	c := gomcache()
	var s *mspan
	var x unsafe.Pointer
	if size <= maxSmallSize {
		if flags&flagNoScan != 0 && size < maxTinySize {
			// Tiny allocator.
			//
			// Tiny allocator combines several tiny allocation requests
			// into a single memory block. The resulting memory block
			// is freed when all subobjects are unreachable. The subobjects
			// must be FlagNoScan (don't have pointers), this ensures that
			// the amount of potentially wasted memory is bounded.
			//
			// Size of the memory block used for combining (maxTinySize) is tunable.
			// Current setting is 16 bytes, which relates to 2x worst case memory
			// wastage (when all but one subobjects are unreachable).
			// 8 bytes would result in no wastage at all, but provides less
			// opportunities for combining.
			// 32 bytes provides more opportunities for combining,
			// but can lead to 4x worst case wastage.
			// The best case winning is 8x regardless of block size.
			//
			// Objects obtained from tiny allocator must not be freed explicitly.
			// So when an object will be freed explicitly, we ensure that
			// its size >= maxTinySize.
			//
			// SetFinalizer has a special case for objects potentially coming
			// from tiny allocator, it such case it allows to set finalizers
			// for an inner byte of a memory block.
			//
			// The main targets of tiny allocator are small strings and
			// standalone escaping variables. On a json benchmark
			// the allocator reduces number of allocations by ~12% and
			// reduces heap size by ~20%.
			off := c.tinyoffset
			// Align tiny pointer for required (conservative) alignment.
			if size&7 == 0 {
				off = round(off, 8)
			} else if size&3 == 0 {
				off = round(off, 4)
			} else if size&1 == 0 {
				off = round(off, 2)
			}
			if off+size <= maxTinySize && c.tiny != nil {
				// The object fits into existing tiny block.
				x = add(c.tiny, off)
				c.tinyoffset = off + size
				c.local_tinyallocs++
				mp.mallocing = 0
				releasem(mp)
				return x
			}
			// Allocate a new maxTinySize block.
			s = c.alloc[tinySizeClass]
			v := s.freelist
			if v.ptr() == nil {
				systemstack(func() {
					mCache_Refill(c, tinySizeClass)
				})
				shouldhelpgc = true
				s = c.alloc[tinySizeClass]
				v = s.freelist
			}
			s.freelist = v.ptr().next
			s.ref++
			//TODO: prefetch v.next
			x = unsafe.Pointer(v)
			(*[2]uint64)(x)[0] = 0
			(*[2]uint64)(x)[1] = 0
			// See if we need to replace the existing tiny block with the new one
			// based on amount of remaining free space.
			if size < c.tinyoffset {
				c.tiny = x
				c.tinyoffset = size
			}
			size = maxTinySize
		} else {
			var sizeclass int8
			if size <= 1024-8 {
				sizeclass = size_to_class8[(size+7)>>3]
			} else {
				sizeclass = size_to_class128[(size-1024+127)>>7]
			}
			size = uintptr(class_to_size[sizeclass])
			s = c.alloc[sizeclass]
			v := s.freelist
			if v.ptr() == nil {
				systemstack(func() {
					mCache_Refill(c, int32(sizeclass))
				})
				shouldhelpgc = true
				s = c.alloc[sizeclass]
				v = s.freelist
			}
			s.freelist = v.ptr().next
			s.ref++
			//TODO: prefetch
			x = unsafe.Pointer(v)
			if flags&flagNoZero == 0 {
				v.ptr().next = 0
				if size > 2*ptrSize && ((*[2]uintptr)(x))[1] != 0 {
					memclr(unsafe.Pointer(v), size)
				}
			}
		}
		c.local_cachealloc += intptr(size)
	} else {
		var s *mspan
		shouldhelpgc = true
		systemstack(func() {
			s = largeAlloc(size, uint32(flags))
		})
		x = unsafe.Pointer(uintptr(s.start << pageShift))
		size = uintptr(s.elemsize)
	}

	if flags&flagNoScan != 0 {
		// All objects are pre-marked as noscan. Nothing to do.
	} else {
		// If allocating a defer+arg block, now that we've picked a malloc size
		// large enough to hold everything, cut the "asked for" size down to
		// just the defer header, so that the GC bitmap will record the arg block
		// as containing nothing at all (as if it were unused space at the end of
		// a malloc block caused by size rounding).
		// The defer arg areas are scanned as part of scanstack.
		if typ == deferType {
			dataSize = unsafe.Sizeof(_defer{})
		}
		heapBitsSetType(uintptr(x), size, dataSize, typ)
	}

	// GCmarkterminate allocates black
	// All slots hold nil so no scanning is needed.
	// This may be racing with GC so do it atomically if there can be
	// a race marking the bit.
	if gcphase == _GCmarktermination {
		systemstack(func() {
			gcmarknewobject_m(uintptr(x))
		})
	}

	if mheap_.shadow_enabled {
		clearshadow(uintptr(x), size)
	}

	if raceenabled {
		racemalloc(x, size)
	}

	mp.mallocing = 0
	releasem(mp)

	if debug.allocfreetrace != 0 {
		tracealloc(x, size, typ)
	}

	if rate := MemProfileRate; rate > 0 {
		if size < uintptr(rate) && int32(size) < c.next_sample {
			c.next_sample -= int32(size)
		} else {
			mp := acquirem()
			profilealloc(mp, x, size)
			releasem(mp)
		}
	}

	if shouldtriggergc() {
		gogc(0)
	} else if shouldhelpgc && atomicloaduint(&bggc.working) == 1 {
		// bggc.lock not taken since race on bggc.working is benign.
		// At worse we don't call gchelpwork.
		// Delay the gchelpwork until the epilogue so that it doesn't
		// interfere with the inner working of malloc such as
		// mcache refills that might happen while doing the gchelpwork
		systemstack(gchelpwork)
	}

	return x
}

// implementation of new builtin
func newobject(typ *_type) unsafe.Pointer {
	flags := uint32(0)
	if typ.kind&kindNoPointers != 0 {
		flags |= flagNoScan
	}
	return mallocgc(uintptr(typ.size), typ, flags)
}

//go:linkname reflect_unsafe_New reflect.unsafe_New
func reflect_unsafe_New(typ *_type) unsafe.Pointer {
	return newobject(typ)
}

// implementation of make builtin for slices
func newarray(typ *_type, n uintptr) unsafe.Pointer {
	flags := uint32(0)
	if typ.kind&kindNoPointers != 0 {
		flags |= flagNoScan
	}
	if int(n) < 0 || (typ.size > 0 && n > _MaxMem/uintptr(typ.size)) {
		panic("runtime: allocation size out of range")
	}
	return mallocgc(uintptr(typ.size)*n, typ, flags)
}

//go:linkname reflect_unsafe_NewArray reflect.unsafe_NewArray
func reflect_unsafe_NewArray(typ *_type, n uintptr) unsafe.Pointer {
	return newarray(typ, n)
}

// rawmem returns a chunk of pointerless memory.  It is
// not zeroed.
func rawmem(size uintptr) unsafe.Pointer {
	return mallocgc(size, nil, flagNoScan|flagNoZero)
}

func profilealloc(mp *m, x unsafe.Pointer, size uintptr) {
	c := mp.mcache
	rate := MemProfileRate
	if size < uintptr(rate) {
		// pick next profile time
		// If you change this, also change allocmcache.
		if rate > 0x3fffffff { // make 2*rate not overflow
			rate = 0x3fffffff
		}
		next := int32(fastrand1()) % (2 * int32(rate))
		// Subtract the "remainder" of the current allocation.
		// Otherwise objects that are close in size to sampling rate
		// will be under-sampled, because we consistently discard this remainder.
		next -= (int32(size) - c.next_sample)
		if next < 0 {
			next = 0
		}
		c.next_sample = next
	}

	mProf_Malloc(x, size)
}

// For now this must be bracketed with a stoptheworld and a starttheworld to ensure
// all go routines see the new barrier.
func gcinstallmarkwb() {
	gcphase = _GCmark
}

// force = 0 - start concurrent GC
// force = 1 - do STW GC regardless of current heap usage
// force = 2 - go STW GC and eager sweep
func gogc(force int32) {
	// The gc is turned off (via enablegc) until the bootstrap has completed.
	// Also, malloc gets called in the guts of a number of libraries that might be
	// holding locks. To avoid deadlocks during stoptheworld, don't bother
	// trying to run gc while holding a lock. The next mallocgc without a lock
	// will do the gc instead.

	mp := acquirem()
	if gp := getg(); gp == mp.g0 || mp.locks > 1 || !memstats.enablegc || panicking != 0 || gcpercent < 0 {
		releasem(mp)
		return
	}
	releasem(mp)
	mp = nil

	if force == 0 {
		lock(&bggc.lock)
		if !bggc.started {
			bggc.working = 1
			bggc.started = true
			go backgroundgc()
		} else if bggc.working == 0 {
			bggc.working = 1
			ready(bggc.g)
		}
		unlock(&bggc.lock)
	} else {
		gcwork(force)
	}
}

func gcwork(force int32) {

	semacquire(&worldsema, false)

	// Pick up the remaining unswept/not being swept spans concurrently
	for gosweepone() != ^uintptr(0) {
		sweep.nbgsweep++
	}

	// Ok, we're doing it!  Stop everybody else

	mp := acquirem()
	mp.gcing = 1
	releasem(mp)
	gctimer.count++
	if force == 0 {
		gctimer.cycle.sweepterm = nanotime()
	}
	// Pick up the remaining unswept/not being swept spans before we STW
	for gosweepone() != ^uintptr(0) {
		sweep.nbgsweep++
	}
	systemstack(stoptheworld)
	systemstack(finishsweep_m) // finish sweep before we start concurrent scan.
	if force == 0 {            // Do as much work concurrently as possible
		gcphase = _GCscan
		systemstack(starttheworld)
		gctimer.cycle.scan = nanotime()
		// Do a concurrent heap scan before we stop the world.
		systemstack(gcscan_m)
		gctimer.cycle.installmarkwb = nanotime()
		systemstack(stoptheworld)
		systemstack(gcinstallmarkwb)
		systemstack(starttheworld)
		gctimer.cycle.mark = nanotime()
		systemstack(gcmark_m)
		gctimer.cycle.markterm = nanotime()
		systemstack(stoptheworld)
		systemstack(gcinstalloffwb_m)
	}

	startTime := nanotime()
	if mp != acquirem() {
		throw("gogc: rescheduled")
	}

	clearpools()

	// Run gc on the g0 stack.  We do this so that the g stack
	// we're currently running on will no longer change.  Cuts
	// the root set down a bit (g0 stacks are not scanned, and
	// we don't need to scan gc's internal state).  We also
	// need to switch to g0 so we can shrink the stack.
	n := 1
	if debug.gctrace > 1 {
		n = 2
	}
	eagersweep := force >= 2
	for i := 0; i < n; i++ {
		if i > 0 {
			// refresh start time if doing a second GC
			startTime = nanotime()
		}
		// switch to g0, call gc, then switch back
		systemstack(func() {
			gc_m(startTime, eagersweep)
		})
	}

	systemstack(func() {
		gccheckmark_m(startTime, eagersweep)
	})

	// all done
	mp.gcing = 0

	if force == 0 {
		gctimer.cycle.sweep = nanotime()
	}

	semrelease(&worldsema)

	if force == 0 {
		if gctimer.verbose > 1 {
			GCprinttimes()
		} else if gctimer.verbose > 0 {
			calctimes() // ignore result
		}
	}

	systemstack(starttheworld)

	releasem(mp)
	mp = nil

	// now that gc is done, kick off finalizer thread if needed
	if !concurrentSweep {
		// give the queued finalizers, if any, a chance to run
		Gosched()
	}
}

// gctimes records the time in nanoseconds of each phase of the concurrent GC.
type gctimes struct {
	sweepterm     int64 // stw
	scan          int64
	installmarkwb int64 // stw
	mark          int64
	markterm      int64 // stw
	sweep         int64
}

// gcchronograph holds timer information related to GC phases
// max records the maximum time spent in each GC phase since GCstarttimes.
// total records the total time spent in each GC phase since GCstarttimes.
// cycle records the absolute time (as returned by nanoseconds()) that each GC phase last started at.
type gcchronograph struct {
	count    int64
	verbose  int64
	maxpause int64
	max      gctimes
	total    gctimes
	cycle    gctimes
}

var gctimer gcchronograph

// GCstarttimes initializes the gc times. All previous times are lost.
func GCstarttimes(verbose int64) {
	gctimer = gcchronograph{verbose: verbose}
}

// GCendtimes stops the gc timers.
func GCendtimes() {
	gctimer.verbose = 0
}

// calctimes converts gctimer.cycle into the elapsed times, updates gctimer.total
// and updates gctimer.max with the max pause time.
func calctimes() gctimes {
	var times gctimes

	var max = func(a, b int64) int64 {
		if a > b {
			return a
		}
		return b
	}

	times.sweepterm = gctimer.cycle.scan - gctimer.cycle.sweepterm
	gctimer.total.sweepterm += times.sweepterm
	gctimer.max.sweepterm = max(gctimer.max.sweepterm, times.sweepterm)
	gctimer.maxpause = max(gctimer.maxpause, gctimer.max.sweepterm)

	times.scan = gctimer.cycle.installmarkwb - gctimer.cycle.scan
	gctimer.total.scan += times.scan
	gctimer.max.scan = max(gctimer.max.scan, times.scan)

	times.installmarkwb = gctimer.cycle.mark - gctimer.cycle.installmarkwb
	gctimer.total.installmarkwb += times.installmarkwb
	gctimer.max.installmarkwb = max(gctimer.max.installmarkwb, times.installmarkwb)
	gctimer.maxpause = max(gctimer.maxpause, gctimer.max.installmarkwb)

	times.mark = gctimer.cycle.markterm - gctimer.cycle.mark
	gctimer.total.mark += times.mark
	gctimer.max.mark = max(gctimer.max.mark, times.mark)

	times.markterm = gctimer.cycle.sweep - gctimer.cycle.markterm
	gctimer.total.markterm += times.markterm
	gctimer.max.markterm = max(gctimer.max.markterm, times.markterm)
	gctimer.maxpause = max(gctimer.maxpause, gctimer.max.markterm)

	return times
}

// GCprinttimes prints latency information in nanoseconds about various
// phases in the GC. The information for each phase includes the maximum pause
// and total time since the most recent call to GCstarttimes as well as
// the information from the most recent Concurent GC cycle. Calls from the
// application to runtime.GC() are ignored.
func GCprinttimes() {
	if gctimer.verbose == 0 {
		println("GC timers not enabled")
		return
	}

	// Explicitly put times on the heap so printPhase can use it.
	times := new(gctimes)
	*times = calctimes()
	cycletime := gctimer.cycle.sweep - gctimer.cycle.sweepterm
	pause := times.sweepterm + times.installmarkwb + times.markterm
	gomaxprocs := GOMAXPROCS(-1)

	printlock()
	print("GC: #", gctimer.count, " ", cycletime, "ns @", gctimer.cycle.sweepterm, " pause=", pause, " maxpause=", gctimer.maxpause, " goroutines=", allglen, " gomaxprocs=", gomaxprocs, "\n")
	printPhase := func(label string, get func(*gctimes) int64, procs int) {
		print("GC:     ", label, " ", get(times), "ns\tmax=", get(&gctimer.max), "\ttotal=", get(&gctimer.total), "\tprocs=", procs, "\n")
	}
	printPhase("sweep term:", func(t *gctimes) int64 { return t.sweepterm }, gomaxprocs)
	printPhase("scan:      ", func(t *gctimes) int64 { return t.scan }, 1)
	printPhase("install wb:", func(t *gctimes) int64 { return t.installmarkwb }, gomaxprocs)
	printPhase("mark:      ", func(t *gctimes) int64 { return t.mark }, 1)
	printPhase("mark term: ", func(t *gctimes) int64 { return t.markterm }, gomaxprocs)
	printunlock()
}

// GC runs a garbage collection.
func GC() {
	gogc(2)
}

// linker-provided
var noptrdata struct{}
var enoptrdata struct{}
var noptrbss struct{}
var enoptrbss struct{}

// round n up to a multiple of a.  a must be a power of 2.
func round(n, a uintptr) uintptr {
	return (n + a - 1) &^ (a - 1)
}

var persistent struct {
	lock mutex
	base unsafe.Pointer
	off  uintptr
}

// Wrapper around sysAlloc that can allocate small chunks.
// There is no associated free operation.
// Intended for things like function/type/debug-related persistent data.
// If align is 0, uses default align (currently 8).
func persistentalloc(size, align uintptr, stat *uint64) unsafe.Pointer {
	const (
		chunk    = 256 << 10
		maxBlock = 64 << 10 // VM reservation granularity is 64K on windows
	)

	if size == 0 {
		throw("persistentalloc: size == 0")
	}
	if align != 0 {
		if align&(align-1) != 0 {
			throw("persistentalloc: align is not a power of 2")
		}
		if align > _PageSize {
			throw("persistentalloc: align is too large")
		}
	} else {
		align = 8
	}

	if size >= maxBlock {
		return sysAlloc(size, stat)
	}

	lock(&persistent.lock)
	persistent.off = round(persistent.off, align)
	if persistent.off+size > chunk || persistent.base == nil {
		persistent.base = sysAlloc(chunk, &memstats.other_sys)
		if persistent.base == nil {
			unlock(&persistent.lock)
			throw("runtime: cannot allocate memory")
		}
		persistent.off = 0
	}
	p := add(persistent.base, persistent.off)
	persistent.off += size
	unlock(&persistent.lock)

	if stat != &memstats.other_sys {
		xadd64(stat, int64(size))
		xadd64(&memstats.other_sys, -int64(size))
	}
	return p
}

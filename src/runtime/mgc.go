// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): The code having to do with the heap bitmap needs very serious cleanup.
// It has gotten completely out of control.

// Garbage collector (GC).
//
// The GC runs concurrently with mutator threads, is type accurate (aka precise), allows multiple
// GC thread to run in parallel. It is a concurrent mark and sweep that uses a write barrier. It is
// non-generational and non-compacting. Allocation is done using size segregated per P allocation
// areas to minimize fragmentation while eliminating locks in the common case.
//
// The algorithm decomposes into several steps.
// This is a high level description of the algorithm being used. For an overview of GC a good
// place to start is Richard Jones' gchandbook.org.
//
// The algorithm's intellectual heritage includes Dijkstra's on-the-fly algorithm, see
// Edsger W. Dijkstra, Leslie Lamport, A. J. Martin, C. S. Scholten, and E. F. M. Steffens. 1978.
// On-the-fly garbage collection: an exercise in cooperation. Commun. ACM 21, 11 (November 1978),
// 966-975.
// For journal quality proofs that these steps are complete, correct, and terminate see
// Hudson, R., and Moss, J.E.B. Copying Garbage Collection without stopping the world.
// Concurrency and Computation: Practice and Experience 15(3-5), 2003.
//
//  0. Set phase = GCscan from GCoff.
//  1. Wait for all P's to acknowledge phase change.
//         At this point all goroutines have passed through a GC safepoint and
//         know we are in the GCscan phase.
//  2. GC scans all goroutine stacks, mark and enqueues all encountered pointers
//       (marking avoids most duplicate enqueuing but races may produce benign duplication).
//       Preempted goroutines are scanned before P schedules next goroutine.
//  3. Set phase = GCmark.
//  4. Wait for all P's to acknowledge phase change.
//  5. Now write barrier marks and enqueues black, grey, or white to white pointers.
//       Malloc still allocates white (non-marked) objects.
//  6. Meanwhile GC transitively walks the heap marking reachable objects.
//  7. When GC finishes marking heap, it preempts P's one-by-one and
//       retakes partial wbufs (filled by write barrier or during a stack scan of the goroutine
//       currently scheduled on the P).
//  8. Once the GC has exhausted all available marking work it sets phase = marktermination.
//  9. Wait for all P's to acknowledge phase change.
// 10. Malloc now allocates black objects, so number of unmarked reachable objects
//        monotonically decreases.
// 11. GC preempts P's one-by-one taking partial wbufs and marks all unmarked yet
//        reachable objects.
// 12. When GC completes a full cycle over P's and discovers no new grey
//         objects, (which means all reachable objects are marked) set phase = GCsweep.
// 13. Wait for all P's to acknowledge phase change.
// 14. Now malloc allocates white (but sweeps spans before use).
//         Write barrier becomes nop.
// 15. GC does background sweeping, see description below.
// 16. When sweeping is complete set phase to GCoff.
// 17. When sufficient allocation has taken place replay the sequence starting at 0 above,
//         see discussion of GC rate below.

// Changing phases.
// Phases are changed by setting the gcphase to the next phase and possibly calling ackgcphase.
// All phase action must be benign in the presence of a change.
// Starting with GCoff
// GCoff to GCscan
//     GSscan scans stacks and globals greying them and never marks an object black.
//     Once all the P's are aware of the new phase they will scan gs on preemption.
//     This means that the scanning of preempted gs can't start until all the Ps
//     have acknowledged.
// GCscan to GCmark
//     GCMark turns on the write barrier which also only greys objects. No scanning
//     of objects (making them black) can happen until all the Ps have acknowledged
//     the phase change.
// GCmark to GCmarktermination
//     The only change here is that we start allocating black so the Ps must acknowledge
//     the change before we begin the termination algorithm
// GCmarktermination to GSsweep
//     Object currently on the freelist must be marked black for this to work.
//     Are things on the free lists black or white? How does the sweep phase work?

// Concurrent sweep.
// The sweep phase proceeds concurrently with normal program execution.
// The heap is swept span-by-span both lazily (when a goroutine needs another span)
// and concurrently in a background goroutine (this helps programs that are not CPU bound).
// However, at the end of the stop-the-world GC phase we don't know the size of the live heap,
// and so next_gc calculation is tricky and happens as follows.
// At the end of the stop-the-world phase next_gc is conservatively set based on total
// heap size; all spans are marked as "needs sweeping".
// Whenever a span is swept, next_gc is decremented by GOGC*newly_freed_memory.
// The background sweeper goroutine simply sweeps spans one-by-one bringing next_gc
// closer to the target value. However, this is not enough to avoid over-allocating memory.
// Consider that a goroutine wants to allocate a new span for a large object and
// there are no free swept spans, but there are small-object unswept spans.
// If the goroutine naively allocates a new span, it can surpass the yet-unknown
// target next_gc value. In order to prevent such cases (1) when a goroutine needs
// to allocate a new small-object span, it sweeps small-object spans for the same
// object size until it frees at least one object; (2) when a goroutine needs to
// allocate large-object span from heap, it sweeps spans until it frees at least
// that many pages into heap. Together these two measures ensure that we don't surpass
// target next_gc value by a large margin. There is an exception: if a goroutine sweeps
// and frees two nonadjacent one-page spans to the heap, it will allocate a new two-page span,
// but there can still be other one-page unswept spans which could be combined into a
// two-page span.
// It's critical to ensure that no operations proceed on unswept spans (that would corrupt
// mark bits in GC bitmap). During GC all mcaches are flushed into the central cache,
// so they are empty. When a goroutine grabs a new span into mcache, it sweeps it.
// When a goroutine explicitly frees an object or sets a finalizer, it ensures that
// the span is swept (either by sweeping it, or by waiting for the concurrent sweep to finish).
// The finalizer goroutine is kicked off only when all spans are swept.
// When the next GC starts, it sweeps all not-yet-swept spans (if any).

// GC rate.
// Next GC is after we've allocated an extra amount of memory proportional to
// the amount already in use. The proportion is controlled by GOGC environment variable
// (100 by default). If GOGC=100 and we're using 4M, we'll GC again when we get to 8M
// (this mark is tracked in next_gc variable). This keeps the GC cost in linear
// proportion to the allocation cost. Adjusting GOGC just changes the linear constant
// (and also the amount of extra memory used).

package runtime

import "unsafe"

const (
	_DebugGC         = 0
	_ConcurrentSweep = true
	_FinBlockSize    = 4 * 1024
	_RootData        = 0
	_RootBss         = 1
	_RootFinalizers  = 2
	_RootSpans       = 3
	_RootFlushCaches = 4
	_RootCount       = 5
)

// linker-provided
var data, edata, bss, ebss, gcdata, gcbss, noptrdata, enoptrdata, noptrbss, enoptrbss, end struct{}

//go:linkname weak_cgo_allocate go.weak.runtime._cgo_allocate_internal
var weak_cgo_allocate byte

// Is _cgo_allocate linked into the binary?
//go:nowritebarrier
func have_cgo_allocate() bool {
	return &weak_cgo_allocate != nil
}

// Slow for now as we serialize this, since this is on a debug path
// speed is not critical at this point.
var andlock mutex

//go:nowritebarrier
func atomicand8(src *byte, val byte) {
	lock(&andlock)
	*src &= val
	unlock(&andlock)
}

var gcdatamask bitvector
var gcbssmask bitvector

// heapminimum is the minimum number of bytes in the heap.
// This cleans up the corner case of where we have a very small live set but a lot
// of allocations and collecting every GOGC * live set is expensive.
var heapminimum = uint64(4 << 20)

// Initialized from $GOGC.  GOGC=off means no GC.
var gcpercent int32

func gcinit() {
	if unsafe.Sizeof(workbuf{}) != _WorkbufSize {
		throw("size of Workbuf is suboptimal")
	}

	work.markfor = parforalloc(_MaxGcproc)
	gcpercent = readgogc()
	gcdatamask = unrollglobgcprog((*byte)(unsafe.Pointer(&gcdata)), uintptr(unsafe.Pointer(&edata))-uintptr(unsafe.Pointer(&data)))
	gcbssmask = unrollglobgcprog((*byte)(unsafe.Pointer(&gcbss)), uintptr(unsafe.Pointer(&ebss))-uintptr(unsafe.Pointer(&bss)))
	memstats.next_gc = heapminimum
}

func setGCPercent(in int32) (out int32) {
	lock(&mheap_.lock)
	out = gcpercent
	if in < 0 {
		in = -1
	}
	gcpercent = in
	unlock(&mheap_.lock)
	return out
}

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

var work struct {
	full    uint64                // lock-free list of full blocks workbuf
	empty   uint64                // lock-free list of empty blocks workbuf
	partial uint64                // lock-free list of partially filled blocks workbuf
	pad0    [_CacheLineSize]uint8 // prevents false-sharing between full/empty and nproc/nwait
	nproc   uint32
	tstart  int64
	nwait   uint32
	ndone   uint32
	alldone note
	markfor *parfor

	// Copy of mheap.allspans for marker or sweeper.
	spans []*mspan
}

// GC runs a garbage collection.
func GC() {
	startGC(gcForceBlockMode)
}

const (
	gcBackgroundMode = iota // concurrent GC
	gcForceMode             // stop-the-world GC now
	gcForceBlockMode        // stop-the-world GC now and wait for sweep
)

func startGC(mode int) {
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

	if mode != gcBackgroundMode {
		// special synchronous cases
		gc(mode)
		return
	}

	// trigger concurrent GC
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
}

// State of the background concurrent GC goroutine.
var bggc struct {
	lock    mutex
	g       *g
	working uint
	started bool
}

// backgroundgc is running in a goroutine and does the concurrent GC work.
// bggc holds the state of the backgroundgc.
func backgroundgc() {
	bggc.g = getg()
	for {
		gc(gcBackgroundMode)
		lock(&bggc.lock)
		bggc.working = 0
		goparkunlock(&bggc.lock, "Concurrent GC wait", traceEvGoBlock)
	}
}

func gc(mode int) {
	// Ok, we're doing it!  Stop everybody else
	semacquire(&worldsema, false)

	// Pick up the remaining unswept/not being swept spans concurrently
	for gosweepone() != ^uintptr(0) {
		sweep.nbgsweep++
	}

	mp := acquirem()
	mp.preemptoff = "gcing"
	releasem(mp)
	gctimer.count++
	if mode == gcBackgroundMode {
		gctimer.cycle.sweepterm = nanotime()
	}

	if trace.enabled {
		traceGoSched()
		traceGCStart()
	}

	systemstack(stoptheworld)
	systemstack(finishsweep_m) // finish sweep before we start concurrent scan.

	if mode == gcBackgroundMode { // Do as much work concurrently as possible
		systemstack(func() {
			gcphase = _GCscan

			// Concurrent scan.
			starttheworld()
			gctimer.cycle.scan = nanotime()
			gcscan_m()
			gctimer.cycle.installmarkwb = nanotime()

			// Sync.
			stoptheworld()
			gcphase = _GCmark
			harvestwbufs()

			// Concurrent mark.
			starttheworld()
			gctimer.cycle.mark = nanotime()
			var gcw gcWork
			gcDrain(&gcw)
			gcw.dispose()

			// Begin mark termination.
			gctimer.cycle.markterm = nanotime()
			stoptheworld()
			gcphase = _GCoff
		})
	} else {
		// For non-concurrent GC (mode != gcBackgroundMode)
		// g stack have not been scanned so set gcscanvalid
		// such that mark termination scans all stacks.
		// No races here since we are in a STW phase.
		for _, gp := range allgs {
			gp.gcworkdone = false  // set to true in gcphasework
			gp.gcscanvalid = false // stack has not been scanned
		}
	}

	startTime := nanotime()
	if mp != acquirem() {
		throw("gcwork: rescheduled")
	}

	// TODO(rsc): Should the concurrent GC clear pools earlier?
	clearpools()

	_g_ := getg()
	_g_.m.traceback = 2
	gp := _g_.m.curg
	casgstatus(gp, _Grunning, _Gwaiting)
	gp.waitreason = "garbage collection"

	// Run gc on the g0 stack.  We do this so that the g stack
	// we're currently running on will no longer change.  Cuts
	// the root set down a bit (g0 stacks are not scanned, and
	// we don't need to scan gc's internal state).  We also
	// need to switch to g0 so we can shrink the stack.
	systemstack(func() {
		gcMark(startTime)
		if debug.gccheckmark > 0 {
			// Run a full stop-the-world mark using checkmark bits,
			// to check that we didn't forget to mark anything during
			// the concurrent mark process.
			initCheckmarks()
			gcMark(startTime)
			clearCheckmarks()
		}
		gcSweep(mode)

		if debug.gctrace > 1 {
			startTime = nanotime()
			finishsweep_m()
			gcMark(startTime)
			gcSweep(mode)
		}
	})

	_g_.m.traceback = 0
	casgstatus(gp, _Gwaiting, _Grunning)

	if trace.enabled {
		traceGCDone()
		traceGoStart()
	}

	// all done
	mp.preemptoff = ""

	if mode == gcBackgroundMode {
		gctimer.cycle.sweep = nanotime()
	}

	semrelease(&worldsema)

	if mode == gcBackgroundMode {
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

// gcMark runs the mark (or, for concurrent GC, mark termination)
// STW is in effect at this point.
//TODO go:nowritebarrier
func gcMark(start_time int64) {
	if debug.allocfreetrace > 0 {
		tracegc()
	}

	t0 := start_time
	work.tstart = start_time
	gcphase = _GCmarktermination

	var t1 int64
	if debug.gctrace > 0 {
		t1 = nanotime()
	}

	gcCopySpans()

	work.nwait = 0
	work.ndone = 0
	work.nproc = uint32(gcprocs())

	// World is stopped so allglen will not change.
	for i := uintptr(0); i < allglen; i++ {
		gp := allgs[i]
		gp.gcworkdone = false // set to true in gcphasework
	}

	if trace.enabled {
		traceGCScanStart()
	}

	parforsetup(work.markfor, work.nproc, uint32(_RootCount+allglen), false, markroot)
	if work.nproc > 1 {
		noteclear(&work.alldone)
		helpgc(int32(work.nproc))
	}

	var t2 int64
	if debug.gctrace > 0 {
		t2 = nanotime()
	}

	harvestwbufs() // move local workbufs onto global queues where the GC can find them
	gchelperstart()
	parfordo(work.markfor)
	var gcw gcWork
	gcDrain(&gcw)
	gcw.dispose()

	if work.full != 0 {
		throw("work.full != 0")
	}
	if work.partial != 0 {
		throw("work.partial != 0")
	}

	gcphase = _GCoff
	var t3 int64
	if debug.gctrace > 0 {
		t3 = nanotime()
	}

	if work.nproc > 1 {
		notesleep(&work.alldone)
	}

	if trace.enabled {
		traceGCScanDone()
	}

	shrinkfinish()

	cachestats()
	// next_gc calculation is tricky with concurrent sweep since we don't know size of live heap
	// estimate what was live heap size after previous GC (for printing only)
	heap0 := memstats.next_gc * 100 / (uint64(gcpercent) + 100)
	// conservatively set next_gc to high value assuming that everything is live
	// concurrent/lazy sweep will reduce this number while discovering new garbage
	memstats.next_gc = memstats.heap_alloc + memstats.heap_alloc*uint64(gcpercent)/100
	if memstats.next_gc < heapminimum {
		memstats.next_gc = heapminimum
	}

	if trace.enabled {
		traceNextGC()
	}

	t4 := nanotime()
	atomicstore64(&memstats.last_gc, uint64(unixnanotime())) // must be Unix time to make sense to user
	memstats.pause_ns[memstats.numgc%uint32(len(memstats.pause_ns))] = uint64(t4 - t0)
	memstats.pause_end[memstats.numgc%uint32(len(memstats.pause_end))] = uint64(t4)
	memstats.pause_total_ns += uint64(t4 - t0)
	memstats.numgc++
	if memstats.debuggc {
		print("pause ", t4-t0, "\n")
	}

	if debug.gctrace > 0 {
		heap1 := memstats.heap_alloc
		var stats gcstats
		updatememstats(&stats)
		if heap1 != memstats.heap_alloc {
			print("runtime: mstats skew: heap=", heap1, "/", memstats.heap_alloc, "\n")
			throw("mstats skew")
		}
		obj := memstats.nmalloc - memstats.nfree

		stats.nprocyield += work.markfor.nprocyield
		stats.nosyield += work.markfor.nosyield
		stats.nsleep += work.markfor.nsleep

		print("gc", memstats.numgc, "(", work.nproc, "): ",
			(t1-t0)/1000, "+", (t2-t1)/1000, "+", (t3-t2)/1000, "+", (t4-t3)/1000, " us, ",
			heap0>>20, " -> ", heap1>>20, " MB, ",
			obj, " (", memstats.nmalloc, "-", memstats.nfree, ") objects, ",
			gcount(), " goroutines, ",
			len(work.spans), "/", sweep.nbgsweep, "/", sweep.npausesweep, " sweeps, ",
			stats.nhandoff, "(", stats.nhandoffcnt, ") handoff, ",
			work.markfor.nsteal, "(", work.markfor.nstealcnt, ") steal, ",
			stats.nprocyield, "/", stats.nosyield, "/", stats.nsleep, " yields\n")
		sweep.nbgsweep = 0
		sweep.npausesweep = 0
	}
}

func gcSweep(mode int) {
	gcCopySpans()

	lock(&mheap_.lock)
	mheap_.sweepgen += 2
	mheap_.sweepdone = 0
	sweep.spanidx = 0
	unlock(&mheap_.lock)

	if !_ConcurrentSweep || mode == gcForceBlockMode {
		// Special case synchronous sweep.
		// Sweep all spans eagerly.
		for sweepone() != ^uintptr(0) {
			sweep.npausesweep++
		}
		// Do an additional mProf_GC, because all 'free' events are now real as well.
		mProf_GC()
		mProf_GC()
		return
	}

	// Background sweep.
	lock(&sweep.lock)
	if !sweep.started {
		go bgsweep()
		sweep.started = true
	} else if sweep.parked {
		sweep.parked = false
		ready(sweep.g)
	}
	unlock(&sweep.lock)
	mProf_GC()
}

func gcCopySpans() {
	// Cache runtime.mheap_.allspans in work.spans to avoid conflicts with
	// resizing/freeing allspans.
	// New spans can be created while GC progresses, but they are not garbage for
	// this round:
	//  - new stack spans can be created even while the world is stopped.
	//  - new malloc spans can be created during the concurrent sweep
	// Even if this is stop-the-world, a concurrent exitsyscall can allocate a stack from heap.
	lock(&mheap_.lock)
	// Free the old cached mark array if necessary.
	if work.spans != nil && &work.spans[0] != &h_allspans[0] {
		sysFree(unsafe.Pointer(&work.spans[0]), uintptr(len(work.spans))*unsafe.Sizeof(work.spans[0]), &memstats.other_sys)
	}
	// Cache the current array for sweeping.
	mheap_.gcspans = mheap_.allspans
	work.spans = h_allspans
	unlock(&mheap_.lock)
}

// Hooks for other packages

var poolcleanup func()

//go:linkname sync_runtime_registerPoolCleanup sync.runtime_registerPoolCleanup
func sync_runtime_registerPoolCleanup(f func()) {
	poolcleanup = f
}

func clearpools() {
	// clear sync.Pools
	if poolcleanup != nil {
		poolcleanup()
	}

	for _, p := range &allp {
		if p == nil {
			break
		}
		// clear tinyalloc pool
		if c := p.mcache; c != nil {
			c.tiny = nil
			c.tinyoffset = 0

			// disconnect cached list before dropping it on the floor,
			// so that a dangling ref to one entry does not pin all of them.
			var sg, sgnext *sudog
			for sg = c.sudogcache; sg != nil; sg = sgnext {
				sgnext = sg.next
				sg.next = nil
			}
			c.sudogcache = nil
		}

		// clear defer pools
		for i := range p.deferpool {
			// disconnect cached list before dropping it on the floor,
			// so that a dangling ref to one entry does not pin all of them.
			var d, dlink *_defer
			for d = p.deferpool[i]; d != nil; d = dlink {
				dlink = d.link
				d.link = nil
			}
			p.deferpool[i] = nil
		}
	}
}

// Timing

//go:nowritebarrier
func gchelper() {
	_g_ := getg()
	_g_.m.traceback = 2
	gchelperstart()

	if trace.enabled {
		traceGCScanStart()
	}

	// parallel mark for over GC roots
	parfordo(work.markfor)
	if gcphase != _GCscan {
		var gcw gcWork
		gcDrain(&gcw) // blocks in getfull
		gcw.dispose()
	}

	if trace.enabled {
		traceGCScanDone()
	}

	nproc := work.nproc // work.nproc can change right after we increment work.ndone
	if xadd(&work.ndone, +1) == nproc-1 {
		notewakeup(&work.alldone)
	}
	_g_.m.traceback = 0
}

func gchelperstart() {
	_g_ := getg()

	if _g_.m.helpgc < 0 || _g_.m.helpgc >= _MaxGcproc {
		throw("gchelperstart: bad m->helpgc")
	}
	if _g_ != _g_.m.g0 {
		throw("gchelper not running on g0 stack")
	}
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

// gctimes records the time in nanoseconds of each phase of the concurrent GC.
type gctimes struct {
	sweepterm     int64 // stw
	scan          int64
	installmarkwb int64 // stw
	mark          int64
	markterm      int64 // stw
	sweep         int64
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

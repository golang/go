// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
// 1. GC performs sweep termination.
//
//    a. Stop the world. This causes all Ps to reach a GC safe-point.
//
//    b. Sweep any unswept spans. There will only be unswept spans if
//    this GC cycle was forced before the expected time.
//
// 2. GC performs the "mark 1" sub-phase. In this sub-phase, Ps are
// allowed to locally cache parts of the work queue.
//
//    a. Prepare for the mark phase by setting gcphase to _GCmark
//    (from _GCoff), enabling the write barrier, enabling mutator
//    assists, and enqueueing root mark jobs. No objects may be
//    scanned until all Ps have enabled the write barrier, which is
//    accomplished using STW.
//
//    b. Start the world. From this point, GC work is done by mark
//    workers started by the scheduler and by assists performed as
//    part of allocation. The write barrier shades both the
//    overwritten pointer and the new pointer value for any pointer
//    writes (see mbarrier.go for details). Newly allocated objects
//    are immediately marked black.
//
//    c. GC performs root marking jobs. This includes scanning all
//    stacks, shading all globals, and shading any heap pointers in
//    off-heap runtime data structures. Scanning a stack stops a
//    goroutine, shades any pointers found on its stack, and then
//    resumes the goroutine.
//
//    d. GC drains the work queue of grey objects, scanning each grey
//    object to black and shading all pointers found in the object
//    (which in turn may add those pointers to the work queue).
//
// 3. Once the global work queue is empty (but local work queue caches
// may still contain work), GC performs the "mark 2" sub-phase.
//
//    a. GC stops all workers, disables local work queue caches,
//    flushes each P's local work queue cache to the global work queue
//    cache, and reenables workers.
//
//    b. GC again drains the work queue, as in 2d above.
//
// 4. Once the work queue is empty, GC performs mark termination.
//
//    a. Stop the world.
//
//    b. Set gcphase to _GCmarktermination, and disable workers and
//    assists.
//
//    c. Drain any remaining work from the work queue (typically there
//    will be none).
//
//    d. Perform other housekeeping like flushing mcaches.
//
// 5. GC performs the sweep phase.
//
//    a. Prepare for the sweep phase by setting gcphase to _GCoff,
//    setting up sweep state and disabling the write barrier.
//
//    b. Start the world. From this point on, newly allocated objects
//    are white, and allocating sweeps spans before use if necessary.
//
//    c. GC does concurrent sweeping in the background and in response
//    to allocation. See description below.
//
// 6. When sufficient allocation has taken place, replay the sequence
// starting with 1 above. See discussion of GC rate below.

// Concurrent sweep.
//
// The sweep phase proceeds concurrently with normal program execution.
// The heap is swept span-by-span both lazily (when a goroutine needs another span)
// and concurrently in a background goroutine (this helps programs that are not CPU bound).
// At the end of STW mark termination all spans are marked as "needs sweeping".
//
// The background sweeper goroutine simply sweeps spans one-by-one.
//
// To avoid requesting more OS memory while there are unswept spans, when a
// goroutine needs another span, it first attempts to reclaim that much memory
// by sweeping. When a goroutine needs to allocate a new small-object span, it
// sweeps small-object spans for the same object size until it frees at least
// one object. When a goroutine needs to allocate large-object span from heap,
// it sweeps spans until it frees at least that many pages into heap. There is
// one case where this may not suffice: if a goroutine sweeps and frees two
// nonadjacent one-page spans to the heap, it will allocate a new two-page
// span, but there can still be other one-page unswept spans which could be
// combined into a two-page span.
//
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

// Oblets
//
// In order to prevent long pauses while scanning large objects and to
// improve parallelism, the garbage collector breaks up scan jobs for
// objects larger than maxObletBytes into "oblets" of at most
// maxObletBytes. When scanning encounters the beginning of a large
// object, it scans only the first oblet and enqueues the remaining
// oblets as new scan jobs.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

const (
	_DebugGC         = 0
	_ConcurrentSweep = true
	_FinBlockSize    = 4 * 1024

	// sweepMinHeapDistance is a lower bound on the heap distance
	// (in bytes) reserved for concurrent sweeping between GC
	// cycles. This will be scaled by gcpercent/100.
	sweepMinHeapDistance = 1024 * 1024
)

// heapminimum is the minimum heap size at which to trigger GC.
// For small heaps, this overrides the usual GOGC*live set rule.
//
// When there is a very small live set but a lot of allocation, simply
// collecting when the heap reaches GOGC*live results in many GC
// cycles and high total per-GC overhead. This minimum amortizes this
// per-GC overhead while keeping the heap reasonably small.
//
// During initialization this is set to 4MB*GOGC/100. In the case of
// GOGC==0, this will set heapminimum to 0, resulting in constant
// collection even when the heap size is small, which is useful for
// debugging.
var heapminimum uint64 = defaultHeapMinimum

// defaultHeapMinimum is the value of heapminimum for GOGC==100.
const defaultHeapMinimum = 4 << 20

// Initialized from $GOGC.  GOGC=off means no GC.
var gcpercent int32

func gcinit() {
	if unsafe.Sizeof(workbuf{}) != _WorkbufSize {
		throw("size of Workbuf is suboptimal")
	}

	_ = setGCPercent(readgogc())
	memstats.gc_trigger = heapminimum
	// Compute the goal heap size based on the trigger:
	//   trigger = marked * (1 + triggerRatio)
	//   marked = trigger / (1 + triggerRatio)
	//   goal = marked * (1 + GOGC/100)
	//        = trigger / (1 + triggerRatio) * (1 + GOGC/100)
	memstats.next_gc = uint64(float64(memstats.gc_trigger) / (1 + gcController.triggerRatio) * (1 + float64(gcpercent)/100))
	if gcpercent < 0 {
		memstats.next_gc = ^uint64(0)
	}
	work.startSema = 1
	work.markDoneSema = 1
}

func readgogc() int32 {
	p := gogetenv("GOGC")
	if p == "off" {
		return -1
	}
	if n, ok := atoi32(p); ok {
		return n
	}
	return 100
}

// gcenable is called after the bulk of the runtime initialization,
// just before we're about to start letting user code run.
// It kicks off the background sweeper goroutine and enables GC.
func gcenable() {
	c := make(chan int, 1)
	go bgsweep(c)
	<-c
	memstats.enablegc = true // now that runtime is initialized, GC is okay
}

//go:linkname setGCPercent runtime/debug.setGCPercent
func setGCPercent(in int32) (out int32) {
	lock(&mheap_.lock)
	out = gcpercent
	if in < 0 {
		in = -1
	}
	gcpercent = in
	heapminimum = defaultHeapMinimum * uint64(gcpercent) / 100
	if gcController.triggerRatio > float64(gcpercent)/100 {
		gcController.triggerRatio = float64(gcpercent) / 100
	}
	// This is either in gcinit or followed by a STW GC, both of
	// which will reset other stats like memstats.gc_trigger and
	// memstats.next_gc to appropriate values.
	unlock(&mheap_.lock)
	return out
}

// Garbage collector phase.
// Indicates to write barrier and synchronization task to perform.
var gcphase uint32

// The compiler knows about this variable.
// If you change it, you must change the compiler too.
var writeBarrier struct {
	enabled bool    // compiler emits a check of this before calling write barrier
	pad     [3]byte // compiler uses 32-bit load for "enabled" field
	needed  bool    // whether we need a write barrier for current GC phase
	cgo     bool    // whether we need a write barrier for a cgo check
	alignme uint64  // guarantee alignment so that compiler can use a 32 or 64-bit load
}

// gcBlackenEnabled is 1 if mutator assists and background mark
// workers are allowed to blacken objects. This must only be set when
// gcphase == _GCmark.
var gcBlackenEnabled uint32

// gcBlackenPromptly indicates that optimizations that may
// hide work from the global work queue should be disabled.
//
// If gcBlackenPromptly is true, per-P gcWork caches should
// be flushed immediately and new objects should be allocated black.
//
// There is a tension between allocating objects white and
// allocating them black. If white and the objects die before being
// marked they can be collected during this GC cycle. On the other
// hand allocating them black will reduce _GCmarktermination latency
// since more work is done in the mark phase. This tension is resolved
// by allocating white until the mark phase is approaching its end and
// then allocating black for the remainder of the mark phase.
var gcBlackenPromptly bool

const (
	_GCoff             = iota // GC not running; sweeping in background, write barrier disabled
	_GCmark                   // GC marking roots and workbufs: allocate black, write barrier ENABLED
	_GCmarktermination        // GC mark termination: allocate black, P's help GC, write barrier ENABLED
)

//go:nosplit
func setGCPhase(x uint32) {
	atomic.Store(&gcphase, x)
	writeBarrier.needed = gcphase == _GCmark || gcphase == _GCmarktermination
	writeBarrier.enabled = writeBarrier.needed || writeBarrier.cgo
}

// gcMarkWorkerMode represents the mode that a concurrent mark worker
// should operate in.
//
// Concurrent marking happens through four different mechanisms. One
// is mutator assists, which happen in response to allocations and are
// not scheduled. The other three are variations in the per-P mark
// workers and are distinguished by gcMarkWorkerMode.
type gcMarkWorkerMode int

const (
	// gcMarkWorkerDedicatedMode indicates that the P of a mark
	// worker is dedicated to running that mark worker. The mark
	// worker should run without preemption.
	gcMarkWorkerDedicatedMode gcMarkWorkerMode = iota

	// gcMarkWorkerFractionalMode indicates that a P is currently
	// running the "fractional" mark worker. The fractional worker
	// is necessary when GOMAXPROCS*gcGoalUtilization is not an
	// integer. The fractional worker should run until it is
	// preempted and will be scheduled to pick up the fractional
	// part of GOMAXPROCS*gcGoalUtilization.
	gcMarkWorkerFractionalMode

	// gcMarkWorkerIdleMode indicates that a P is running the mark
	// worker because it has nothing else to do. The idle worker
	// should run until it is preempted and account its time
	// against gcController.idleMarkTime.
	gcMarkWorkerIdleMode
)

// gcMarkWorkerModeStrings are the strings labels of gcMarkWorkerModes
// to use in execution traces.
var gcMarkWorkerModeStrings = [...]string{
	"GC (dedicated)",
	"GC (fractional)",
	"GC (idle)",
}

// gcController implements the GC pacing controller that determines
// when to trigger concurrent garbage collection and how much marking
// work to do in mutator assists and background marking.
//
// It uses a feedback control algorithm to adjust the memstats.gc_trigger
// trigger based on the heap growth and GC CPU utilization each cycle.
// This algorithm optimizes for heap growth to match GOGC and for CPU
// utilization between assist and background marking to be 25% of
// GOMAXPROCS. The high-level design of this algorithm is documented
// at https://golang.org/s/go15gcpacing.
var gcController = gcControllerState{
	// Initial trigger ratio guess.
	triggerRatio: 7 / 8.0,
}

type gcControllerState struct {
	// scanWork is the total scan work performed this cycle. This
	// is updated atomically during the cycle. Updates occur in
	// bounded batches, since it is both written and read
	// throughout the cycle. At the end of the cycle, this is how
	// much of the retained heap is scannable.
	//
	// Currently this is the bytes of heap scanned. For most uses,
	// this is an opaque unit of work, but for estimation the
	// definition is important.
	scanWork int64

	// bgScanCredit is the scan work credit accumulated by the
	// concurrent background scan. This credit is accumulated by
	// the background scan and stolen by mutator assists. This is
	// updated atomically. Updates occur in bounded batches, since
	// it is both written and read throughout the cycle.
	bgScanCredit int64

	// assistTime is the nanoseconds spent in mutator assists
	// during this cycle. This is updated atomically. Updates
	// occur in bounded batches, since it is both written and read
	// throughout the cycle.
	assistTime int64

	// dedicatedMarkTime is the nanoseconds spent in dedicated
	// mark workers during this cycle. This is updated atomically
	// at the end of the concurrent mark phase.
	dedicatedMarkTime int64

	// fractionalMarkTime is the nanoseconds spent in the
	// fractional mark worker during this cycle. This is updated
	// atomically throughout the cycle and will be up-to-date if
	// the fractional mark worker is not currently running.
	fractionalMarkTime int64

	// idleMarkTime is the nanoseconds spent in idle marking
	// during this cycle. This is updated atomically throughout
	// the cycle.
	idleMarkTime int64

	// markStartTime is the absolute start time in nanoseconds
	// that assists and background mark workers started.
	markStartTime int64

	// dedicatedMarkWorkersNeeded is the number of dedicated mark
	// workers that need to be started. This is computed at the
	// beginning of each cycle and decremented atomically as
	// dedicated mark workers get started.
	dedicatedMarkWorkersNeeded int64

	// assistWorkPerByte is the ratio of scan work to allocated
	// bytes that should be performed by mutator assists. This is
	// computed at the beginning of each cycle and updated every
	// time heap_scan is updated.
	assistWorkPerByte float64

	// assistBytesPerWork is 1/assistWorkPerByte.
	assistBytesPerWork float64

	// fractionalUtilizationGoal is the fraction of wall clock
	// time that should be spent in the fractional mark worker.
	// For example, if the overall mark utilization goal is 25%
	// and GOMAXPROCS is 6, one P will be a dedicated mark worker
	// and this will be set to 0.5 so that 50% of the time some P
	// is in a fractional mark worker. This is computed at the
	// beginning of each cycle.
	fractionalUtilizationGoal float64

	// triggerRatio is the heap growth ratio at which the garbage
	// collection cycle should start. E.g., if this is 0.6, then
	// GC should start when the live heap has reached 1.6 times
	// the heap size marked by the previous cycle. This should be
	// ≤ GOGC/100 so the trigger heap size is less than the goal
	// heap size. This is updated at the end of of each cycle.
	triggerRatio float64

	_ [sys.CacheLineSize]byte

	// fractionalMarkWorkersNeeded is the number of fractional
	// mark workers that need to be started. This is either 0 or
	// 1. This is potentially updated atomically at every
	// scheduling point (hence it gets its own cache line).
	fractionalMarkWorkersNeeded int64

	_ [sys.CacheLineSize]byte
}

// startCycle resets the GC controller's state and computes estimates
// for a new GC cycle. The caller must hold worldsema.
func (c *gcControllerState) startCycle() {
	c.scanWork = 0
	c.bgScanCredit = 0
	c.assistTime = 0
	c.dedicatedMarkTime = 0
	c.fractionalMarkTime = 0
	c.idleMarkTime = 0

	// If this is the first GC cycle or we're operating on a very
	// small heap, fake heap_marked so it looks like gc_trigger is
	// the appropriate growth from heap_marked, even though the
	// real heap_marked may not have a meaningful value (on the
	// first cycle) or may be much smaller (resulting in a large
	// error response).
	if memstats.gc_trigger <= heapminimum {
		memstats.heap_marked = uint64(float64(memstats.gc_trigger) / (1 + c.triggerRatio))
	}

	// Re-compute the heap goal for this cycle in case something
	// changed. This is the same calculation we use elsewhere.
	memstats.next_gc = memstats.heap_marked + memstats.heap_marked*uint64(gcpercent)/100
	if gcpercent < 0 {
		memstats.next_gc = ^uint64(0)
	}

	// Ensure that the heap goal is at least a little larger than
	// the current live heap size. This may not be the case if GC
	// start is delayed or if the allocation that pushed heap_live
	// over gc_trigger is large or if the trigger is really close to
	// GOGC. Assist is proportional to this distance, so enforce a
	// minimum distance, even if it means going over the GOGC goal
	// by a tiny bit.
	if memstats.next_gc < memstats.heap_live+1024*1024 {
		memstats.next_gc = memstats.heap_live + 1024*1024
	}

	// Compute the total mark utilization goal and divide it among
	// dedicated and fractional workers.
	totalUtilizationGoal := float64(gomaxprocs) * gcGoalUtilization
	c.dedicatedMarkWorkersNeeded = int64(totalUtilizationGoal)
	c.fractionalUtilizationGoal = totalUtilizationGoal - float64(c.dedicatedMarkWorkersNeeded)
	if c.fractionalUtilizationGoal > 0 {
		c.fractionalMarkWorkersNeeded = 1
	} else {
		c.fractionalMarkWorkersNeeded = 0
	}

	// Clear per-P state
	for _, p := range &allp {
		if p == nil {
			break
		}
		p.gcAssistTime = 0
	}

	// Compute initial values for controls that are updated
	// throughout the cycle.
	c.revise()

	if debug.gcpacertrace > 0 {
		print("pacer: assist ratio=", c.assistWorkPerByte,
			" (scan ", memstats.heap_scan>>20, " MB in ",
			work.initialHeapLive>>20, "->",
			memstats.next_gc>>20, " MB)",
			" workers=", c.dedicatedMarkWorkersNeeded,
			"+", c.fractionalMarkWorkersNeeded, "\n")
	}
}

// revise updates the assist ratio during the GC cycle to account for
// improved estimates. This should be called either under STW or
// whenever memstats.heap_scan or memstats.heap_live is updated (with
// mheap_.lock held).
//
// It should only be called when gcBlackenEnabled != 0 (because this
// is when assists are enabled and the necessary statistics are
// available).
//
// TODO: Consider removing the periodic controller update altogether.
// Since we switched to allocating black, in theory we shouldn't have
// to change the assist ratio. However, this is still a useful hook
// that we've found many uses for when experimenting.
func (c *gcControllerState) revise() {
	// Compute the expected scan work remaining.
	//
	// Note that we currently count allocations during GC as both
	// scannable heap (heap_scan) and scan work completed
	// (scanWork), so this difference won't be changed by
	// allocations during GC.
	//
	// This particular estimate is a strict upper bound on the
	// possible remaining scan work for the current heap.
	// You might consider dividing this by 2 (or by
	// (100+GOGC)/100) to counter this over-estimation, but
	// benchmarks show that this has almost no effect on mean
	// mutator utilization, heap size, or assist time and it
	// introduces the danger of under-estimating and letting the
	// mutator outpace the garbage collector.
	scanWorkExpected := int64(memstats.heap_scan) - c.scanWork
	if scanWorkExpected < 1000 {
		// We set a somewhat arbitrary lower bound on
		// remaining scan work since if we aim a little high,
		// we can miss by a little.
		//
		// We *do* need to enforce that this is at least 1,
		// since marking is racy and double-scanning objects
		// may legitimately make the expected scan work
		// negative.
		scanWorkExpected = 1000
	}

	// Compute the heap distance remaining.
	heapDistance := int64(memstats.next_gc) - int64(memstats.heap_live)
	if heapDistance <= 0 {
		// This shouldn't happen, but if it does, avoid
		// dividing by zero or setting the assist negative.
		heapDistance = 1
	}

	// Compute the mutator assist ratio so by the time the mutator
	// allocates the remaining heap bytes up to next_gc, it will
	// have done (or stolen) the remaining amount of scan work.
	c.assistWorkPerByte = float64(scanWorkExpected) / float64(heapDistance)
	c.assistBytesPerWork = float64(heapDistance) / float64(scanWorkExpected)
}

// endCycle updates the GC controller state at the end of the
// concurrent part of the GC cycle.
func (c *gcControllerState) endCycle() {
	h_t := c.triggerRatio // For debugging

	// Proportional response gain for the trigger controller. Must
	// be in [0, 1]. Lower values smooth out transient effects but
	// take longer to respond to phase changes. Higher values
	// react to phase changes quickly, but are more affected by
	// transient changes. Values near 1 may be unstable.
	const triggerGain = 0.5

	// Compute next cycle trigger ratio. First, this computes the
	// "error" for this cycle; that is, how far off the trigger
	// was from what it should have been, accounting for both heap
	// growth and GC CPU utilization. We compute the actual heap
	// growth during this cycle and scale that by how far off from
	// the goal CPU utilization we were (to estimate the heap
	// growth if we had the desired CPU utilization). The
	// difference between this estimate and the GOGC-based goal
	// heap growth is the error.
	goalGrowthRatio := float64(gcpercent) / 100
	actualGrowthRatio := float64(memstats.heap_live)/float64(memstats.heap_marked) - 1
	assistDuration := nanotime() - c.markStartTime

	// Assume background mark hit its utilization goal.
	utilization := gcGoalUtilization
	// Add assist utilization; avoid divide by zero.
	if assistDuration > 0 {
		utilization += float64(c.assistTime) / float64(assistDuration*int64(gomaxprocs))
	}

	triggerError := goalGrowthRatio - c.triggerRatio - utilization/gcGoalUtilization*(actualGrowthRatio-c.triggerRatio)

	// Finally, we adjust the trigger for next time by this error,
	// damped by the proportional gain.
	c.triggerRatio += triggerGain * triggerError
	if c.triggerRatio < 0 {
		// This can happen if the mutator is allocating very
		// quickly or the GC is scanning very slowly.
		c.triggerRatio = 0
	} else if c.triggerRatio > goalGrowthRatio*0.95 {
		// Ensure there's always a little margin so that the
		// mutator assist ratio isn't infinity.
		c.triggerRatio = goalGrowthRatio * 0.95
	}

	if debug.gcpacertrace > 0 {
		// Print controller state in terms of the design
		// document.
		H_m_prev := memstats.heap_marked
		H_T := memstats.gc_trigger
		h_a := actualGrowthRatio
		H_a := memstats.heap_live
		h_g := goalGrowthRatio
		H_g := int64(float64(H_m_prev) * (1 + h_g))
		u_a := utilization
		u_g := gcGoalUtilization
		W_a := c.scanWork
		print("pacer: H_m_prev=", H_m_prev,
			" h_t=", h_t, " H_T=", H_T,
			" h_a=", h_a, " H_a=", H_a,
			" h_g=", h_g, " H_g=", H_g,
			" u_a=", u_a, " u_g=", u_g,
			" W_a=", W_a,
			" goalΔ=", goalGrowthRatio-h_t,
			" actualΔ=", h_a-h_t,
			" u_a/u_g=", u_a/u_g,
			"\n")
	}
}

// enlistWorker encourages another dedicated mark worker to start on
// another P if there are spare worker slots. It is used by putfull
// when more work is made available.
//
//go:nowritebarrier
func (c *gcControllerState) enlistWorker() {
	// If there are idle Ps, wake one so it will run an idle worker.
	// NOTE: This is suspected of causing deadlocks. See golang.org/issue/19112.
	//
	//	if atomic.Load(&sched.npidle) != 0 && atomic.Load(&sched.nmspinning) == 0 {
	//		wakep()
	//		return
	//	}

	// There are no idle Ps. If we need more dedicated workers,
	// try to preempt a running P so it will switch to a worker.
	if c.dedicatedMarkWorkersNeeded <= 0 {
		return
	}
	// Pick a random other P to preempt.
	if gomaxprocs <= 1 {
		return
	}
	gp := getg()
	if gp == nil || gp.m == nil || gp.m.p == 0 {
		return
	}
	myID := gp.m.p.ptr().id
	for tries := 0; tries < 5; tries++ {
		id := int32(fastrand() % uint32(gomaxprocs-1))
		if id >= myID {
			id++
		}
		p := allp[id]
		if p.status != _Prunning {
			continue
		}
		if preemptone(p) {
			return
		}
	}
}

// findRunnableGCWorker returns the background mark worker for _p_ if it
// should be run. This must only be called when gcBlackenEnabled != 0.
func (c *gcControllerState) findRunnableGCWorker(_p_ *p) *g {
	if gcBlackenEnabled == 0 {
		throw("gcControllerState.findRunnable: blackening not enabled")
	}
	if _p_.gcBgMarkWorker == 0 {
		// The mark worker associated with this P is blocked
		// performing a mark transition. We can't run it
		// because it may be on some other run or wait queue.
		return nil
	}

	if !gcMarkWorkAvailable(_p_) {
		// No work to be done right now. This can happen at
		// the end of the mark phase when there are still
		// assists tapering off. Don't bother running a worker
		// now because it'll just return immediately.
		return nil
	}

	decIfPositive := func(ptr *int64) bool {
		if *ptr > 0 {
			if atomic.Xaddint64(ptr, -1) >= 0 {
				return true
			}
			// We lost a race
			atomic.Xaddint64(ptr, +1)
		}
		return false
	}

	if decIfPositive(&c.dedicatedMarkWorkersNeeded) {
		// This P is now dedicated to marking until the end of
		// the concurrent mark phase.
		_p_.gcMarkWorkerMode = gcMarkWorkerDedicatedMode
		// TODO(austin): This P isn't going to run anything
		// else for a while, so kick everything out of its run
		// queue.
	} else {
		if !decIfPositive(&c.fractionalMarkWorkersNeeded) {
			// No more workers are need right now.
			return nil
		}

		// This P has picked the token for the fractional worker.
		// Is the GC currently under or at the utilization goal?
		// If so, do more work.
		//
		// We used to check whether doing one time slice of work
		// would remain under the utilization goal, but that has the
		// effect of delaying work until the mutator has run for
		// enough time slices to pay for the work. During those time
		// slices, write barriers are enabled, so the mutator is running slower.
		// Now instead we do the work whenever we're under or at the
		// utilization work and pay for it by letting the mutator run later.
		// This doesn't change the overall utilization averages, but it
		// front loads the GC work so that the GC finishes earlier and
		// write barriers can be turned off sooner, effectively giving
		// the mutator a faster machine.
		//
		// The old, slower behavior can be restored by setting
		//	gcForcePreemptNS = forcePreemptNS.
		const gcForcePreemptNS = 0

		// TODO(austin): We could fast path this and basically
		// eliminate contention on c.fractionalMarkWorkersNeeded by
		// precomputing the minimum time at which it's worth
		// next scheduling the fractional worker. Then Ps
		// don't have to fight in the window where we've
		// passed that deadline and no one has started the
		// worker yet.
		//
		// TODO(austin): Shorter preemption interval for mark
		// worker to improve fairness and give this
		// finer-grained control over schedule?
		now := nanotime() - gcController.markStartTime
		then := now + gcForcePreemptNS
		timeUsed := c.fractionalMarkTime + gcForcePreemptNS
		if then > 0 && float64(timeUsed)/float64(then) > c.fractionalUtilizationGoal {
			// Nope, we'd overshoot the utilization goal
			atomic.Xaddint64(&c.fractionalMarkWorkersNeeded, +1)
			return nil
		}
		_p_.gcMarkWorkerMode = gcMarkWorkerFractionalMode
	}

	// Run the background mark worker
	gp := _p_.gcBgMarkWorker.ptr()
	casgstatus(gp, _Gwaiting, _Grunnable)
	if trace.enabled {
		traceGoUnpark(gp, 0)
	}
	return gp
}

// gcGoalUtilization is the goal CPU utilization for background
// marking as a fraction of GOMAXPROCS.
const gcGoalUtilization = 0.25

// gcCreditSlack is the amount of scan work credit that can can
// accumulate locally before updating gcController.scanWork and,
// optionally, gcController.bgScanCredit. Lower values give a more
// accurate assist ratio and make it more likely that assists will
// successfully steal background credit. Higher values reduce memory
// contention.
const gcCreditSlack = 2000

// gcAssistTimeSlack is the nanoseconds of mutator assist time that
// can accumulate on a P before updating gcController.assistTime.
const gcAssistTimeSlack = 5000

// gcOverAssistWork determines how many extra units of scan work a GC
// assist does when an assist happens. This amortizes the cost of an
// assist by pre-paying for this many bytes of future allocations.
const gcOverAssistWork = 64 << 10

var work struct {
	full  uint64                   // lock-free list of full blocks workbuf
	empty uint64                   // lock-free list of empty blocks workbuf
	pad0  [sys.CacheLineSize]uint8 // prevents false-sharing between full/empty and nproc/nwait

	// bytesMarked is the number of bytes marked this cycle. This
	// includes bytes blackened in scanned objects, noscan objects
	// that go straight to black, and permagrey objects scanned by
	// markroot during the concurrent scan phase. This is updated
	// atomically during the cycle. Updates may be batched
	// arbitrarily, since the value is only read at the end of the
	// cycle.
	//
	// Because of benign races during marking, this number may not
	// be the exact number of marked bytes, but it should be very
	// close.
	//
	// Put this field here because it needs 64-bit atomic access
	// (and thus 8-byte alignment even on 32-bit architectures).
	bytesMarked uint64

	markrootNext uint32 // next markroot job
	markrootJobs uint32 // number of markroot jobs

	nproc   uint32
	tstart  int64
	nwait   uint32
	ndone   uint32
	alldone note

	// helperDrainBlock indicates that GC mark termination helpers
	// should pass gcDrainBlock to gcDrain to block in the
	// getfull() barrier. Otherwise, they should pass gcDrainNoBlock.
	//
	// TODO: This is a temporary fallback to support
	// debug.gcrescanstacks > 0 and to work around some known
	// races. Remove this when we remove the debug option and fix
	// the races.
	helperDrainBlock bool

	// Number of roots of various root types. Set by gcMarkRootPrepare.
	nFlushCacheRoots                                             int
	nDataRoots, nBSSRoots, nSpanRoots, nStackRoots, nRescanRoots int

	// markrootDone indicates that roots have been marked at least
	// once during the current GC cycle. This is checked by root
	// marking operations that have to happen only during the
	// first root marking pass, whether that's during the
	// concurrent mark phase in current GC or mark termination in
	// STW GC.
	markrootDone bool

	// Each type of GC state transition is protected by a lock.
	// Since multiple threads can simultaneously detect the state
	// transition condition, any thread that detects a transition
	// condition must acquire the appropriate transition lock,
	// re-check the transition condition and return if it no
	// longer holds or perform the transition if it does.
	// Likewise, any transition must invalidate the transition
	// condition before releasing the lock. This ensures that each
	// transition is performed by exactly one thread and threads
	// that need the transition to happen block until it has
	// happened.
	//
	// startSema protects the transition from "off" to mark or
	// mark termination.
	startSema uint32
	// markDoneSema protects transitions from mark 1 to mark 2 and
	// from mark 2 to mark termination.
	markDoneSema uint32

	bgMarkReady note   // signal background mark worker has started
	bgMarkDone  uint32 // cas to 1 when at a background mark completion point
	// Background mark completion signaling

	// mode is the concurrency mode of the current GC cycle.
	mode gcMode

	// totaltime is the CPU nanoseconds spent in GC since the
	// program started if debug.gctrace > 0.
	totaltime int64

	// initialHeapLive is the value of memstats.heap_live at the
	// beginning of this GC cycle.
	initialHeapLive uint64

	// assistQueue is a queue of assists that are blocked because
	// there was neither enough credit to steal or enough work to
	// do.
	assistQueue struct {
		lock       mutex
		head, tail guintptr
	}

	// rescan is a list of G's that need to be rescanned during
	// mark termination. A G adds itself to this list when it
	// first invalidates its stack scan.
	rescan struct {
		lock mutex
		list []guintptr
	}

	// Timing/utilization stats for this cycle.
	stwprocs, maxprocs                 int32
	tSweepTerm, tMark, tMarkTerm, tEnd int64 // nanotime() of phase start

	pauseNS    int64 // total STW time this cycle
	pauseStart int64 // nanotime() of last STW

	// debug.gctrace heap sizes for this cycle.
	heap0, heap1, heap2, heapGoal uint64
}

// GC runs a garbage collection and blocks the caller until the
// garbage collection is complete. It may also block the entire
// program.
func GC() {
	gcStart(gcForceBlockMode, false)
}

// gcMode indicates how concurrent a GC cycle should be.
type gcMode int

const (
	gcBackgroundMode gcMode = iota // concurrent GC and sweep
	gcForceMode                    // stop-the-world GC now, concurrent sweep
	gcForceBlockMode               // stop-the-world GC now and STW sweep (forced by user)
)

// gcShouldStart returns true if the exit condition for the _GCoff
// phase has been met. The exit condition should be tested when
// allocating.
//
// If forceTrigger is true, it ignores the current heap size, but
// checks all other conditions. In general this should be false.
func gcShouldStart(forceTrigger bool) bool {
	return gcphase == _GCoff && (forceTrigger || memstats.heap_live >= memstats.gc_trigger) && memstats.enablegc && panicking == 0 && gcpercent >= 0
}

// gcStart transitions the GC from _GCoff to _GCmark (if mode ==
// gcBackgroundMode) or _GCmarktermination (if mode !=
// gcBackgroundMode) by performing sweep termination and GC
// initialization.
//
// This may return without performing this transition in some cases,
// such as when called on a system stack or with locks held.
func gcStart(mode gcMode, forceTrigger bool) {
	// Since this is called from malloc and malloc is called in
	// the guts of a number of libraries that might be holding
	// locks, don't attempt to start GC in non-preemptible or
	// potentially unstable situations.
	mp := acquirem()
	if gp := getg(); gp == mp.g0 || mp.locks > 1 || mp.preemptoff != "" {
		releasem(mp)
		return
	}
	releasem(mp)
	mp = nil

	// Pick up the remaining unswept/not being swept spans concurrently
	//
	// This shouldn't happen if we're being invoked in background
	// mode since proportional sweep should have just finished
	// sweeping everything, but rounding errors, etc, may leave a
	// few spans unswept. In forced mode, this is necessary since
	// GC can be forced at any point in the sweeping cycle.
	//
	// We check the transition condition continuously here in case
	// this G gets delayed in to the next GC cycle.
	for (mode != gcBackgroundMode || gcShouldStart(forceTrigger)) && gosweepone() != ^uintptr(0) {
		sweep.nbgsweep++
	}

	// Perform GC initialization and the sweep termination
	// transition.
	//
	// If this is a forced GC, don't acquire the transition lock
	// or re-check the transition condition because we
	// specifically *don't* want to share the transition with
	// another thread.
	useStartSema := mode == gcBackgroundMode
	if useStartSema {
		semacquire(&work.startSema, 0)
		// Re-check transition condition under transition lock.
		if !gcShouldStart(forceTrigger) {
			semrelease(&work.startSema)
			return
		}
	}

	// For stats, check if this GC was forced by the user.
	forced := mode != gcBackgroundMode

	// In gcstoptheworld debug mode, upgrade the mode accordingly.
	// We do this after re-checking the transition condition so
	// that multiple goroutines that detect the heap trigger don't
	// start multiple STW GCs.
	if mode == gcBackgroundMode {
		if debug.gcstoptheworld == 1 {
			mode = gcForceMode
		} else if debug.gcstoptheworld == 2 {
			mode = gcForceBlockMode
		}
	}

	// Ok, we're doing it!  Stop everybody else
	semacquire(&worldsema, 0)

	if trace.enabled {
		traceGCStart()
	}

	if mode == gcBackgroundMode {
		gcBgMarkStartWorkers()
	}

	gcResetMarkState()

	now := nanotime()
	work.stwprocs, work.maxprocs = gcprocs(), gomaxprocs
	work.tSweepTerm = now
	work.heap0 = memstats.heap_live
	work.pauseNS = 0
	work.mode = mode

	work.pauseStart = now
	systemstack(stopTheWorldWithSema)
	// Finish sweep before we start concurrent scan.
	systemstack(func() {
		finishsweep_m()
	})
	// clearpools before we start the GC. If we wait they memory will not be
	// reclaimed until the next GC cycle.
	clearpools()

	if mode == gcBackgroundMode { // Do as much work concurrently as possible
		gcController.startCycle()
		work.heapGoal = memstats.next_gc

		// Enter concurrent mark phase and enable
		// write barriers.
		//
		// Because the world is stopped, all Ps will
		// observe that write barriers are enabled by
		// the time we start the world and begin
		// scanning.
		//
		// It's necessary to enable write barriers
		// during the scan phase for several reasons:
		//
		// They must be enabled for writes to higher
		// stack frames before we scan stacks and
		// install stack barriers because this is how
		// we track writes to inactive stack frames.
		// (Alternatively, we could not install stack
		// barriers over frame boundaries with
		// up-pointers).
		//
		// They must be enabled before assists are
		// enabled because they must be enabled before
		// any non-leaf heap objects are marked. Since
		// allocations are blocked until assists can
		// happen, we want enable assists as early as
		// possible.
		setGCPhase(_GCmark)

		gcBgMarkPrepare() // Must happen before assist enable.
		gcMarkRootPrepare()

		// Mark all active tinyalloc blocks. Since we're
		// allocating from these, they need to be black like
		// other allocations. The alternative is to blacken
		// the tiny block on every allocation from it, which
		// would slow down the tiny allocator.
		gcMarkTinyAllocs()

		// At this point all Ps have enabled the write
		// barrier, thus maintaining the no white to
		// black invariant. Enable mutator assists to
		// put back-pressure on fast allocating
		// mutators.
		atomic.Store(&gcBlackenEnabled, 1)

		// Assists and workers can start the moment we start
		// the world.
		gcController.markStartTime = now

		// Concurrent mark.
		systemstack(startTheWorldWithSema)
		now = nanotime()
		work.pauseNS += now - work.pauseStart
		work.tMark = now
	} else {
		t := nanotime()
		work.tMark, work.tMarkTerm = t, t
		work.heapGoal = work.heap0

		if forced {
			memstats.numforcedgc++
		}

		// Perform mark termination. This will restart the world.
		gcMarkTermination()
	}

	if useStartSema {
		semrelease(&work.startSema)
	}
}

// gcMarkDone transitions the GC from mark 1 to mark 2 and from mark 2
// to mark termination.
//
// This should be called when all mark work has been drained. In mark
// 1, this includes all root marking jobs, global work buffers, and
// active work buffers in assists and background workers; however,
// work may still be cached in per-P work buffers. In mark 2, per-P
// caches are disabled.
//
// The calling context must be preemptible.
//
// Note that it is explicitly okay to have write barriers in this
// function because completion of concurrent mark is best-effort
// anyway. Any work created by write barriers here will be cleaned up
// by mark termination.
func gcMarkDone() {
top:
	semacquire(&work.markDoneSema, 0)

	// Re-check transition condition under transition lock.
	if !(gcphase == _GCmark && work.nwait == work.nproc && !gcMarkWorkAvailable(nil)) {
		semrelease(&work.markDoneSema)
		return
	}

	// Disallow starting new workers so that any remaining workers
	// in the current mark phase will drain out.
	//
	// TODO(austin): Should dedicated workers keep an eye on this
	// and exit gcDrain promptly?
	atomic.Xaddint64(&gcController.dedicatedMarkWorkersNeeded, -0xffffffff)
	atomic.Xaddint64(&gcController.fractionalMarkWorkersNeeded, -0xffffffff)

	if !gcBlackenPromptly {
		// Transition from mark 1 to mark 2.
		//
		// The global work list is empty, but there can still be work
		// sitting in the per-P work caches.
		// Flush and disable work caches.

		// Disallow caching workbufs and indicate that we're in mark 2.
		gcBlackenPromptly = true

		// Prevent completion of mark 2 until we've flushed
		// cached workbufs.
		atomic.Xadd(&work.nwait, -1)

		// GC is set up for mark 2. Let Gs blocked on the
		// transition lock go while we flush caches.
		semrelease(&work.markDoneSema)

		systemstack(func() {
			// Flush all currently cached workbufs and
			// ensure all Ps see gcBlackenPromptly. This
			// also blocks until any remaining mark 1
			// workers have exited their loop so we can
			// start new mark 2 workers.
			forEachP(func(_p_ *p) {
				_p_.gcw.dispose()
			})
		})

		// Check that roots are marked. We should be able to
		// do this before the forEachP, but based on issue
		// #16083 there may be a (harmless) race where we can
		// enter mark 2 while some workers are still scanning
		// stacks. The forEachP ensures these scans are done.
		//
		// TODO(austin): Figure out the race and fix this
		// properly.
		gcMarkRootCheck()

		// Now we can start up mark 2 workers.
		atomic.Xaddint64(&gcController.dedicatedMarkWorkersNeeded, 0xffffffff)
		atomic.Xaddint64(&gcController.fractionalMarkWorkersNeeded, 0xffffffff)

		incnwait := atomic.Xadd(&work.nwait, +1)
		if incnwait == work.nproc && !gcMarkWorkAvailable(nil) {
			// This loop will make progress because
			// gcBlackenPromptly is now true, so it won't
			// take this same "if" branch.
			goto top
		}
	} else {
		// Transition to mark termination.
		now := nanotime()
		work.tMarkTerm = now
		work.pauseStart = now
		getg().m.preemptoff = "gcing"
		systemstack(stopTheWorldWithSema)
		// The gcphase is _GCmark, it will transition to _GCmarktermination
		// below. The important thing is that the wb remains active until
		// all marking is complete. This includes writes made by the GC.

		// Record that one root marking pass has completed.
		work.markrootDone = true

		// Disable assists and background workers. We must do
		// this before waking blocked assists.
		atomic.Store(&gcBlackenEnabled, 0)

		// Wake all blocked assists. These will run when we
		// start the world again.
		gcWakeAllAssists()

		// Likewise, release the transition lock. Blocked
		// workers and assists will run when we start the
		// world again.
		semrelease(&work.markDoneSema)

		// endCycle depends on all gcWork cache stats being
		// flushed. This is ensured by mark 2.
		gcController.endCycle()

		// Perform mark termination. This will restart the world.
		gcMarkTermination()
	}
}

func gcMarkTermination() {
	// World is stopped.
	// Start marktermination which includes enabling the write barrier.
	atomic.Store(&gcBlackenEnabled, 0)
	gcBlackenPromptly = false
	setGCPhase(_GCmarktermination)

	work.heap1 = memstats.heap_live
	startTime := nanotime()

	mp := acquirem()
	mp.preemptoff = "gcing"
	_g_ := getg()
	_g_.m.traceback = 2
	gp := _g_.m.curg
	casgstatus(gp, _Grunning, _Gwaiting)
	gp.waitreason = "garbage collection"

	// Run gc on the g0 stack. We do this so that the g stack
	// we're currently running on will no longer change. Cuts
	// the root set down a bit (g0 stacks are not scanned, and
	// we don't need to scan gc's internal state).  We also
	// need to switch to g0 so we can shrink the stack.
	systemstack(func() {
		gcMark(startTime)
		// Must return immediately.
		// The outer function's stack may have moved
		// during gcMark (it shrinks stacks, including the
		// outer function's stack), so we must not refer
		// to any of its variables. Return back to the
		// non-system stack to pick up the new addresses
		// before continuing.
	})

	systemstack(func() {
		work.heap2 = work.bytesMarked
		if debug.gccheckmark > 0 {
			// Run a full stop-the-world mark using checkmark bits,
			// to check that we didn't forget to mark anything during
			// the concurrent mark process.
			gcResetMarkState()
			initCheckmarks()
			gcMark(startTime)
			clearCheckmarks()
		}

		// marking is complete so we can turn the write barrier off
		setGCPhase(_GCoff)
		gcSweep(work.mode)

		if debug.gctrace > 1 {
			startTime = nanotime()
			// The g stacks have been scanned so
			// they have gcscanvalid==true and gcworkdone==true.
			// Reset these so that all stacks will be rescanned.
			gcResetMarkState()
			finishsweep_m()

			// Still in STW but gcphase is _GCoff, reset to _GCmarktermination
			// At this point all objects will be found during the gcMark which
			// does a complete STW mark and object scan.
			setGCPhase(_GCmarktermination)
			gcMark(startTime)
			setGCPhase(_GCoff) // marking is done, turn off wb.
			gcSweep(work.mode)
		}
	})

	_g_.m.traceback = 0
	casgstatus(gp, _Gwaiting, _Grunning)

	if trace.enabled {
		traceGCDone()
	}

	// all done
	mp.preemptoff = ""

	if gcphase != _GCoff {
		throw("gc done but gcphase != _GCoff")
	}

	// Update timing memstats
	now, unixNow := nanotime(), unixnanotime()
	work.pauseNS += now - work.pauseStart
	work.tEnd = now
	atomic.Store64(&memstats.last_gc, uint64(unixNow)) // must be Unix time to make sense to user
	memstats.pause_ns[memstats.numgc%uint32(len(memstats.pause_ns))] = uint64(work.pauseNS)
	memstats.pause_end[memstats.numgc%uint32(len(memstats.pause_end))] = uint64(unixNow)
	memstats.pause_total_ns += uint64(work.pauseNS)

	// Update work.totaltime.
	sweepTermCpu := int64(work.stwprocs) * (work.tMark - work.tSweepTerm)
	// We report idle marking time below, but omit it from the
	// overall utilization here since it's "free".
	markCpu := gcController.assistTime + gcController.dedicatedMarkTime + gcController.fractionalMarkTime
	markTermCpu := int64(work.stwprocs) * (work.tEnd - work.tMarkTerm)
	cycleCpu := sweepTermCpu + markCpu + markTermCpu
	work.totaltime += cycleCpu

	// Compute overall GC CPU utilization.
	totalCpu := sched.totaltime + (now-sched.procresizetime)*int64(gomaxprocs)
	memstats.gc_cpu_fraction = float64(work.totaltime) / float64(totalCpu)

	memstats.numgc++

	// Reset sweep state.
	sweep.nbgsweep = 0
	sweep.npausesweep = 0

	systemstack(startTheWorldWithSema)

	// Update heap profile stats if gcSweep didn't do it. This is
	// relatively expensive, so we don't want to do it while the
	// world is stopped, but it needs to happen ASAP after
	// starting the world to prevent too many allocations from the
	// next cycle leaking in. It must happen before releasing
	// worldsema since there are applications that do a
	// runtime.GC() to update the heap profile and then
	// immediately collect the profile.
	if _ConcurrentSweep && work.mode != gcForceBlockMode {
		mProf_GC()
	}

	// Free stack spans. This must be done between GC cycles.
	systemstack(freeStackSpans)

	// Best-effort remove stack barriers so they don't get in the
	// way of things like GDB and perf.
	lock(&allglock)
	myallgs := allgs
	unlock(&allglock)
	gcTryRemoveAllStackBarriers(myallgs)

	// Print gctrace before dropping worldsema. As soon as we drop
	// worldsema another cycle could start and smash the stats
	// we're trying to print.
	if debug.gctrace > 0 {
		util := int(memstats.gc_cpu_fraction * 100)

		var sbuf [24]byte
		printlock()
		print("gc ", memstats.numgc,
			" @", string(itoaDiv(sbuf[:], uint64(work.tSweepTerm-runtimeInitTime)/1e6, 3)), "s ",
			util, "%: ")
		prev := work.tSweepTerm
		for i, ns := range []int64{work.tMark, work.tMarkTerm, work.tEnd} {
			if i != 0 {
				print("+")
			}
			print(string(fmtNSAsMS(sbuf[:], uint64(ns-prev))))
			prev = ns
		}
		print(" ms clock, ")
		for i, ns := range []int64{sweepTermCpu, gcController.assistTime, gcController.dedicatedMarkTime + gcController.fractionalMarkTime, gcController.idleMarkTime, markTermCpu} {
			if i == 2 || i == 3 {
				// Separate mark time components with /.
				print("/")
			} else if i != 0 {
				print("+")
			}
			print(string(fmtNSAsMS(sbuf[:], uint64(ns))))
		}
		print(" ms cpu, ",
			work.heap0>>20, "->", work.heap1>>20, "->", work.heap2>>20, " MB, ",
			work.heapGoal>>20, " MB goal, ",
			work.maxprocs, " P")
		if work.mode != gcBackgroundMode {
			print(" (forced)")
		}
		print("\n")
		printunlock()
	}

	semrelease(&worldsema)
	// Careful: another GC cycle may start now.

	releasem(mp)
	mp = nil

	// now that gc is done, kick off finalizer thread if needed
	if !concurrentSweep {
		// give the queued finalizers, if any, a chance to run
		Gosched()
	}
}

// gcBgMarkStartWorkers prepares background mark worker goroutines.
// These goroutines will not run until the mark phase, but they must
// be started while the work is not stopped and from a regular G
// stack. The caller must hold worldsema.
func gcBgMarkStartWorkers() {
	// Background marking is performed by per-P G's. Ensure that
	// each P has a background GC G.
	for _, p := range &allp {
		if p == nil || p.status == _Pdead {
			break
		}
		if p.gcBgMarkWorker == 0 {
			go gcBgMarkWorker(p)
			notetsleepg(&work.bgMarkReady, -1)
			noteclear(&work.bgMarkReady)
		}
	}
}

// gcBgMarkPrepare sets up state for background marking.
// Mutator assists must not yet be enabled.
func gcBgMarkPrepare() {
	// Background marking will stop when the work queues are empty
	// and there are no more workers (note that, since this is
	// concurrent, this may be a transient state, but mark
	// termination will clean it up). Between background workers
	// and assists, we don't really know how many workers there
	// will be, so we pretend to have an arbitrarily large number
	// of workers, almost all of which are "waiting". While a
	// worker is working it decrements nwait. If nproc == nwait,
	// there are no workers.
	work.nproc = ^uint32(0)
	work.nwait = ^uint32(0)
}

func gcBgMarkWorker(_p_ *p) {
	gp := getg()

	type parkInfo struct {
		m      muintptr // Release this m on park.
		attach puintptr // If non-nil, attach to this p on park.
	}
	// We pass park to a gopark unlock function, so it can't be on
	// the stack (see gopark). Prevent deadlock from recursively
	// starting GC by disabling preemption.
	gp.m.preemptoff = "GC worker init"
	park := new(parkInfo)
	gp.m.preemptoff = ""

	park.m.set(acquirem())
	park.attach.set(_p_)
	// Inform gcBgMarkStartWorkers that this worker is ready.
	// After this point, the background mark worker is scheduled
	// cooperatively by gcController.findRunnable. Hence, it must
	// never be preempted, as this would put it into _Grunnable
	// and put it on a run queue. Instead, when the preempt flag
	// is set, this puts itself into _Gwaiting to be woken up by
	// gcController.findRunnable at the appropriate time.
	notewakeup(&work.bgMarkReady)

	for {
		// Go to sleep until woken by gcController.findRunnable.
		// We can't releasem yet since even the call to gopark
		// may be preempted.
		gopark(func(g *g, parkp unsafe.Pointer) bool {
			park := (*parkInfo)(parkp)

			// The worker G is no longer running, so it's
			// now safe to allow preemption.
			releasem(park.m.ptr())

			// If the worker isn't attached to its P,
			// attach now. During initialization and after
			// a phase change, the worker may have been
			// running on a different P. As soon as we
			// attach, the owner P may schedule the
			// worker, so this must be done after the G is
			// stopped.
			if park.attach != 0 {
				p := park.attach.ptr()
				park.attach.set(nil)
				// cas the worker because we may be
				// racing with a new worker starting
				// on this P.
				if !p.gcBgMarkWorker.cas(0, guintptr(unsafe.Pointer(g))) {
					// The P got a new worker.
					// Exit this worker.
					return false
				}
			}
			return true
		}, unsafe.Pointer(park), "GC worker (idle)", traceEvGoBlock, 0)

		// Loop until the P dies and disassociates this
		// worker (the P may later be reused, in which case
		// it will get a new worker) or we failed to associate.
		if _p_.gcBgMarkWorker.ptr() != gp {
			break
		}

		// Disable preemption so we can use the gcw. If the
		// scheduler wants to preempt us, we'll stop draining,
		// dispose the gcw, and then preempt.
		park.m.set(acquirem())

		if gcBlackenEnabled == 0 {
			throw("gcBgMarkWorker: blackening not enabled")
		}

		startTime := nanotime()

		decnwait := atomic.Xadd(&work.nwait, -1)
		if decnwait == work.nproc {
			println("runtime: work.nwait=", decnwait, "work.nproc=", work.nproc)
			throw("work.nwait was > work.nproc")
		}

		systemstack(func() {
			// Mark our goroutine preemptible so its stack
			// can be scanned. This lets two mark workers
			// scan each other (otherwise, they would
			// deadlock). We must not modify anything on
			// the G stack. However, stack shrinking is
			// disabled for mark workers, so it is safe to
			// read from the G stack.
			casgstatus(gp, _Grunning, _Gwaiting)
			switch _p_.gcMarkWorkerMode {
			default:
				throw("gcBgMarkWorker: unexpected gcMarkWorkerMode")
			case gcMarkWorkerDedicatedMode:
				gcDrain(&_p_.gcw, gcDrainNoBlock|gcDrainFlushBgCredit)
			case gcMarkWorkerFractionalMode:
				gcDrain(&_p_.gcw, gcDrainUntilPreempt|gcDrainFlushBgCredit)
			case gcMarkWorkerIdleMode:
				gcDrain(&_p_.gcw, gcDrainIdle|gcDrainUntilPreempt|gcDrainFlushBgCredit)
			}
			casgstatus(gp, _Gwaiting, _Grunning)
		})

		// If we are nearing the end of mark, dispose
		// of the cache promptly. We must do this
		// before signaling that we're no longer
		// working so that other workers can't observe
		// no workers and no work while we have this
		// cached, and before we compute done.
		if gcBlackenPromptly {
			_p_.gcw.dispose()
		}

		// Account for time.
		duration := nanotime() - startTime
		switch _p_.gcMarkWorkerMode {
		case gcMarkWorkerDedicatedMode:
			atomic.Xaddint64(&gcController.dedicatedMarkTime, duration)
			atomic.Xaddint64(&gcController.dedicatedMarkWorkersNeeded, 1)
		case gcMarkWorkerFractionalMode:
			atomic.Xaddint64(&gcController.fractionalMarkTime, duration)
			atomic.Xaddint64(&gcController.fractionalMarkWorkersNeeded, 1)
		case gcMarkWorkerIdleMode:
			atomic.Xaddint64(&gcController.idleMarkTime, duration)
		}

		// Was this the last worker and did we run out
		// of work?
		incnwait := atomic.Xadd(&work.nwait, +1)
		if incnwait > work.nproc {
			println("runtime: p.gcMarkWorkerMode=", _p_.gcMarkWorkerMode,
				"work.nwait=", incnwait, "work.nproc=", work.nproc)
			throw("work.nwait > work.nproc")
		}

		// If this worker reached a background mark completion
		// point, signal the main GC goroutine.
		if incnwait == work.nproc && !gcMarkWorkAvailable(nil) {
			// Make this G preemptible and disassociate it
			// as the worker for this P so
			// findRunnableGCWorker doesn't try to
			// schedule it.
			_p_.gcBgMarkWorker.set(nil)
			releasem(park.m.ptr())

			gcMarkDone()

			// Disable preemption and prepare to reattach
			// to the P.
			//
			// We may be running on a different P at this
			// point, so we can't reattach until this G is
			// parked.
			park.m.set(acquirem())
			park.attach.set(_p_)
		}
	}
}

// gcMarkWorkAvailable returns true if executing a mark worker
// on p is potentially useful. p may be nil, in which case it only
// checks the global sources of work.
func gcMarkWorkAvailable(p *p) bool {
	if p != nil && !p.gcw.empty() {
		return true
	}
	if atomic.Load64(&work.full) != 0 {
		return true // global work available
	}
	if work.markrootNext < work.markrootJobs {
		return true // root scan work available
	}
	return false
}

// gcMark runs the mark (or, for concurrent GC, mark termination)
// All gcWork caches must be empty.
// STW is in effect at this point.
//TODO go:nowritebarrier
func gcMark(start_time int64) {
	if debug.allocfreetrace > 0 {
		tracegc()
	}

	if gcphase != _GCmarktermination {
		throw("in gcMark expecting to see gcphase as _GCmarktermination")
	}
	work.tstart = start_time

	// Queue root marking jobs.
	gcMarkRootPrepare()

	work.nwait = 0
	work.ndone = 0
	work.nproc = uint32(gcprocs())

	if debug.gcrescanstacks == 0 && work.full == 0 && work.nDataRoots+work.nBSSRoots+work.nSpanRoots+work.nStackRoots+work.nRescanRoots == 0 {
		// There's no work on the work queue and no root jobs
		// that can produce work, so don't bother entering the
		// getfull() barrier.
		//
		// With the hybrid barrier enabled, this will be the
		// situation the vast majority of the time after
		// concurrent mark. However, we still need a fallback
		// for STW GC and because there are some known races
		// that occasionally leave work around for mark
		// termination.
		//
		// We're still hedging our bets here: if we do
		// accidentally produce some work, we'll still process
		// it, just not necessarily in parallel.
		//
		// TODO(austin): When we eliminate
		// debug.gcrescanstacks: fix the races, and remove
		// work draining from mark termination so we don't
		// need the fallback path.
		work.helperDrainBlock = false
	} else {
		work.helperDrainBlock = true
	}

	if trace.enabled {
		traceGCScanStart()
	}

	if work.nproc > 1 {
		noteclear(&work.alldone)
		helpgc(int32(work.nproc))
	}

	gchelperstart()

	gcw := &getg().m.p.ptr().gcw
	if work.helperDrainBlock {
		gcDrain(gcw, gcDrainBlock)
	} else {
		gcDrain(gcw, gcDrainNoBlock)
	}
	gcw.dispose()

	if debug.gccheckmark > 0 {
		// This is expensive when there's a large number of
		// Gs, so only do it if checkmark is also enabled.
		gcMarkRootCheck()
	}
	if work.full != 0 {
		throw("work.full != 0")
	}

	if work.nproc > 1 {
		notesleep(&work.alldone)
	}

	// Record that at least one root marking pass has completed.
	work.markrootDone = true

	// Double-check that all gcWork caches are empty. This should
	// be ensured by mark 2 before we enter mark termination.
	for i := 0; i < int(gomaxprocs); i++ {
		gcw := &allp[i].gcw
		if !gcw.empty() {
			throw("P has cached GC work at end of mark termination")
		}
		if gcw.scanWork != 0 || gcw.bytesMarked != 0 {
			throw("P has unflushed stats at end of mark termination")
		}
	}

	if trace.enabled {
		traceGCScanDone()
	}

	cachestats()

	// Update the marked heap stat.
	memstats.heap_marked = work.bytesMarked

	// Trigger the next GC cycle when the allocated heap has grown
	// by triggerRatio over the marked heap size. Assume that
	// we're in steady state, so the marked heap size is the
	// same now as it was at the beginning of the GC cycle.
	memstats.gc_trigger = uint64(float64(memstats.heap_marked) * (1 + gcController.triggerRatio))
	if memstats.gc_trigger < heapminimum {
		memstats.gc_trigger = heapminimum
	}
	if int64(memstats.gc_trigger) < 0 {
		print("next_gc=", memstats.next_gc, " bytesMarked=", work.bytesMarked, " heap_live=", memstats.heap_live, " initialHeapLive=", work.initialHeapLive, "\n")
		throw("gc_trigger underflow")
	}

	// Update other GC heap size stats. This must happen after
	// cachestats (which flushes local statistics to these) and
	// flushallmcaches (which modifies heap_live).
	memstats.heap_live = work.bytesMarked
	memstats.heap_scan = uint64(gcController.scanWork)

	minTrigger := memstats.heap_live + sweepMinHeapDistance*uint64(gcpercent)/100
	if memstats.gc_trigger < minTrigger {
		// The allocated heap is already past the trigger.
		// This can happen if the triggerRatio is very low and
		// the marked heap is less than the live heap size.
		//
		// Concurrent sweep happens in the heap growth from
		// heap_live to gc_trigger, so bump gc_trigger up to ensure
		// that concurrent sweep has some heap growth in which
		// to perform sweeping before we start the next GC
		// cycle.
		memstats.gc_trigger = minTrigger
	}

	// The next GC cycle should finish before the allocated heap
	// has grown by GOGC/100.
	memstats.next_gc = memstats.heap_marked + memstats.heap_marked*uint64(gcpercent)/100
	if gcpercent < 0 {
		memstats.next_gc = ^uint64(0)
	}
	if memstats.next_gc < memstats.gc_trigger {
		memstats.next_gc = memstats.gc_trigger
	}

	if trace.enabled {
		traceHeapAlloc()
		traceNextGC()
	}
}

func gcSweep(mode gcMode) {
	if gcphase != _GCoff {
		throw("gcSweep being done but phase is not GCoff")
	}

	lock(&mheap_.lock)
	mheap_.sweepgen += 2
	mheap_.sweepdone = 0
	if mheap_.sweepSpans[mheap_.sweepgen/2%2].index != 0 {
		// We should have drained this list during the last
		// sweep phase. We certainly need to start this phase
		// with an empty swept list.
		throw("non-empty swept list")
	}
	unlock(&mheap_.lock)

	if !_ConcurrentSweep || mode == gcForceBlockMode {
		// Special case synchronous sweep.
		// Record that no proportional sweeping has to happen.
		lock(&mheap_.lock)
		mheap_.sweepPagesPerByte = 0
		mheap_.pagesSwept = 0
		unlock(&mheap_.lock)
		// Sweep all spans eagerly.
		for sweepone() != ^uintptr(0) {
			sweep.npausesweep++
		}
		// Do an additional mProf_GC, because all 'free' events are now real as well.
		mProf_GC()
		mProf_GC()
		return
	}

	// Concurrent sweep needs to sweep all of the in-use pages by
	// the time the allocated heap reaches the GC trigger. Compute
	// the ratio of in-use pages to sweep per byte allocated.
	heapDistance := int64(memstats.gc_trigger) - int64(memstats.heap_live)
	// Add a little margin so rounding errors and concurrent
	// sweep are less likely to leave pages unswept when GC starts.
	heapDistance -= 1024 * 1024
	if heapDistance < _PageSize {
		// Avoid setting the sweep ratio extremely high
		heapDistance = _PageSize
	}
	lock(&mheap_.lock)
	mheap_.sweepPagesPerByte = float64(mheap_.pagesInUse) / float64(heapDistance)
	mheap_.pagesSwept = 0
	mheap_.spanBytesAlloc = 0
	unlock(&mheap_.lock)

	// Background sweep.
	lock(&sweep.lock)
	if sweep.parked {
		sweep.parked = false
		ready(sweep.g, 0, true)
	}
	unlock(&sweep.lock)
}

// gcResetMarkState resets global state prior to marking (concurrent
// or STW) and resets the stack scan state of all Gs.
//
// This is safe to do without the world stopped because any Gs created
// during or after this will start out in the reset state.
func gcResetMarkState() {
	// This may be called during a concurrent phase, so make sure
	// allgs doesn't change.
	if !(gcphase == _GCoff || gcphase == _GCmarktermination) {
		// Accessing gcRescan is unsafe.
		throw("bad GC phase")
	}
	lock(&allglock)
	for _, gp := range allgs {
		gp.gcscandone = false  // set to true in gcphasework
		gp.gcscanvalid = false // stack has not been scanned
		gp.gcRescan = -1
		gp.gcAssistBytes = 0
	}
	unlock(&allglock)

	// Clear rescan list.
	work.rescan.list = work.rescan.list[:0]

	work.bytesMarked = 0
	work.initialHeapLive = memstats.heap_live
	work.markrootDone = false
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

	// Clear central sudog cache.
	// Leave per-P caches alone, they have strictly bounded size.
	// Disconnect cached list before dropping it on the floor,
	// so that a dangling ref to one entry does not pin all of them.
	lock(&sched.sudoglock)
	var sg, sgnext *sudog
	for sg = sched.sudogcache; sg != nil; sg = sgnext {
		sgnext = sg.next
		sg.next = nil
	}
	sched.sudogcache = nil
	unlock(&sched.sudoglock)

	// Clear central defer pools.
	// Leave per-P pools alone, they have strictly bounded size.
	lock(&sched.deferlock)
	for i := range sched.deferpool {
		// disconnect cached list before dropping it on the floor,
		// so that a dangling ref to one entry does not pin all of them.
		var d, dlink *_defer
		for d = sched.deferpool[i]; d != nil; d = dlink {
			dlink = d.link
			d.link = nil
		}
		sched.deferpool[i] = nil
	}
	unlock(&sched.deferlock)
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

	// Parallel mark over GC roots and heap
	if gcphase == _GCmarktermination {
		gcw := &_g_.m.p.ptr().gcw
		if work.helperDrainBlock {
			gcDrain(gcw, gcDrainBlock) // blocks in getfull
		} else {
			gcDrain(gcw, gcDrainNoBlock)
		}
		gcw.dispose()
	}

	if trace.enabled {
		traceGCScanDone()
	}

	nproc := work.nproc // work.nproc can change right after we increment work.ndone
	if atomic.Xadd(&work.ndone, +1) == nproc-1 {
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

// itoaDiv formats val/(10**dec) into buf.
func itoaDiv(buf []byte, val uint64, dec int) []byte {
	i := len(buf) - 1
	idec := i - dec
	for val >= 10 || i >= idec {
		buf[i] = byte(val%10 + '0')
		i--
		if i == idec {
			buf[i] = '.'
			i--
		}
		val /= 10
	}
	buf[i] = byte(val + '0')
	return buf[i:]
}

// fmtNSAsMS nicely formats ns nanoseconds as milliseconds.
func fmtNSAsMS(buf []byte, ns uint64) []byte {
	if ns >= 10e6 {
		// Format as whole milliseconds.
		return itoaDiv(buf, ns/1e6, 0)
	}
	// Format two digits of precision, with at most three decimal places.
	x := ns / 1e3
	if x == 0 {
		buf[0] = '0'
		return buf[:1]
	}
	dec := 3
	for x >= 100 {
		x /= 10
		dec--
	}
	return itoaDiv(buf, x, dec)
}

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
//         objects, (which means all reachable objects are marked) set phase = GCoff.
// 13. Wait for all P's to acknowledge phase change.
// 14. Now malloc allocates white (but sweeps spans before use).
//         Write barrier becomes nop.
// 15. GC does background sweeping, see description below.
// 16. When sufficient allocation has taken place replay the sequence starting at 0 above,
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
//     When a stack is scanned, this phase also installs stack barriers to
//     track how much of the stack has been active.
//     This transition enables write barriers because stack barriers
//     assume that writes to higher frames will be tracked by write
//     barriers. Technically this only needs write barriers for writes
//     to stack slots, but we enable write barriers in general.
// GCscan to GCmark
//     In GCmark, work buffers are drained until there are no more
//     pointers to scan.
//     No scanning of objects (making them black) can happen until all
//     Ps have enabled the write barrier, but that already happened in
//     the transition to GCscan.
// GCmark to GCmarktermination
//     The only change here is that we start allocating black so the Ps must acknowledge
//     the change before we begin the termination algorithm
// GCmarktermination to GSsweep
//     Object currently on the freelist must be marked black for this to work.
//     Are things on the free lists black or white? How does the sweep phase work?

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
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		datap.gcdatamask = progToPointerMask((*byte)(unsafe.Pointer(datap.gcdata)), datap.edata-datap.data)
		datap.gcbssmask = progToPointerMask((*byte)(unsafe.Pointer(datap.gcbss)), datap.ebss-datap.bss)
	}
	memstats.next_gc = heapminimum
	work.startSema = 1
	work.markDoneSema = 1
}

func readgogc() int32 {
	p := gogetenv("GOGC")
	if p == "" {
		return 100
	}
	if p == "off" {
		return -1
	}
	return int32(atoi(p))
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
	unlock(&mheap_.lock)
	return out
}

// Garbage collector phase.
// Indicates to write barrier and sychronization task to preform.
var gcphase uint32

// The compiler knows about this variable.
// If you change it, you must change the compiler too.
var writeBarrier struct {
	enabled bool // compiler emits a check of this before calling write barrier
	needed  bool // whether we need a write barrier for current GC phase
	cgo     bool // whether we need a write barrier for a cgo check
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
	_GCmark                   // GC marking roots and workbufs, write barrier ENABLED
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

// gcController implements the GC pacing controller that determines
// when to trigger concurrent garbage collection and how much marking
// work to do in mutator assists and background marking.
//
// It uses a feedback control algorithm to adjust the memstats.next_gc
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
	// throughout the cycle.
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

	// bgMarkStartTime is the absolute start time in nanoseconds
	// that the background mark phase started.
	bgMarkStartTime int64

	// assistTime is the absolute start time in nanoseconds that
	// mutator assists were enabled.
	assistStartTime int64

	// heapGoal is the goal memstats.heap_live for when this cycle
	// ends. This is computed at the beginning of each cycle.
	heapGoal uint64

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
	// the heap size marked by the previous cycle. This is updated
	// at the end of of each cycle.
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
	// small heap, fake heap_marked so it looks like next_gc is
	// the appropriate growth from heap_marked, even though the
	// real heap_marked may not have a meaningful value (on the
	// first cycle) or may be much smaller (resulting in a large
	// error response).
	if memstats.next_gc <= heapminimum {
		memstats.heap_marked = uint64(float64(memstats.next_gc) / (1 + c.triggerRatio))
		memstats.heap_reachable = memstats.heap_marked
	}

	// Compute the heap goal for this cycle
	c.heapGoal = memstats.heap_reachable + memstats.heap_reachable*uint64(gcpercent)/100

	// Ensure that the heap goal is at least a little larger than
	// the current live heap size. This may not be the case if GC
	// start is delayed or if the allocation that pushed heap_live
	// over next_gc is large or if the trigger is really close to
	// GOGC. Assist is proportional to this distance, so enforce a
	// minimum distance, even if it means going over the GOGC goal
	// by a tiny bit.
	if c.heapGoal < memstats.heap_live+1024*1024 {
		c.heapGoal = memstats.heap_live + 1024*1024
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
			c.heapGoal>>20, " MB)",
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
func (c *gcControllerState) revise() {
	// Compute the expected scan work remaining.
	//
	// Note that the scannable heap size is likely to increase
	// during the GC cycle. This is why it's important to revise
	// the assist ratio throughout the cycle: if the scannable
	// heap size increases, the assist ratio based on the initial
	// scannable heap size may target too little scan work.
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
	heapDistance := int64(c.heapGoal) - int64(memstats.heap_live)
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
	//
	// TODO(austin): next_gc is based on heap_reachable, not
	// heap_marked, which means the actual growth ratio
	// technically isn't comparable to the trigger ratio.
	goalGrowthRatio := float64(gcpercent) / 100
	actualGrowthRatio := float64(memstats.heap_live)/float64(memstats.heap_marked) - 1
	assistDuration := nanotime() - c.assistStartTime

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
		H_T := memstats.next_gc
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
		id := int32(fastrand1() % uint32(gomaxprocs-1))
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
	if _p_.gcBgMarkWorker == nil {
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
		now := nanotime() - gcController.bgMarkStartTime
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
	gp := _p_.gcBgMarkWorker
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

// gcOverAssistBytes determines how many extra allocation bytes of
// assist credit a GC assist builds up when an assist happens. This
// amortizes the cost of an assist by pre-paying for this many bytes
// of future allocations.
const gcOverAssistBytes = 1 << 20

var work struct {
	full  uint64                   // lock-free list of full blocks workbuf
	empty uint64                   // lock-free list of empty blocks workbuf
	pad0  [sys.CacheLineSize]uint8 // prevents false-sharing between full/empty and nproc/nwait

	markrootNext uint32 // next markroot job
	markrootJobs uint32 // number of markroot jobs

	nproc   uint32
	tstart  int64
	nwait   uint32
	ndone   uint32
	alldone note

	// Number of roots of various root types. Set by gcMarkRootPrepare.
	nDataRoots, nBSSRoots, nSpanRoots, nStackRoots int

	// finalizersDone indicates that finalizers and objects with
	// finalizers have been scanned by markroot. During concurrent
	// GC, this happens during the concurrent scan phase. During
	// STW GC, this happens during mark termination.
	finalizersDone bool

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

	// Copy of mheap.allspans for marker or sweeper.
	spans []*mspan

	// totaltime is the CPU nanoseconds spent in GC since the
	// program started if debug.gctrace > 0.
	totaltime int64

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
	bytesMarked uint64

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
	gcForceBlockMode               // stop-the-world GC now and STW sweep
)

// gcShouldStart returns true if the exit condition for the _GCoff
// phase has been met. The exit condition should be tested when
// allocating.
//
// If forceTrigger is true, it ignores the current heap size, but
// checks all other conditions. In general this should be false.
func gcShouldStart(forceTrigger bool) bool {
	return gcphase == _GCoff && (forceTrigger || memstats.heap_live >= memstats.next_gc) && memstats.enablegc && panicking == 0 && gcpercent >= 0
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
		semacquire(&work.startSema, false)
		// Re-check transition condition under transition lock.
		if !gcShouldStart(forceTrigger) {
			semrelease(&work.startSema)
			return
		}
	}

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
	semacquire(&worldsema, false)

	if trace.enabled {
		traceGCStart()
	}

	if mode == gcBackgroundMode {
		gcBgMarkStartWorkers()
	}
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
		finishsweep_m(true)
	})
	// clearpools before we start the GC. If we wait they memory will not be
	// reclaimed until the next GC cycle.
	clearpools()

	gcResetMarkState()

	work.finalizersDone = false

	if mode == gcBackgroundMode { // Do as much work concurrently as possible
		gcController.startCycle()
		work.heapGoal = gcController.heapGoal

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

		// markrootSpans uses work.spans, so make sure
		// it is up to date.
		gcCopySpans()

		gcBgMarkPrepare() // Must happen before assist enable.
		gcMarkRootPrepare()

		// At this point all Ps have enabled the write
		// barrier, thus maintaining the no white to
		// black invariant. Enable mutator assists to
		// put back-pressure on fast allocating
		// mutators.
		atomic.Store(&gcBlackenEnabled, 1)

		// Assists and workers can start the moment we start
		// the world.
		gcController.assistStartTime = now
		gcController.bgMarkStartTime = now

		// Concurrent mark.
		systemstack(startTheWorldWithSema)
		now = nanotime()
		work.pauseNS += now - work.pauseStart
		work.tMark = now
	} else {
		t := nanotime()
		work.tMark, work.tMarkTerm = t, t
		work.heapGoal = work.heap0

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
	semacquire(&work.markDoneSema, false)

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
		// sitting in the per-P work caches and there can be more
		// objects reachable from global roots since they don't have write
		// barriers. Rescan some roots and flush work caches.

		gcMarkRootCheck()

		// Disallow caching workbufs and indicate that we're in mark 2.
		gcBlackenPromptly = true

		// Prevent completion of mark 2 until we've flushed
		// cached workbufs.
		atomic.Xadd(&work.nwait, -1)

		// Rescan global data and BSS. There may still work
		// workers running at this point, so bump "jobs" down
		// before "next" so they won't try running root jobs
		// until we set next.
		atomic.Store(&work.markrootJobs, uint32(fixedRootCount+work.nDataRoots+work.nBSSRoots))
		atomic.Store(&work.markrootNext, fixedRootCount)

		// GC is set up for mark 2. Let Gs blocked on the
		// transition lock go while we flush caches.
		semrelease(&work.markDoneSema)

		systemstack(func() {
			// Flush all currently cached workbufs and
			// ensure all Ps see gcBlackenPromptly. This
			// also blocks until any remaining mark 1
			// workers have exited their loop so we can
			// start new mark 2 workers that will observe
			// the new root marking jobs.
			forEachP(func(_p_ *p) {
				_p_.gcw.dispose()
			})
		})

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

		// markroot is done now, so record that objects with
		// finalizers have been scanned.
		work.finalizersDone = true

		// Disable assists and background workers. We must do
		// this before waking blocked assists.
		atomic.Store(&gcBlackenEnabled, 0)

		// Flush the gcWork caches. This must be done before
		// endCycle since endCycle depends on statistics kept
		// in these caches.
		gcFlushGCWork()

		// Wake all blocked assists. These will run when we
		// start the world again.
		gcWakeAllAssists()

		// Likewise, release the transition lock. Blocked
		// workers and assists will run when we start the
		// world again.
		semrelease(&work.markDoneSema)

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

	// Run gc on the g0 stack.  We do this so that the g stack
	// we're currently running on will no longer change.  Cuts
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
			finishsweep_m(true)

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

	// Free stack spans. This must be done between GC cycles.
	systemstack(freeStackSpans)

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
		if p.gcBgMarkWorker == nil {
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

func gcBgMarkWorker(p *p) {
	// Register this G as the background mark worker for p.
	casgp := func(gpp **g, old, new *g) bool {
		return casp((*unsafe.Pointer)(unsafe.Pointer(gpp)), unsafe.Pointer(old), unsafe.Pointer(new))
	}

	gp := getg()
	mp := acquirem()
	owned := casgp(&p.gcBgMarkWorker, nil, gp)
	// After this point, the background mark worker is scheduled
	// cooperatively by gcController.findRunnable. Hence, it must
	// never be preempted, as this would put it into _Grunnable
	// and put it on a run queue. Instead, when the preempt flag
	// is set, this puts itself into _Gwaiting to be woken up by
	// gcController.findRunnable at the appropriate time.
	notewakeup(&work.bgMarkReady)
	if !owned {
		// A sleeping worker came back and reassociated with
		// the P. That's fine.
		releasem(mp)
		return
	}

	for {
		// Go to sleep until woken by gcContoller.findRunnable.
		// We can't releasem yet since even the call to gopark
		// may be preempted.
		gopark(func(g *g, mp unsafe.Pointer) bool {
			releasem((*m)(mp))
			return true
		}, unsafe.Pointer(mp), "GC worker (idle)", traceEvGoBlock, 0)

		// Loop until the P dies and disassociates this
		// worker. (The P may later be reused, in which case
		// it will get a new worker.)
		if p.gcBgMarkWorker != gp {
			break
		}

		// Disable preemption so we can use the gcw. If the
		// scheduler wants to preempt us, we'll stop draining,
		// dispose the gcw, and then preempt.
		mp = acquirem()

		if gcBlackenEnabled == 0 {
			throw("gcBgMarkWorker: blackening not enabled")
		}

		startTime := nanotime()

		decnwait := atomic.Xadd(&work.nwait, -1)
		if decnwait == work.nproc {
			println("runtime: work.nwait=", decnwait, "work.nproc=", work.nproc)
			throw("work.nwait was > work.nproc")
		}

		switch p.gcMarkWorkerMode {
		default:
			throw("gcBgMarkWorker: unexpected gcMarkWorkerMode")
		case gcMarkWorkerDedicatedMode:
			gcDrain(&p.gcw, gcDrainNoBlock|gcDrainFlushBgCredit)
		case gcMarkWorkerFractionalMode, gcMarkWorkerIdleMode:
			gcDrain(&p.gcw, gcDrainUntilPreempt|gcDrainFlushBgCredit)
		}

		// If we are nearing the end of mark, dispose
		// of the cache promptly. We must do this
		// before signaling that we're no longer
		// working so that other workers can't observe
		// no workers and no work while we have this
		// cached, and before we compute done.
		if gcBlackenPromptly {
			p.gcw.dispose()
		}

		// Account for time.
		duration := nanotime() - startTime
		switch p.gcMarkWorkerMode {
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
			println("runtime: p.gcMarkWorkerMode=", p.gcMarkWorkerMode,
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
			p.gcBgMarkWorker = nil
			releasem(mp)

			gcMarkDone()

			// Disable preemption and reassociate with the P.
			//
			// We may be running on a different P at this
			// point, so this has to be done carefully.
			mp = acquirem()
			if !casgp(&p.gcBgMarkWorker, nil, gp) {
				// The P got a new worker.
				releasem(mp)
				break
			}
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

// gcFlushGCWork disposes the gcWork caches of all Ps. The world must
// be stopped.
//go:nowritebarrier
func gcFlushGCWork() {
	// Gather all cached GC work. All other Ps are stopped, so
	// it's safe to manipulate their GC work caches.
	for i := 0; i < int(gomaxprocs); i++ {
		allp[i].gcw.dispose()
	}
}

// gcMark runs the mark (or, for concurrent GC, mark termination)
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

	gcCopySpans() // TODO(rlh): should this be hoisted and done only once? Right now it is done for normal marking and also for checkmarking.

	// Make sure the per-P gcWork caches are empty. During mark
	// termination, these caches can still be used temporarily,
	// but must be disposed to the global lists immediately.
	gcFlushGCWork()

	// Queue root marking jobs.
	gcMarkRootPrepare()

	work.nwait = 0
	work.ndone = 0
	work.nproc = uint32(gcprocs())

	if trace.enabled {
		traceGCScanStart()
	}

	if work.nproc > 1 {
		noteclear(&work.alldone)
		helpgc(int32(work.nproc))
	}

	gchelperstart()

	var gcw gcWork
	gcDrain(&gcw, gcDrainBlock)
	gcw.dispose()

	gcMarkRootCheck()
	if work.full != 0 {
		throw("work.full != 0")
	}

	if work.nproc > 1 {
		notesleep(&work.alldone)
	}

	// markroot is done now, so record that objects with
	// finalizers have been scanned.
	work.finalizersDone = true

	for i := 0; i < int(gomaxprocs); i++ {
		if !allp[i].gcw.empty() {
			throw("P has cached GC work at end of mark termination")
		}
	}

	if trace.enabled {
		traceGCScanDone()
	}

	cachestats()

	// Compute the reachable heap size at the beginning of the
	// cycle. This is approximately the marked heap size at the
	// end (which we know) minus the amount of marked heap that
	// was allocated after marking began (which we don't know, but
	// is approximately the amount of heap that was allocated
	// since marking began).
	allocatedDuringCycle := memstats.heap_live - work.initialHeapLive
	if memstats.heap_live < work.initialHeapLive {
		// This can happen if mCentral_UncacheSpan tightens
		// the heap_live approximation.
		allocatedDuringCycle = 0
	}
	if work.bytesMarked >= allocatedDuringCycle {
		memstats.heap_reachable = work.bytesMarked - allocatedDuringCycle
	} else {
		// This can happen if most of the allocation during
		// the cycle never became reachable from the heap.
		// Just set the reachable heap approximation to 0 and
		// let the heapminimum kick in below.
		memstats.heap_reachable = 0
	}

	// Trigger the next GC cycle when the allocated heap has grown
	// by triggerRatio over the reachable heap size. Assume that
	// we're in steady state, so the reachable heap size is the
	// same now as it was at the beginning of the GC cycle.
	memstats.next_gc = uint64(float64(memstats.heap_reachable) * (1 + gcController.triggerRatio))
	if memstats.next_gc < heapminimum {
		memstats.next_gc = heapminimum
	}
	if int64(memstats.next_gc) < 0 {
		print("next_gc=", memstats.next_gc, " bytesMarked=", work.bytesMarked, " heap_live=", memstats.heap_live, " initialHeapLive=", work.initialHeapLive, "\n")
		throw("next_gc underflow")
	}

	// Update other GC heap size stats. This must happen after
	// cachestats (which flushes local statistics to these) and
	// flushallmcaches (which modifies heap_live).
	memstats.heap_live = work.bytesMarked
	memstats.heap_marked = work.bytesMarked
	memstats.heap_scan = uint64(gcController.scanWork)

	minNextGC := memstats.heap_live + sweepMinHeapDistance*uint64(gcpercent)/100
	if memstats.next_gc < minNextGC {
		// The allocated heap is already past the trigger.
		// This can happen if the triggerRatio is very low and
		// the reachable heap estimate is less than the live
		// heap size.
		//
		// Concurrent sweep happens in the heap growth from
		// heap_live to next_gc, so bump next_gc up to ensure
		// that concurrent sweep has some heap growth in which
		// to perform sweeping before we start the next GC
		// cycle.
		memstats.next_gc = minNextGC
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
	gcCopySpans()

	lock(&mheap_.lock)
	mheap_.sweepgen += 2
	mheap_.sweepdone = 0
	sweep.spanidx = 0
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
	heapDistance := int64(memstats.next_gc) - int64(memstats.heap_live)
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
		ready(sweep.g, 0)
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

// gcResetMarkState resets global state prior to marking (concurrent
// or STW) and resets the stack scan state of all Gs. Any Gs created
// after this will also be in the reset state.
func gcResetMarkState() {
	// This may be called during a concurrent phase, so make sure
	// allgs doesn't change.
	lock(&allglock)
	for _, gp := range allgs {
		gp.gcscandone = false  // set to true in gcphasework
		gp.gcscanvalid = false // stack has not been scanned
		gp.gcAssistBytes = 0
	}
	unlock(&allglock)

	work.bytesMarked = 0
	work.initialHeapLive = memstats.heap_live
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
		var gcw gcWork
		gcDrain(&gcw, gcDrainBlock) // blocks in getfull
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

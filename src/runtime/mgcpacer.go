// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
	"internal/goexperiment"
	"internal/runtime/atomic"
	"internal/runtime/math"
	"internal/strconv"
	_ "unsafe" // for go:linkname
)

const (
	// gcGoalUtilization is the goal CPU utilization for
	// marking as a fraction of GOMAXPROCS.
	//
	// Increasing the goal utilization will shorten GC cycles as the GC
	// has more resources behind it, lessening costs from the write barrier,
	// but comes at the cost of increasing mutator latency.
	gcGoalUtilization = gcBackgroundUtilization

	// gcBackgroundUtilization is the fixed CPU utilization for background
	// marking. It must be <= gcGoalUtilization. The difference between
	// gcGoalUtilization and gcBackgroundUtilization will be made up by
	// mark assists. The scheduler will aim to use within 50% of this
	// goal.
	//
	// As a general rule, there's little reason to set gcBackgroundUtilization
	// < gcGoalUtilization. One reason might be in mostly idle applications,
	// where goroutines are unlikely to assist at all, so the actual
	// utilization will be lower than the goal. But this is moot point
	// because the idle mark workers already soak up idle CPU resources.
	// These two values are still kept separate however because they are
	// distinct conceptually, and in previous iterations of the pacer the
	// distinction was more important.
	gcBackgroundUtilization = 0.25

	// gcCreditSlack is the amount of scan work credit that can
	// accumulate locally before updating gcController.heapScanWork and,
	// optionally, gcController.bgScanCredit. Lower values give a more
	// accurate assist ratio and make it more likely that assists will
	// successfully steal background credit. Higher values reduce memory
	// contention.
	gcCreditSlack = 2000

	// gcAssistTimeSlack is the nanoseconds of mutator assist time that
	// can accumulate on a P before updating gcController.assistTime.
	gcAssistTimeSlack = 5000

	// gcOverAssistWork determines how many extra units of scan work a GC
	// assist does when an assist happens. This amortizes the cost of an
	// assist by pre-paying for this many bytes of future allocations.
	gcOverAssistWork = 64 << 10

	// defaultHeapMinimum is the value of heapMinimum for GOGC==100.
	defaultHeapMinimum = (goexperiment.HeapMinimum512KiBInt)*(512<<10) +
		(1-goexperiment.HeapMinimum512KiBInt)*(4<<20)

	// maxStackScanSlack is the bytes of stack space allocated or freed
	// that can accumulate on a P before updating gcController.stackSize.
	maxStackScanSlack = 8 << 10

	// memoryLimitMinHeapGoalHeadroom is the minimum amount of headroom the
	// pacer gives to the heap goal when operating in the memory-limited regime.
	// That is, it'll reduce the heap goal by this many extra bytes off of the
	// base calculation, at minimum.
	memoryLimitMinHeapGoalHeadroom = 1 << 20

	// memoryLimitHeapGoalHeadroomPercent is how headroom the memory-limit-based
	// heap goal should have as a percent of the maximum possible heap goal allowed
	// to maintain the memory limit.
	memoryLimitHeapGoalHeadroomPercent = 3
)

// gcController implements the GC pacing controller that determines
// when to trigger concurrent garbage collection and how much marking
// work to do in mutator assists and background marking.
//
// It calculates the ratio between the allocation rate (in terms of CPU
// time) and the GC scan throughput to determine the heap size at which to
// trigger a GC cycle such that no GC assists are required to finish on time.
// This algorithm thus optimizes GC CPU utilization to the dedicated background
// mark utilization of 25% of GOMAXPROCS by minimizing GC assists.
// GOMAXPROCS. The high-level design of this algorithm is documented
// at https://github.com/golang/proposal/blob/master/design/44167-gc-pacer-redesign.md.
// See https://golang.org/s/go15gcpacing for additional historical context.
var gcController gcControllerState

type gcControllerState struct {
	// Initialized from GOGC. GOGC=off means no GC.
	gcPercent atomic.Int32

	// memoryLimit is the soft memory limit in bytes.
	//
	// Initialized from GOMEMLIMIT. GOMEMLIMIT=off is equivalent to MaxInt64
	// which means no soft memory limit in practice.
	//
	// This is an int64 instead of a uint64 to more easily maintain parity with
	// the SetMemoryLimit API, which sets a maximum at MaxInt64. This value
	// should never be negative.
	memoryLimit atomic.Int64

	// heapMinimum is the minimum heap size at which to trigger GC.
	// For small heaps, this overrides the usual GOGC*live set rule.
	//
	// When there is a very small live set but a lot of allocation, simply
	// collecting when the heap reaches GOGC*live results in many GC
	// cycles and high total per-GC overhead. This minimum amortizes this
	// per-GC overhead while keeping the heap reasonably small.
	//
	// During initialization this is set to 4MB*GOGC/100. In the case of
	// GOGC==0, this will set heapMinimum to 0, resulting in constant
	// collection even when the heap size is small, which is useful for
	// debugging.
	heapMinimum uint64

	// runway is the amount of runway in heap bytes allocated by the
	// application that we want to give the GC once it starts.
	//
	// This is computed from consMark during mark termination.
	runway atomic.Uint64

	// consMark is the estimated per-CPU consMark ratio for the application.
	//
	// It represents the ratio between the application's allocation
	// rate, as bytes allocated per CPU-time, and the GC's scan rate,
	// as bytes scanned per CPU-time.
	// The units of this ratio are (B / cpu-ns) / (B / cpu-ns).
	//
	// At a high level, this value is computed as the bytes of memory
	// allocated (cons) per unit of scan work completed (mark) in a GC
	// cycle, divided by the CPU time spent on each activity.
	//
	// Updated at the end of each GC cycle, in endCycle.
	consMark float64

	// lastConsMark is the computed cons/mark value for the previous 4 GC
	// cycles. Note that this is *not* the last value of consMark, but the
	// measured cons/mark value in endCycle.
	lastConsMark [4]float64

	// gcPercentHeapGoal is the goal heapLive for when next GC ends derived
	// from gcPercent.
	//
	// Set to ^uint64(0) if gcPercent is disabled.
	gcPercentHeapGoal atomic.Uint64

	// sweepDistMinTrigger is the minimum trigger to ensure a minimum
	// sweep distance.
	//
	// This bound is also special because it applies to both the trigger
	// *and* the goal (all other trigger bounds must be based *on* the goal).
	//
	// It is computed ahead of time, at commit time. The theory is that,
	// absent a sudden change to a parameter like gcPercent, the trigger
	// will be chosen to always give the sweeper enough headroom. However,
	// such a change might dramatically and suddenly move up the trigger,
	// in which case we need to ensure the sweeper still has enough headroom.
	sweepDistMinTrigger atomic.Uint64

	// triggered is the point at which the current GC cycle actually triggered.
	// Only valid during the mark phase of a GC cycle, otherwise set to ^uint64(0).
	//
	// Updated while the world is stopped.
	triggered uint64

	// lastHeapGoal is the value of heapGoal at the moment the last GC
	// ended. Note that this is distinct from the last value heapGoal had,
	// because it could change if e.g. gcPercent changes.
	//
	// Read and written with the world stopped or with mheap_.lock held.
	lastHeapGoal uint64

	// heapLive is the number of bytes considered live by the GC.
	// That is: retained by the most recent GC plus allocated
	// since then. heapLive ≤ memstats.totalAlloc-memstats.totalFree, since
	// heapAlloc includes unmarked objects that have not yet been swept (and
	// hence goes up as we allocate and down as we sweep) while heapLive
	// excludes these objects (and hence only goes up between GCs).
	//
	// To reduce contention, this is updated only when obtaining a span
	// from an mcentral and at this point it counts all of the unallocated
	// slots in that span (which will be allocated before that mcache
	// obtains another span from that mcentral). Hence, it slightly
	// overestimates the "true" live heap size. It's better to overestimate
	// than to underestimate because 1) this triggers the GC earlier than
	// necessary rather than potentially too late and 2) this leads to a
	// conservative GC rate rather than a GC rate that is potentially too
	// low.
	//
	// Whenever this is updated, call traceHeapAlloc() and
	// this gcControllerState's revise() method.
	heapLive atomic.Uint64

	// heapScan is the number of bytes of "scannable" heap. This is the
	// live heap (as counted by heapLive), but omitting no-scan objects and
	// no-scan tails of objects.
	//
	// This value is fixed at the start of a GC cycle. It represents the
	// maximum scannable heap.
	heapScan atomic.Uint64

	// lastHeapScan is the number of bytes of heap that were scanned
	// last GC cycle. It is the same as heapMarked, but only
	// includes the "scannable" parts of objects.
	//
	// Updated when the world is stopped.
	lastHeapScan uint64

	// lastStackScan is the number of bytes of stack that were scanned
	// last GC cycle.
	lastStackScan atomic.Uint64

	// maxStackScan is the amount of allocated goroutine stack space in
	// use by goroutines.
	//
	// This number tracks allocated goroutine stack space rather than used
	// goroutine stack space (i.e. what is actually scanned) because used
	// goroutine stack space is much harder to measure cheaply. By using
	// allocated space, we make an overestimate; this is OK, it's better
	// to conservatively overcount than undercount.
	maxStackScan atomic.Uint64

	// globalsScan is the total amount of global variable space
	// that is scannable.
	globalsScan atomic.Uint64

	// heapMarked is the number of bytes marked by the previous
	// GC. After mark termination, heapLive == heapMarked, but
	// unlike heapLive, heapMarked does not change until the
	// next mark termination.
	heapMarked uint64

	// heapScanWork is the total heap scan work performed this cycle.
	// stackScanWork is the total stack scan work performed this cycle.
	// globalsScanWork is the total globals scan work performed this cycle.
	//
	// These are updated atomically during the cycle. Updates occur in
	// bounded batches, since they are both written and read
	// throughout the cycle. At the end of the cycle, heapScanWork is how
	// much of the retained heap is scannable.
	//
	// Currently these are measured in bytes. For most uses, this is an
	// opaque unit of work, but for estimation the definition is important.
	//
	// Note that stackScanWork includes only stack space scanned, not all
	// of the allocated stack.
	heapScanWork    atomic.Int64
	stackScanWork   atomic.Int64
	globalsScanWork atomic.Int64

	// bgScanCredit is the scan work credit accumulated by the concurrent
	// background scan. This credit is accumulated by the background scan
	// and stolen by mutator assists.  Updates occur in bounded batches,
	// since it is both written and read throughout the cycle.
	bgScanCredit atomic.Int64

	// assistTime is the nanoseconds spent in mutator assists
	// during this cycle. This is updated atomically, and must also
	// be updated atomically even during a STW, because it is read
	// by sysmon. Updates occur in bounded batches, since it is both
	// written and read throughout the cycle.
	assistTime atomic.Int64

	// dedicatedMarkTime is the nanoseconds spent in dedicated mark workers
	// during this cycle. This is updated at the end of the concurrent mark
	// phase.
	dedicatedMarkTime atomic.Int64

	// fractionalMarkTime is the nanoseconds spent in the fractional mark
	// worker during this cycle. This is updated throughout the cycle and
	// will be up-to-date if the fractional mark worker is not currently
	// running.
	fractionalMarkTime atomic.Int64

	// idleMarkTime is the nanoseconds spent in idle marking during this
	// cycle. This is updated throughout the cycle.
	idleMarkTime atomic.Int64

	// markStartTime is the absolute start time in nanoseconds
	// that assists and background mark workers started.
	markStartTime int64

	// dedicatedMarkWorkersNeeded is the number of dedicated mark workers
	// that need to be started. This is computed at the beginning of each
	// cycle and decremented as dedicated mark workers get started.
	dedicatedMarkWorkersNeeded atomic.Int64

	// idleMarkWorkers is two packed int32 values in a single uint64.
	// These two values are always updated simultaneously.
	//
	// The bottom int32 is the current number of idle mark workers executing.
	//
	// The top int32 is the maximum number of idle mark workers allowed to
	// execute concurrently. Normally, this number is just gomaxprocs. However,
	// during periodic GC cycles it is set to 0 because the system is idle
	// anyway; there's no need to go full blast on all of GOMAXPROCS.
	//
	// The maximum number of idle mark workers is used to prevent new workers
	// from starting, but it is not a hard maximum. It is possible (but
	// exceedingly rare) for the current number of idle mark workers to
	// transiently exceed the maximum. This could happen if the maximum changes
	// just after a GC ends, and an M with no P.
	//
	// Note that if we have no dedicated mark workers, we set this value to
	// 1 in this case we only have fractional GC workers which aren't scheduled
	// strictly enough to ensure GC progress. As a result, idle-priority mark
	// workers are vital to GC progress in these situations.
	//
	// For example, consider a situation in which goroutines block on the GC
	// (such as via runtime.GOMAXPROCS) and only fractional mark workers are
	// scheduled (e.g. GOMAXPROCS=1). Without idle-priority mark workers, the
	// last running M might skip scheduling a fractional mark worker if its
	// utilization goal is met, such that once it goes to sleep (because there's
	// nothing to do), there will be nothing else to spin up a new M for the
	// fractional worker in the future, stalling GC progress and causing a
	// deadlock. However, idle-priority workers will *always* run when there is
	// nothing left to do, ensuring the GC makes progress.
	//
	// See github.com/golang/go/issues/44163 for more details.
	idleMarkWorkers atomic.Uint64

	// assistWorkPerByte is the ratio of scan work to allocated
	// bytes that should be performed by mutator assists. This is
	// computed at the beginning of each cycle and updated every
	// time heapScan is updated.
	assistWorkPerByte atomic.Float64

	// assistBytesPerWork is 1/assistWorkPerByte.
	//
	// Note that because this is read and written independently
	// from assistWorkPerByte users may notice a skew between
	// the two values, and such a state should be safe.
	assistBytesPerWork atomic.Float64

	// fractionalUtilizationGoal is the fraction of wall clock
	// time that should be spent in the fractional mark worker on
	// each P that isn't running a dedicated worker.
	//
	// For example, if the utilization goal is 25% and there are
	// no dedicated workers, this will be 0.25. If the goal is
	// 25%, there is one dedicated worker, and GOMAXPROCS is 5,
	// this will be 0.05 to make up the missing 5%.
	//
	// If this is zero, no fractional workers are needed.
	fractionalUtilizationGoal float64

	// These memory stats are effectively duplicates of fields from
	// memstats.heapStats but are updated atomically or with the world
	// stopped and don't provide the same consistency guarantees.
	//
	// Because the runtime is responsible for managing a memory limit, it's
	// useful to couple these stats more tightly to the gcController, which
	// is intimately connected to how that memory limit is maintained.
	heapInUse    sysMemStat    // bytes in mSpanInUse spans
	heapReleased sysMemStat    // bytes released to the OS
	heapFree     sysMemStat    // bytes not in any span, but not released to the OS
	totalAlloc   atomic.Uint64 // total bytes allocated
	totalFree    atomic.Uint64 // total bytes freed
	mappedReady  atomic.Uint64 // total virtual memory in the Ready state (see mem.go).

	// test indicates that this is a test-only copy of gcControllerState.
	test bool

	_ cpu.CacheLinePad
}

func (c *gcControllerState) init(gcPercent int32, memoryLimit int64) {
	c.heapMinimum = defaultHeapMinimum
	c.triggered = ^uint64(0)
	c.setGCPercent(gcPercent)
	c.setMemoryLimit(memoryLimit)
	c.commit(true) // No sweep phase in the first GC cycle.
	// N.B. Don't bother calling traceHeapGoal. Tracing is never enabled at
	// initialization time.
	// N.B. No need to call revise; there's no GC enabled during
	// initialization.
}

// startCycle resets the GC controller's state and computes estimates
// for a new GC cycle. The caller must hold worldsema and the world
// must be stopped.
func (c *gcControllerState) startCycle(markStartTime int64, procs int, trigger gcTrigger) {
	c.heapScanWork.Store(0)
	c.stackScanWork.Store(0)
	c.globalsScanWork.Store(0)
	c.bgScanCredit.Store(0)
	c.assistTime.Store(0)
	c.dedicatedMarkTime.Store(0)
	c.fractionalMarkTime.Store(0)
	c.idleMarkTime.Store(0)
	c.markStartTime = markStartTime
	c.triggered = c.heapLive.Load()

	// Compute the background mark utilization goal. In general,
	// this may not come out exactly. We round the number of
	// dedicated workers so that the utilization is closest to
	// 25%. For small GOMAXPROCS, this would introduce too much
	// error, so we add fractional workers in that case.
	totalUtilizationGoal := float64(procs) * gcBackgroundUtilization
	dedicatedMarkWorkersNeeded := int64(totalUtilizationGoal + 0.5)
	utilError := float64(dedicatedMarkWorkersNeeded)/totalUtilizationGoal - 1
	const maxUtilError = 0.3
	if utilError < -maxUtilError || utilError > maxUtilError {
		// Rounding put us more than 30% off our goal. With
		// gcBackgroundUtilization of 25%, this happens for
		// GOMAXPROCS<=3 or GOMAXPROCS=6. Enable fractional
		// workers to compensate.
		if float64(dedicatedMarkWorkersNeeded) > totalUtilizationGoal {
			// Too many dedicated workers.
			dedicatedMarkWorkersNeeded--
		}
		c.fractionalUtilizationGoal = (totalUtilizationGoal - float64(dedicatedMarkWorkersNeeded)) / float64(procs)
	} else {
		c.fractionalUtilizationGoal = 0
	}

	// In STW mode, we just want dedicated workers.
	if debug.gcstoptheworld > 0 {
		dedicatedMarkWorkersNeeded = int64(procs)
		c.fractionalUtilizationGoal = 0
	}

	// Clear per-P state
	for _, p := range allp {
		p.gcAssistTime = 0
		p.gcFractionalMarkTime = 0
	}

	if trigger.kind == gcTriggerTime {
		// During a periodic GC cycle, reduce the number of idle mark workers
		// required. However, we need at least one dedicated mark worker or
		// idle GC worker to ensure GC progress in some scenarios (see comment
		// on maxIdleMarkWorkers).
		if dedicatedMarkWorkersNeeded > 0 {
			c.setMaxIdleMarkWorkers(0)
		} else {
			// TODO(mknyszek): The fundamental reason why we need this is because
			// we can't count on the fractional mark worker to get scheduled.
			// Fix that by ensuring it gets scheduled according to its quota even
			// if the rest of the application is idle.
			c.setMaxIdleMarkWorkers(1)
		}
	} else {
		// N.B. gomaxprocs and dedicatedMarkWorkersNeeded are guaranteed not to
		// change during a GC cycle.
		c.setMaxIdleMarkWorkers(int32(procs) - int32(dedicatedMarkWorkersNeeded))
	}

	// Compute initial values for controls that are updated
	// throughout the cycle.
	c.dedicatedMarkWorkersNeeded.Store(dedicatedMarkWorkersNeeded)
	c.revise()

	if debug.gcpacertrace > 0 {
		heapGoal := c.heapGoal()
		assistRatio := c.assistWorkPerByte.Load()
		print("pacer: assist ratio=", assistRatio,
			" (scan ", gcController.heapScan.Load()>>20, " MB in ",
			work.initialHeapLive>>20, "->",
			heapGoal>>20, " MB)",
			" workers=", dedicatedMarkWorkersNeeded,
			"+", c.fractionalUtilizationGoal, "\n")
	}
}

// revise updates the assist ratio during the GC cycle to account for
// improved estimates. This should be called whenever gcController.heapScan,
// gcController.heapLive, or if any inputs to gcController.heapGoal are
// updated. It is safe to call concurrently, but it may race with other
// calls to revise.
//
// The result of this race is that the two assist ratio values may not line
// up or may be stale. In practice this is OK because the assist ratio
// moves slowly throughout a GC cycle, and the assist ratio is a best-effort
// heuristic anyway. Furthermore, no part of the heuristic depends on
// the two assist ratio values being exact reciprocals of one another, since
// the two values are used to convert values from different sources.
//
// The worst case result of this raciness is that we may miss a larger shift
// in the ratio (say, if we decide to pace more aggressively against the
// hard heap goal) but even this "hard goal" is best-effort (see #40460).
// The dedicated GC should ensure we don't exceed the hard goal by too much
// in the rare case we do exceed it.
//
// It should only be called when gcBlackenEnabled != 0 (because this
// is when assists are enabled and the necessary statistics are
// available).
func (c *gcControllerState) revise() {
	gcPercent := c.gcPercent.Load()
	if gcPercent < 0 {
		// If GC is disabled but we're running a forced GC,
		// act like GOGC is huge for the below calculations.
		gcPercent = 100000
	}
	live := c.heapLive.Load()
	scan := c.heapScan.Load()
	work := c.heapScanWork.Load() + c.stackScanWork.Load() + c.globalsScanWork.Load()

	// Assume we're under the soft goal. Pace GC to complete at
	// heapGoal assuming the heap is in steady-state.
	heapGoal := int64(c.heapGoal())

	// The expected scan work is computed as the amount of bytes scanned last
	// GC cycle (both heap and stack), plus our estimate of globals work for this cycle.
	scanWorkExpected := int64(c.lastHeapScan + c.lastStackScan.Load() + c.globalsScan.Load())

	// maxScanWork is a worst-case estimate of the amount of scan work that
	// needs to be performed in this GC cycle. Specifically, it represents
	// the case where *all* scannable memory turns out to be live, and
	// *all* allocated stack space is scannable.
	maxStackScan := c.maxStackScan.Load()
	maxScanWork := int64(scan + maxStackScan + c.globalsScan.Load())
	if work > scanWorkExpected {
		// We've already done more scan work than expected. Because our expectation
		// is based on a steady-state scannable heap size, we assume this means our
		// heap is growing. Compute a new heap goal that takes our existing runway
		// computed for scanWorkExpected and extrapolates it to maxScanWork, the worst-case
		// scan work. This keeps our assist ratio stable if the heap continues to grow.
		//
		// The effect of this mechanism is that assists stay flat in the face of heap
		// growths. It's OK to use more memory this cycle to scan all the live heap,
		// because the next GC cycle is inevitably going to use *at least* that much
		// memory anyway.
		extHeapGoal := int64(float64(heapGoal-int64(c.triggered))/float64(scanWorkExpected)*float64(maxScanWork)) + int64(c.triggered)
		scanWorkExpected = maxScanWork

		// hardGoal is a hard limit on the amount that we're willing to push back the
		// heap goal, and that's twice the heap goal (i.e. if GOGC=100 and the heap and/or
		// stacks and/or globals grow to twice their size, this limits the current GC cycle's
		// growth to 4x the original live heap's size).
		//
		// This maintains the invariant that we use no more memory than the next GC cycle
		// will anyway.
		hardGoal := int64((1.0 + float64(gcPercent)/100.0) * float64(heapGoal))
		if extHeapGoal > hardGoal {
			extHeapGoal = hardGoal
		}
		heapGoal = extHeapGoal
	}
	if int64(live) > heapGoal {
		// We're already past our heap goal, even the extrapolated one.
		// Leave ourselves some extra runway, so in the worst case we
		// finish by that point.
		const maxOvershoot = 1.1
		heapGoal = int64(float64(heapGoal) * maxOvershoot)

		// Compute the upper bound on the scan work remaining.
		scanWorkExpected = maxScanWork
	}

	// Compute the remaining scan work estimate.
	//
	// Note that we currently count allocations during GC as both
	// scannable heap (heapScan) and scan work completed
	// (scanWork), so allocation will change this difference
	// slowly in the soft regime and not at all in the hard
	// regime.
	scanWorkRemaining := scanWorkExpected - work
	if scanWorkRemaining < 1000 {
		// We set a somewhat arbitrary lower bound on
		// remaining scan work since if we aim a little high,
		// we can miss by a little.
		//
		// We *do* need to enforce that this is at least 1,
		// since marking is racy and double-scanning objects
		// may legitimately make the remaining scan work
		// negative, even in the hard goal regime.
		scanWorkRemaining = 1000
	}

	// Compute the heap distance remaining.
	heapRemaining := heapGoal - int64(live)
	if heapRemaining <= 0 {
		// This shouldn't happen, but if it does, avoid
		// dividing by zero or setting the assist negative.
		heapRemaining = 1
	}

	// Compute the mutator assist ratio so by the time the mutator
	// allocates the remaining heap bytes up to heapGoal, it will
	// have done (or stolen) the remaining amount of scan work.
	// Note that the assist ratio values are updated atomically
	// but not together. This means there may be some degree of
	// skew between the two values. This is generally OK as the
	// values shift relatively slowly over the course of a GC
	// cycle.
	assistWorkPerByte := float64(scanWorkRemaining) / float64(heapRemaining)
	assistBytesPerWork := float64(heapRemaining) / float64(scanWorkRemaining)
	c.assistWorkPerByte.Store(assistWorkPerByte)
	c.assistBytesPerWork.Store(assistBytesPerWork)
}

// endCycle computes the consMark estimate for the next cycle.
// userForced indicates whether the current GC cycle was forced
// by the application.
func (c *gcControllerState) endCycle(now int64, procs int, userForced bool) {
	// Record last heap goal for the scavenger.
	// We'll be updating the heap goal soon.
	gcController.lastHeapGoal = c.heapGoal()

	// Compute the duration of time for which assists were turned on.
	assistDuration := now - c.markStartTime

	// Assume background mark hit its utilization goal.
	utilization := gcBackgroundUtilization
	// Add assist utilization; avoid divide by zero.
	if assistDuration > 0 {
		utilization += float64(c.assistTime.Load()) / float64(assistDuration*int64(procs))
	}

	if c.heapLive.Load() <= c.triggered {
		// Shouldn't happen, but let's be very safe about this in case the
		// GC is somehow extremely short.
		//
		// In this case though, the only reasonable value for c.heapLive-c.triggered
		// would be 0, which isn't really all that useful, i.e. the GC was so short
		// that it didn't matter.
		//
		// Ignore this case and don't update anything.
		return
	}
	idleUtilization := 0.0
	if assistDuration > 0 {
		idleUtilization = float64(c.idleMarkTime.Load()) / float64(assistDuration*int64(procs))
	}
	// Determine the cons/mark ratio.
	//
	// The units we want for the numerator and denominator are both B / cpu-ns.
	// We get this by taking the bytes allocated or scanned, and divide by the amount of
	// CPU time it took for those operations. For allocations, that CPU time is
	//
	//    assistDuration * procs * (1 - utilization)
	//
	// Where utilization includes just background GC workers and assists. It does *not*
	// include idle GC work time, because in theory the mutator is free to take that at
	// any point.
	//
	// For scanning, that CPU time is
	//
	//    assistDuration * procs * (utilization + idleUtilization)
	//
	// In this case, we *include* idle utilization, because that is additional CPU time that
	// the GC had available to it.
	//
	// In effect, idle GC time is sort of double-counted here, but it's very weird compared
	// to other kinds of GC work, because of how fluid it is. Namely, because the mutator is
	// *always* free to take it.
	//
	// So this calculation is really:
	//     (heapLive-trigger) / (assistDuration * procs * (1-utilization)) /
	//         (scanWork) / (assistDuration * procs * (utilization+idleUtilization))
	//
	// Note that because we only care about the ratio, assistDuration and procs cancel out.
	scanWork := c.heapScanWork.Load() + c.stackScanWork.Load() + c.globalsScanWork.Load()
	currentConsMark := (float64(c.heapLive.Load()-c.triggered) * (utilization + idleUtilization)) /
		(float64(scanWork) * (1 - utilization))

	// Update our cons/mark estimate. This is the maximum of the value we just computed and the last
	// 4 cons/mark values we measured. The reason we take the maximum here is to bias a noisy
	// cons/mark measurement toward fewer assists at the expense of additional GC cycles (starting
	// earlier).
	oldConsMark := c.consMark
	c.consMark = currentConsMark
	for i := range c.lastConsMark {
		if c.lastConsMark[i] > c.consMark {
			c.consMark = c.lastConsMark[i]
		}
	}
	copy(c.lastConsMark[:], c.lastConsMark[1:])
	c.lastConsMark[len(c.lastConsMark)-1] = currentConsMark

	if debug.gcpacertrace > 0 {
		printlock()
		goal := gcGoalUtilization * 100
		print("pacer: ", int(utilization*100), "% CPU (", int(goal), " exp.) for ")
		print(c.heapScanWork.Load(), "+", c.stackScanWork.Load(), "+", c.globalsScanWork.Load(), " B work (", c.lastHeapScan+c.lastStackScan.Load()+c.globalsScan.Load(), " B exp.) ")
		live := c.heapLive.Load()
		print("in ", c.triggered, " B -> ", live, " B (∆goal ", int64(live)-int64(c.lastHeapGoal), ", cons/mark ", oldConsMark, ")")
		println()
		printunlock()
	}
}

// enlistWorker encourages another dedicated mark worker to start on
// another P if there are spare worker slots. It is used by putfull
// when more work is made available.
//
// If goexperiment.GreenTeaGC, the caller must not hold a G's scan bit,
// otherwise this could cause a deadlock. This is already enforced by
// the static lock ranking.
//
//go:nowritebarrier
func (c *gcControllerState) enlistWorker() {
	needDedicated := c.dedicatedMarkWorkersNeeded.Load() > 0

	// Create new workers from idle Ps with goexperiment.GreenTeaGC.
	//
	// Note: with Green Tea, this places a requirement on enlistWorker
	// that it must not be called while a G's scan bit is held.
	if goexperiment.GreenTeaGC {
		needIdle := c.needIdleMarkWorker()

		// If we're all full on dedicated and idle workers, nothing
		// to do.
		if !needDedicated && !needIdle {
			return
		}

		// If there are idle Ps, wake one so it will run a worker
		// (the scheduler will already prefer to spin up a new
		// dedicated worker over an idle one).
		if sched.npidle.Load() != 0 && sched.nmspinning.Load() == 0 {
			wakep() // Likely to consume our worker request.
			return
		}
	}

	// If we still need more dedicated workers, try to preempt a running P
	// so it will switch to a worker.
	if !needDedicated {
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
		id := int32(cheaprandn(uint32(gomaxprocs - 1)))
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

// findRunnableGCWorker returns a background mark worker for pp if it
// should be run. This must only be called when gcBlackenEnabled != 0.
func (c *gcControllerState) findRunnableGCWorker(pp *p, now int64) (*g, int64) {
	if gcBlackenEnabled == 0 {
		throw("gcControllerState.findRunnable: blackening not enabled")
	}

	// Since we have the current time, check if the GC CPU limiter
	// hasn't had an update in a while. This check is necessary in
	// case the limiter is on but hasn't been checked in a while and
	// so may have left sufficient headroom to turn off again.
	if now == 0 {
		now = nanotime()
	}
	if gcCPULimiter.needUpdate(now) {
		gcCPULimiter.update(now)
	}

	if !gcShouldScheduleWorker(pp) {
		// No good reason to schedule a worker. This can happen at
		// the end of the mark phase when there are still
		// assists tapering off. Don't bother running a worker
		// now because it'll just return immediately.
		return nil, now
	}

	if c.dedicatedMarkWorkersNeeded.Load() <= 0 && c.fractionalUtilizationGoal == 0 {
		// No current need for dedicated workers, and no need at all for
		// fractional workers. Check before trying to acquire a worker; when
		// GOMAXPROCS is large, that can be expensive and is often unnecessary.
		//
		// When a dedicated worker stops running, the gcBgMarkWorker loop notes
		// the need for the worker before returning it to the pool. If we don't
		// see the need now, we wouldn't have found it in the pool anyway.
		return nil, now
	}

	// Grab a worker before we commit to running below.
	node := (*gcBgMarkWorkerNode)(gcBgMarkWorkerPool.pop())
	if node == nil {
		// There is at least one worker per P, so normally there are
		// enough workers to run on all Ps, if necessary. However, once
		// a worker enters gcMarkDone it may park without rejoining the
		// pool, thus freeing a P with no corresponding worker.
		// gcMarkDone never depends on another worker doing work, so it
		// is safe to simply do nothing here.
		//
		// If gcMarkDone bails out without completing the mark phase,
		// it will always do so with queued global work. Thus, that P
		// will be immediately eligible to re-run the worker G it was
		// just using, ensuring work can complete.
		return nil, now
	}

	decIfPositive := func(val *atomic.Int64) bool {
		for {
			v := val.Load()
			if v <= 0 {
				return false
			}

			if val.CompareAndSwap(v, v-1) {
				return true
			}
		}
	}

	if decIfPositive(&c.dedicatedMarkWorkersNeeded) {
		// This P is now dedicated to marking until the end of
		// the concurrent mark phase.
		pp.gcMarkWorkerMode = gcMarkWorkerDedicatedMode
	} else if c.fractionalUtilizationGoal == 0 {
		// No need for fractional workers.
		gcBgMarkWorkerPool.push(&node.node)
		return nil, now
	} else {
		// Is this P behind on the fractional utilization
		// goal?
		//
		// This should be kept in sync with pollFractionalWorkerExit.
		delta := now - c.markStartTime
		if delta > 0 && float64(pp.gcFractionalMarkTime)/float64(delta) > c.fractionalUtilizationGoal {
			// Nope. No need to run a fractional worker.
			gcBgMarkWorkerPool.push(&node.node)
			return nil, now
		}
		// Run a fractional worker.
		pp.gcMarkWorkerMode = gcMarkWorkerFractionalMode
	}

	// Run the background mark worker.
	gp := node.gp.ptr()
	trace := traceAcquire()
	casgstatus(gp, _Gwaiting, _Grunnable)
	if trace.ok() {
		trace.GoUnpark(gp, 0)
		traceRelease(trace)
	}
	return gp, now
}

// resetLive sets up the controller state for the next mark phase after the end
// of the previous one. Must be called after endCycle and before commit, before
// the world is started.
//
// The world must be stopped.
func (c *gcControllerState) resetLive(bytesMarked uint64) {
	c.heapMarked = bytesMarked
	c.heapLive.Store(bytesMarked)
	c.heapScan.Store(uint64(c.heapScanWork.Load()))
	c.lastHeapScan = uint64(c.heapScanWork.Load())
	c.lastStackScan.Store(uint64(c.stackScanWork.Load()))
	c.triggered = ^uint64(0) // Reset triggered.

	// heapLive was updated, so emit a trace event.
	trace := traceAcquire()
	if trace.ok() {
		trace.HeapAlloc(bytesMarked)
		traceRelease(trace)
	}
}

// markWorkerStop must be called whenever a mark worker stops executing.
//
// It updates mark work accounting in the controller by a duration of
// work in nanoseconds and other bookkeeping.
//
// Safe to execute at any time.
func (c *gcControllerState) markWorkerStop(mode gcMarkWorkerMode, duration int64) {
	switch mode {
	case gcMarkWorkerDedicatedMode:
		c.dedicatedMarkTime.Add(duration)
		c.dedicatedMarkWorkersNeeded.Add(1)
	case gcMarkWorkerFractionalMode:
		c.fractionalMarkTime.Add(duration)
	case gcMarkWorkerIdleMode:
		c.idleMarkTime.Add(duration)
		c.removeIdleMarkWorker()
	default:
		throw("markWorkerStop: unknown mark worker mode")
	}
}

func (c *gcControllerState) update(dHeapLive, dHeapScan int64) {
	if dHeapLive != 0 {
		trace := traceAcquire()
		live := gcController.heapLive.Add(dHeapLive)
		if trace.ok() {
			// gcController.heapLive changed.
			trace.HeapAlloc(live)
			traceRelease(trace)
		}
	}
	if gcBlackenEnabled == 0 {
		// Update heapScan when we're not in a current GC. It is fixed
		// at the beginning of a cycle.
		if dHeapScan != 0 {
			gcController.heapScan.Add(dHeapScan)
		}
	} else {
		// gcController.heapLive changed.
		c.revise()
	}
}

func (c *gcControllerState) addScannableStack(pp *p, amount int64) {
	if pp == nil {
		c.maxStackScan.Add(amount)
		return
	}
	pp.maxStackScanDelta += amount
	if pp.maxStackScanDelta >= maxStackScanSlack || pp.maxStackScanDelta <= -maxStackScanSlack {
		c.maxStackScan.Add(pp.maxStackScanDelta)
		pp.maxStackScanDelta = 0
	}
}

func (c *gcControllerState) addGlobals(amount int64) {
	c.globalsScan.Add(amount)
}

// heapGoal returns the current heap goal.
func (c *gcControllerState) heapGoal() uint64 {
	goal, _ := c.heapGoalInternal()
	return goal
}

// heapGoalInternal is the implementation of heapGoal which returns additional
// information that is necessary for computing the trigger.
//
// The returned minTrigger is always <= goal.
func (c *gcControllerState) heapGoalInternal() (goal, minTrigger uint64) {
	// Start with the goal calculated for gcPercent.
	goal = c.gcPercentHeapGoal.Load()

	// Check if the memory-limit-based goal is smaller, and if so, pick that.
	if newGoal := c.memoryLimitHeapGoal(); newGoal < goal {
		goal = newGoal
	} else {
		// We're not limited by the memory limit goal, so perform a series of
		// adjustments that might move the goal forward in a variety of circumstances.

		sweepDistTrigger := c.sweepDistMinTrigger.Load()
		if sweepDistTrigger > goal {
			// Set the goal to maintain a minimum sweep distance since
			// the last call to commit. Note that we never want to do this
			// if we're in the memory limit regime, because it could push
			// the goal up.
			goal = sweepDistTrigger
		}
		// Since we ignore the sweep distance trigger in the memory
		// limit regime, we need to ensure we don't propagate it to
		// the trigger, because it could cause a violation of the
		// invariant that the trigger < goal.
		minTrigger = sweepDistTrigger

		// Ensure that the heap goal is at least a little larger than
		// the point at which we triggered. This may not be the case if GC
		// start is delayed or if the allocation that pushed gcController.heapLive
		// over trigger is large or if the trigger is really close to
		// GOGC. Assist is proportional to this distance, so enforce a
		// minimum distance, even if it means going over the GOGC goal
		// by a tiny bit.
		//
		// Ignore this if we're in the memory limit regime: we'd prefer to
		// have the GC respond hard about how close we are to the goal than to
		// push the goal back in such a manner that it could cause us to exceed
		// the memory limit.
		const minRunway = 64 << 10
		if c.triggered != ^uint64(0) && goal < c.triggered+minRunway {
			goal = c.triggered + minRunway
		}
	}
	return
}

// memoryLimitHeapGoal returns a heap goal derived from memoryLimit.
func (c *gcControllerState) memoryLimitHeapGoal() uint64 {
	// Start by pulling out some values we'll need. Be careful about overflow.
	var heapFree, heapAlloc, mappedReady uint64
	for {
		heapFree = c.heapFree.load()                         // Free and unscavenged memory.
		heapAlloc = c.totalAlloc.Load() - c.totalFree.Load() // Heap object bytes in use.
		mappedReady = c.mappedReady.Load()                   // Total unreleased mapped memory.
		if heapFree+heapAlloc <= mappedReady {
			break
		}
		// It is impossible for total unreleased mapped memory to exceed heap memory, but
		// because these stats are updated independently, we may observe a partial update
		// including only some values. Thus, we appear to break the invariant. However,
		// this condition is necessarily transient, so just try again. In the case of a
		// persistent accounting error, we'll deadlock here.
	}

	// Below we compute a goal from memoryLimit. There are a few things to be aware of.
	// Firstly, the memoryLimit does not easily compare to the heap goal: the former
	// is total mapped memory by the runtime that hasn't been released, while the latter is
	// only heap object memory. Intuitively, the way we convert from one to the other is to
	// subtract everything from memoryLimit that both contributes to the memory limit (so,
	// ignore scavenged memory) and doesn't contain heap objects. This isn't quite what
	// lines up with reality, but it's a good starting point.
	//
	// In practice this computation looks like the following:
	//
	//    goal := memoryLimit - ((mappedReady - heapFree - heapAlloc) + max(mappedReady - memoryLimit, 0))
	//                    ^1                                    ^2
	//    goal -= goal / 100 * memoryLimitHeapGoalHeadroomPercent
	//    ^3
	//
	// Let's break this down.
	//
	// The first term (marker 1) is everything that contributes to the memory limit and isn't
	// or couldn't become heap objects. It represents, broadly speaking, non-heap overheads.
	// One oddity you may have noticed is that we also subtract out heapFree, i.e. unscavenged
	// memory that may contain heap objects in the future.
	//
	// Let's take a step back. In an ideal world, this term would look something like just
	// the heap goal. That is, we "reserve" enough space for the heap to grow to the heap
	// goal, and subtract out everything else. This is of course impossible; the definition
	// is circular! However, this impossible definition contains a key insight: the amount
	// we're *going* to use matters just as much as whatever we're currently using.
	//
	// Consider if the heap shrinks to 1/10th its size, leaving behind lots of free and
	// unscavenged memory. mappedReady - heapAlloc will be quite large, because of that free
	// and unscavenged memory, pushing the goal down significantly.
	//
	// heapFree is also safe to exclude from the memory limit because in the steady-state, it's
	// just a pool of memory for future heap allocations, and making new allocations from heapFree
	// memory doesn't increase overall memory use. In transient states, the scavenger and the
	// allocator actively manage the pool of heapFree memory to maintain the memory limit.
	//
	// The second term (marker 2) is the amount of memory we've exceeded the limit by, and is
	// intended to help recover from such a situation. By pushing the heap goal down, we also
	// push the trigger down, triggering and finishing a GC sooner in order to make room for
	// other memory sources. Note that since we're effectively reducing the heap goal by X bytes,
	// we're actually giving more than X bytes of headroom back, because the heap goal is in
	// terms of heap objects, but it takes more than X bytes (e.g. due to fragmentation) to store
	// X bytes worth of objects.
	//
	// The final adjustment (marker 3) reduces the maximum possible memory limit heap goal by
	// memoryLimitHeapGoalPercent. As the name implies, this is to provide additional headroom in
	// the face of pacing inaccuracies, and also to leave a buffer of unscavenged memory so the
	// allocator isn't constantly scavenging. The reduction amount also has a fixed minimum
	// (memoryLimitMinHeapGoalHeadroom, not pictured) because the aforementioned pacing inaccuracies
	// disproportionately affect small heaps: as heaps get smaller, the pacer's inputs get fuzzier.
	// Shorter GC cycles and less GC work means noisy external factors like the OS scheduler have a
	// greater impact.

	memoryLimit := uint64(c.memoryLimit.Load())

	// Compute term 1.
	nonHeapMemory := mappedReady - heapFree - heapAlloc

	// Compute term 2.
	var overage uint64
	if mappedReady > memoryLimit {
		overage = mappedReady - memoryLimit
	}

	if nonHeapMemory+overage >= memoryLimit {
		// We're at a point where non-heap memory exceeds the memory limit on its own.
		// There's honestly not much we can do here but just trigger GCs continuously
		// and let the CPU limiter reign that in. Something has to give at this point.
		// Set it to heapMarked, the lowest possible goal.
		return c.heapMarked
	}

	// Compute the goal.
	goal := memoryLimit - (nonHeapMemory + overage)

	// Apply some headroom to the goal to account for pacing inaccuracies and to reduce
	// the impact of scavenging at allocation time in response to a high allocation rate
	// when GOGC=off. See issue #57069. Also, be careful about small limits.
	headroom := goal / 100 * memoryLimitHeapGoalHeadroomPercent
	if headroom < memoryLimitMinHeapGoalHeadroom {
		// Set a fixed minimum to deal with the particularly large effect pacing inaccuracies
		// have for smaller heaps.
		headroom = memoryLimitMinHeapGoalHeadroom
	}
	if goal < headroom || goal-headroom < headroom {
		goal = headroom
	} else {
		goal = goal - headroom
	}
	// Don't let us go below the live heap. A heap goal below the live heap doesn't make sense.
	if goal < c.heapMarked {
		goal = c.heapMarked
	}
	return goal
}

const (
	// These constants determine the bounds on the GC trigger as a fraction
	// of heap bytes allocated between the start of a GC (heapLive == heapMarked)
	// and the end of a GC (heapLive == heapGoal).
	//
	// The constants are obscured in this way for efficiency. The denominator
	// of the fraction is always a power-of-two for a quick division, so that
	// the numerator is a single constant integer multiplication.
	triggerRatioDen = 64

	// The minimum trigger constant was chosen empirically: given a sufficiently
	// fast/scalable allocator with 48 Ps that could drive the trigger ratio
	// to <0.05, this constant causes applications to retain the same peak
	// RSS compared to not having this allocator.
	minTriggerRatioNum = 45 // ~0.7

	// The maximum trigger constant is chosen somewhat arbitrarily, but the
	// current constant has served us well over the years.
	maxTriggerRatioNum = 61 // ~0.95
)

// trigger returns the current point at which a GC should trigger along with
// the heap goal.
//
// The returned value may be compared against heapLive to determine whether
// the GC should trigger. Thus, the GC trigger condition should be (but may
// not be, in the case of small movements for efficiency) checked whenever
// the heap goal may change.
func (c *gcControllerState) trigger() (uint64, uint64) {
	goal, minTrigger := c.heapGoalInternal()

	// Invariant: the trigger must always be less than the heap goal.
	//
	// Note that the memory limit sets a hard maximum on our heap goal,
	// but the live heap may grow beyond it.

	if c.heapMarked >= goal {
		// The goal should never be smaller than heapMarked, but let's be
		// defensive about it. The only reasonable trigger here is one that
		// causes a continuous GC cycle at heapMarked, but respect the goal
		// if it came out as smaller than that.
		return goal, goal
	}

	// Below this point, c.heapMarked < goal.

	// heapMarked is our absolute minimum, and it's possible the trigger
	// bound we get from heapGoalinternal is less than that.
	if minTrigger < c.heapMarked {
		minTrigger = c.heapMarked
	}

	// If we let the trigger go too low, then if the application
	// is allocating very rapidly we might end up in a situation
	// where we're allocating black during a nearly always-on GC.
	// The result of this is a growing heap and ultimately an
	// increase in RSS. By capping us at a point >0, we're essentially
	// saying that we're OK using more CPU during the GC to prevent
	// this growth in RSS.
	triggerLowerBound := ((goal-c.heapMarked)/triggerRatioDen)*minTriggerRatioNum + c.heapMarked
	if minTrigger < triggerLowerBound {
		minTrigger = triggerLowerBound
	}

	// For small heaps, set the max trigger point at maxTriggerRatio of the way
	// from the live heap to the heap goal. This ensures we always have *some*
	// headroom when the GC actually starts. For larger heaps, set the max trigger
	// point at the goal, minus the minimum heap size.
	//
	// This choice follows from the fact that the minimum heap size is chosen
	// to reflect the costs of a GC with no work to do. With a large heap but
	// very little scan work to perform, this gives us exactly as much runway
	// as we would need, in the worst case.
	maxTrigger := ((goal-c.heapMarked)/triggerRatioDen)*maxTriggerRatioNum + c.heapMarked
	if goal > defaultHeapMinimum && goal-defaultHeapMinimum > maxTrigger {
		maxTrigger = goal - defaultHeapMinimum
	}
	maxTrigger = max(maxTrigger, minTrigger)

	// Compute the trigger from our bounds and the runway stored by commit.
	var trigger uint64
	runway := c.runway.Load()
	if runway > goal {
		trigger = minTrigger
	} else {
		trigger = goal - runway
	}
	trigger = max(trigger, minTrigger)
	trigger = min(trigger, maxTrigger)
	if trigger > goal {
		print("trigger=", trigger, " heapGoal=", goal, "\n")
		print("minTrigger=", minTrigger, " maxTrigger=", maxTrigger, "\n")
		throw("produced a trigger greater than the heap goal")
	}
	return trigger, goal
}

// commit recomputes all pacing parameters needed to derive the
// trigger and the heap goal. Namely, the gcPercent-based heap goal,
// and the amount of runway we want to give the GC this cycle.
//
// This can be called any time. If GC is the in the middle of a
// concurrent phase, it will adjust the pacing of that phase.
//
// isSweepDone should be the result of calling isSweepDone(),
// unless we're testing or we know we're executing during a GC cycle.
//
// This depends on gcPercent, gcController.heapMarked, and
// gcController.heapLive. These must be up to date.
//
// Callers must call gcControllerState.revise after calling this
// function if the GC is enabled.
//
// mheap_.lock must be held or the world must be stopped.
func (c *gcControllerState) commit(isSweepDone bool) {
	if !c.test {
		assertWorldStoppedOrLockHeld(&mheap_.lock)
	}

	if isSweepDone {
		// The sweep is done, so there aren't any restrictions on the trigger
		// we need to think about.
		c.sweepDistMinTrigger.Store(0)
	} else {
		// Concurrent sweep happens in the heap growth
		// from gcController.heapLive to trigger. Make sure we
		// give the sweeper some runway if it doesn't have enough.
		c.sweepDistMinTrigger.Store(c.heapLive.Load() + sweepMinHeapDistance)
	}

	// Compute the next GC goal, which is when the allocated heap
	// has grown by GOGC/100 over where it started the last cycle,
	// plus additional runway for non-heap sources of GC work.
	gcPercentHeapGoal := ^uint64(0)
	if gcPercent := c.gcPercent.Load(); gcPercent >= 0 {
		gcPercentHeapGoal = c.heapMarked + (c.heapMarked+c.lastStackScan.Load()+c.globalsScan.Load())*uint64(gcPercent)/100
	}
	// Apply the minimum heap size here. It's defined in terms of gcPercent
	// and is only updated by functions that call commit.
	if gcPercentHeapGoal < c.heapMinimum {
		gcPercentHeapGoal = c.heapMinimum
	}
	c.gcPercentHeapGoal.Store(gcPercentHeapGoal)

	// Compute the amount of runway we want the GC to have by using our
	// estimate of the cons/mark ratio.
	//
	// The idea is to take our expected scan work, and multiply it by
	// the cons/mark ratio to determine how long it'll take to complete
	// that scan work in terms of bytes allocated. This gives us our GC's
	// runway.
	//
	// However, the cons/mark ratio is a ratio of rates per CPU-second, but
	// here we care about the relative rates for some division of CPU
	// resources among the mutator and the GC.
	//
	// To summarize, we have B / cpu-ns, and we want B / ns. We get that
	// by multiplying by our desired division of CPU resources. We choose
	// to express CPU resources as GOMAPROCS*fraction. Note that because
	// we're working with a ratio here, we can omit the number of CPU cores,
	// because they'll appear in the numerator and denominator and cancel out.
	// As a result, this is basically just "weighing" the cons/mark ratio by
	// our desired division of resources.
	//
	// Furthermore, by setting the runway so that CPU resources are divided
	// this way, assuming that the cons/mark ratio is correct, we make that
	// division a reality.
	c.runway.Store(uint64((c.consMark * (1 - gcGoalUtilization) / (gcGoalUtilization)) * float64(c.lastHeapScan+c.lastStackScan.Load()+c.globalsScan.Load())))
}

// setGCPercent updates gcPercent. commit must be called after.
// Returns the old value of gcPercent.
//
// The world must be stopped, or mheap_.lock must be held.
func (c *gcControllerState) setGCPercent(in int32) int32 {
	if !c.test {
		assertWorldStoppedOrLockHeld(&mheap_.lock)
	}

	out := c.gcPercent.Load()
	if in < 0 {
		in = -1
	}
	c.heapMinimum = defaultHeapMinimum * uint64(in) / 100
	c.gcPercent.Store(in)

	return out
}

//go:linkname setGCPercent runtime/debug.setGCPercent
func setGCPercent(in int32) (out int32) {
	// Run on the system stack since we grab the heap lock.
	systemstack(func() {
		lock(&mheap_.lock)
		out = gcController.setGCPercent(in)
		gcControllerCommit()
		unlock(&mheap_.lock)
	})

	// If we just disabled GC, wait for any concurrent GC mark to
	// finish so we always return with no GC running.
	if in < 0 {
		gcWaitOnMark(work.cycles.Load())
	}

	return out
}

func readGOGC() int32 {
	p := gogetenv("GOGC")
	if p == "off" {
		return -1
	}
	if n, err := strconv.ParseInt(p, 10, 32); err == nil {
		return int32(n)
	}
	return 100
}

// setMemoryLimit updates memoryLimit. commit must be called after
// Returns the old value of memoryLimit.
//
// The world must be stopped, or mheap_.lock must be held.
func (c *gcControllerState) setMemoryLimit(in int64) int64 {
	if !c.test {
		assertWorldStoppedOrLockHeld(&mheap_.lock)
	}

	out := c.memoryLimit.Load()
	if in >= 0 {
		c.memoryLimit.Store(in)
	}

	return out
}

//go:linkname setMemoryLimit runtime/debug.setMemoryLimit
func setMemoryLimit(in int64) (out int64) {
	// Run on the system stack since we grab the heap lock.
	systemstack(func() {
		lock(&mheap_.lock)
		out = gcController.setMemoryLimit(in)
		if in < 0 || out == in {
			// If we're just checking the value or not changing
			// it, there's no point in doing the rest.
			unlock(&mheap_.lock)
			return
		}
		gcControllerCommit()
		unlock(&mheap_.lock)
	})
	return out
}

func readGOMEMLIMIT() int64 {
	p := gogetenv("GOMEMLIMIT")
	if p == "" || p == "off" {
		return math.MaxInt64
	}
	n, ok := parseByteCount(p)
	if !ok {
		print("GOMEMLIMIT=", p, "\n")
		throw("malformed GOMEMLIMIT; see `go doc runtime/debug.SetMemoryLimit`")
	}
	return n
}

// addIdleMarkWorker attempts to add a new idle mark worker.
//
// If this returns true, the caller must become an idle mark worker unless
// there's no background mark worker goroutines in the pool. This case is
// harmless because there are already background mark workers running.
// If this returns false, the caller must NOT become an idle mark worker.
//
// nosplit because it may be called without a P.
//
//go:nosplit
func (c *gcControllerState) addIdleMarkWorker() bool {
	for {
		old := c.idleMarkWorkers.Load()
		n, max := int32(old&uint64(^uint32(0))), int32(old>>32)
		if n >= max {
			// See the comment on idleMarkWorkers for why
			// n > max is tolerated.
			return false
		}
		if n < 0 {
			print("n=", n, " max=", max, "\n")
			throw("negative idle mark workers")
		}
		new := uint64(uint32(n+1)) | (uint64(max) << 32)
		if c.idleMarkWorkers.CompareAndSwap(old, new) {
			return true
		}
	}
}

// needIdleMarkWorker is a hint as to whether another idle mark worker is needed.
//
// The caller must still call addIdleMarkWorker to become one. This is mainly
// useful for a quick check before an expensive operation.
//
// nosplit because it may be called without a P.
//
//go:nosplit
func (c *gcControllerState) needIdleMarkWorker() bool {
	p := c.idleMarkWorkers.Load()
	n, max := int32(p&uint64(^uint32(0))), int32(p>>32)
	return n < max
}

// removeIdleMarkWorker must be called when a new idle mark worker stops executing.
func (c *gcControllerState) removeIdleMarkWorker() {
	for {
		old := c.idleMarkWorkers.Load()
		n, max := int32(old&uint64(^uint32(0))), int32(old>>32)
		if n-1 < 0 {
			print("n=", n, " max=", max, "\n")
			throw("negative idle mark workers")
		}
		new := uint64(uint32(n-1)) | (uint64(max) << 32)
		if c.idleMarkWorkers.CompareAndSwap(old, new) {
			return
		}
	}
}

// setMaxIdleMarkWorkers sets the maximum number of idle mark workers allowed.
//
// This method is optimistic in that it does not wait for the number of
// idle mark workers to reduce to max before returning; it assumes the workers
// will deschedule themselves.
func (c *gcControllerState) setMaxIdleMarkWorkers(max int32) {
	for {
		old := c.idleMarkWorkers.Load()
		n := int32(old & uint64(^uint32(0)))
		if n < 0 {
			print("n=", n, " max=", max, "\n")
			throw("negative idle mark workers")
		}
		new := uint64(uint32(n)) | (uint64(max) << 32)
		if c.idleMarkWorkers.CompareAndSwap(old, new) {
			return
		}
	}
}

// gcControllerCommit is gcController.commit, but passes arguments from live
// (non-test) data. It also updates any consumers of the GC pacing, such as
// sweep pacing and the background scavenger.
//
// Calls gcController.commit.
//
// The heap lock must be held, so this must be executed on the system stack.
//
//go:systemstack
func gcControllerCommit() {
	assertWorldStoppedOrLockHeld(&mheap_.lock)

	gcController.commit(isSweepDone())

	// Update mark pacing.
	if gcphase != _GCoff {
		gcController.revise()
	}

	// TODO(mknyszek): This isn't really accurate any longer because the heap
	// goal is computed dynamically. Still useful to snapshot, but not as useful.
	trace := traceAcquire()
	if trace.ok() {
		trace.HeapGoal()
		traceRelease(trace)
	}

	trigger, heapGoal := gcController.trigger()
	gcPaceSweeper(trigger)
	gcPaceScavenger(gcController.memoryLimit.Load(), heapGoal, gcController.lastHeapGoal)
}

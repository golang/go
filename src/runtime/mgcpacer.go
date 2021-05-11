// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
	"runtime/internal/atomic"
	"unsafe"
)

const (
	// gcGoalUtilization is the goal CPU utilization for
	// marking as a fraction of GOMAXPROCS.
	gcGoalUtilization = 0.30

	// gcBackgroundUtilization is the fixed CPU utilization for background
	// marking. It must be <= gcGoalUtilization. The difference between
	// gcGoalUtilization and gcBackgroundUtilization will be made up by
	// mark assists. The scheduler will aim to use within 50% of this
	// goal.
	//
	// Setting this to < gcGoalUtilization avoids saturating the trigger
	// feedback controller when there are no assists, which allows it to
	// better control CPU and heap growth. However, the larger the gap,
	// the more mutator assists are expected to happen, which impact
	// mutator latency.
	gcBackgroundUtilization = 0.25

	// gcCreditSlack is the amount of scan work credit that can
	// accumulate locally before updating gcController.scanWork and,
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
	defaultHeapMinimum = 4 << 20
)

func init() {
	if offset := unsafe.Offsetof(gcController.heapLive); offset%8 != 0 {
		println(offset)
		throw("gcController.heapLive not aligned to 8 bytes")
	}
}

// gcController implements the GC pacing controller that determines
// when to trigger concurrent garbage collection and how much marking
// work to do in mutator assists and background marking.
//
// It uses a feedback control algorithm to adjust the gcController.trigger
// trigger based on the heap growth and GC CPU utilization each cycle.
// This algorithm optimizes for heap growth to match GOGC and for CPU
// utilization between assist and background marking to be 25% of
// GOMAXPROCS. The high-level design of this algorithm is documented
// at https://golang.org/s/go15gcpacing.
//
// All fields of gcController are used only during a single mark
// cycle.
var gcController gcControllerState

type gcControllerState struct {
	// Initialized from $GOGC. GOGC=off means no GC.
	gcPercent int32

	_ uint32 // padding so following 64-bit values are 8-byte aligned

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

	// triggerRatio is the heap growth ratio that triggers marking.
	//
	// E.g., if this is 0.6, then GC should start when the live
	// heap has reached 1.6 times the heap size marked by the
	// previous cycle. This should be ≤ GOGC/100 so the trigger
	// heap size is less than the goal heap size. This is set
	// during mark termination for the next cycle's trigger.
	//
	// Protected by mheap_.lock or a STW.
	triggerRatio float64

	// trigger is the heap size that triggers marking.
	//
	// When heapLive ≥ trigger, the mark phase will start.
	// This is also the heap size by which proportional sweeping
	// must be complete.
	//
	// This is computed from triggerRatio during mark termination
	// for the next cycle's trigger.
	//
	// Protected by mheap_.lock or a STW.
	trigger uint64

	// heapGoal is the goal heapLive for when next GC ends.
	// Set to ^uint64(0) if disabled.
	//
	// Read and written atomically, unless the world is stopped.
	heapGoal uint64

	// lastHeapGoal is the value of heapGoal for the previous GC.
	// Note that this is distinct from the last value heapGoal had,
	// because it could change if e.g. gcPercent changes.
	//
	// Read and written with the world stopped or with mheap_.lock held.
	lastHeapGoal uint64

	// heapLive is the number of bytes considered live by the GC.
	// That is: retained by the most recent GC plus allocated
	// since then. heapLive ≤ memstats.heapAlloc, since heapAlloc includes
	// unmarked objects that have not yet been swept (and hence goes up as we
	// allocate and down as we sweep) while heapLive excludes these
	// objects (and hence only goes up between GCs).
	//
	// This is updated atomically without locking. To reduce
	// contention, this is updated only when obtaining a span from
	// an mcentral and at this point it counts all of the
	// unallocated slots in that span (which will be allocated
	// before that mcache obtains another span from that
	// mcentral). Hence, it slightly overestimates the "true" live
	// heap size. It's better to overestimate than to
	// underestimate because 1) this triggers the GC earlier than
	// necessary rather than potentially too late and 2) this
	// leads to a conservative GC rate rather than a GC rate that
	// is potentially too low.
	//
	// Reads should likewise be atomic (or during STW).
	//
	// Whenever this is updated, call traceHeapAlloc() and
	// this gcControllerState's revise() method.
	heapLive uint64

	// heapScan is the number of bytes of "scannable" heap. This
	// is the live heap (as counted by heapLive), but omitting
	// no-scan objects and no-scan tails of objects.
	//
	// Whenever this is updated, call this gcControllerState's
	// revise() method.
	//
	// Read and written atomically or with the world stopped.
	heapScan uint64

	// heapMarked is the number of bytes marked by the previous
	// GC. After mark termination, heapLive == heapMarked, but
	// unlike heapLive, heapMarked does not change until the
	// next mark termination.
	heapMarked uint64

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
	// time heapScan is updated.
	//
	// Stored as a uint64, but it's actually a float64. Use
	// float64frombits to get the value.
	//
	// Read and written atomically.
	assistWorkPerByte uint64

	// assistBytesPerWork is 1/assistWorkPerByte.
	//
	// Stored as a uint64, but it's actually a float64. Use
	// float64frombits to get the value.
	//
	// Read and written atomically.
	//
	// Note that because this is read and written independently
	// from assistWorkPerByte users may notice a skew between
	// the two values, and such a state should be safe.
	assistBytesPerWork uint64

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

	_ cpu.CacheLinePad
}

func (c *gcControllerState) init(gcPercent int32) {
	c.heapMinimum = defaultHeapMinimum

	// Set a reasonable initial GC trigger.
	c.triggerRatio = 7 / 8.0

	// Fake a heapMarked value so it looks like a trigger at
	// heapMinimum is the appropriate growth from heapMarked.
	// This will go into computing the initial GC goal.
	c.heapMarked = uint64(float64(c.heapMinimum) / (1 + c.triggerRatio))

	// This will also compute and set the GC trigger and goal.
	c.setGCPercent(gcPercent)
}

// startCycle resets the GC controller's state and computes estimates
// for a new GC cycle. The caller must hold worldsema and the world
// must be stopped.
func (c *gcControllerState) startCycle() {
	c.scanWork = 0
	c.bgScanCredit = 0
	c.assistTime = 0
	c.dedicatedMarkTime = 0
	c.fractionalMarkTime = 0
	c.idleMarkTime = 0

	// Ensure that the heap goal is at least a little larger than
	// the current live heap size. This may not be the case if GC
	// start is delayed or if the allocation that pushed gcController.heapLive
	// over trigger is large or if the trigger is really close to
	// GOGC. Assist is proportional to this distance, so enforce a
	// minimum distance, even if it means going over the GOGC goal
	// by a tiny bit.
	if c.heapGoal < c.heapLive+1024*1024 {
		c.heapGoal = c.heapLive + 1024*1024
	}

	// Compute the background mark utilization goal. In general,
	// this may not come out exactly. We round the number of
	// dedicated workers so that the utilization is closest to
	// 25%. For small GOMAXPROCS, this would introduce too much
	// error, so we add fractional workers in that case.
	totalUtilizationGoal := float64(gomaxprocs) * gcBackgroundUtilization
	c.dedicatedMarkWorkersNeeded = int64(totalUtilizationGoal + 0.5)
	utilError := float64(c.dedicatedMarkWorkersNeeded)/totalUtilizationGoal - 1
	const maxUtilError = 0.3
	if utilError < -maxUtilError || utilError > maxUtilError {
		// Rounding put us more than 30% off our goal. With
		// gcBackgroundUtilization of 25%, this happens for
		// GOMAXPROCS<=3 or GOMAXPROCS=6. Enable fractional
		// workers to compensate.
		if float64(c.dedicatedMarkWorkersNeeded) > totalUtilizationGoal {
			// Too many dedicated workers.
			c.dedicatedMarkWorkersNeeded--
		}
		c.fractionalUtilizationGoal = (totalUtilizationGoal - float64(c.dedicatedMarkWorkersNeeded)) / float64(gomaxprocs)
	} else {
		c.fractionalUtilizationGoal = 0
	}

	// In STW mode, we just want dedicated workers.
	if debug.gcstoptheworld > 0 {
		c.dedicatedMarkWorkersNeeded = int64(gomaxprocs)
		c.fractionalUtilizationGoal = 0
	}

	// Clear per-P state
	for _, p := range allp {
		p.gcAssistTime = 0
		p.gcFractionalMarkTime = 0
	}

	// Compute initial values for controls that are updated
	// throughout the cycle.
	c.revise()

	if debug.gcpacertrace > 0 {
		assistRatio := float64frombits(atomic.Load64(&c.assistWorkPerByte))
		print("pacer: assist ratio=", assistRatio,
			" (scan ", gcController.heapScan>>20, " MB in ",
			work.initialHeapLive>>20, "->",
			c.heapGoal>>20, " MB)",
			" workers=", c.dedicatedMarkWorkersNeeded,
			"+", c.fractionalUtilizationGoal, "\n")
	}
}

// revise updates the assist ratio during the GC cycle to account for
// improved estimates. This should be called whenever gcController.heapScan,
// gcController.heapLive, or gcController.heapGoal is updated. It is safe to
// call concurrently, but it may race with other calls to revise.
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
	gcPercent := c.gcPercent
	if gcPercent < 0 {
		// If GC is disabled but we're running a forced GC,
		// act like GOGC is huge for the below calculations.
		gcPercent = 100000
	}
	live := atomic.Load64(&c.heapLive)
	scan := atomic.Load64(&c.heapScan)
	work := atomic.Loadint64(&c.scanWork)

	// Assume we're under the soft goal. Pace GC to complete at
	// heapGoal assuming the heap is in steady-state.
	heapGoal := int64(atomic.Load64(&c.heapGoal))

	// Compute the expected scan work remaining.
	//
	// This is estimated based on the expected
	// steady-state scannable heap. For example, with
	// GOGC=100, only half of the scannable heap is
	// expected to be live, so that's what we target.
	//
	// (This is a float calculation to avoid overflowing on
	// 100*heapScan.)
	scanWorkExpected := int64(float64(scan) * 100 / float64(100+gcPercent))

	if int64(live) > heapGoal || work > scanWorkExpected {
		// We're past the soft goal, or we've already done more scan
		// work than we expected. Pace GC so that in the worst case it
		// will complete by the hard goal.
		const maxOvershoot = 1.1
		heapGoal = int64(float64(heapGoal) * maxOvershoot)

		// Compute the upper bound on the scan work remaining.
		scanWorkExpected = int64(scan)
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
	atomic.Store64(&c.assistWorkPerByte, float64bits(assistWorkPerByte))
	atomic.Store64(&c.assistBytesPerWork, float64bits(assistBytesPerWork))
}

// endCycle computes the trigger ratio for the next cycle.
// userForced indicates whether the current GC cycle was forced
// by the application.
func (c *gcControllerState) endCycle(userForced bool) float64 {
	if userForced {
		// Forced GC means this cycle didn't start at the
		// trigger, so where it finished isn't good
		// information about how to adjust the trigger.
		// Just leave it where it is.
		return c.triggerRatio
	}

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
	goalGrowthRatio := c.effectiveGrowthRatio()
	actualGrowthRatio := float64(c.heapLive)/float64(c.heapMarked) - 1
	assistDuration := nanotime() - c.markStartTime

	// Assume background mark hit its utilization goal.
	utilization := gcBackgroundUtilization
	// Add assist utilization; avoid divide by zero.
	if assistDuration > 0 {
		utilization += float64(c.assistTime) / float64(assistDuration*int64(gomaxprocs))
	}

	triggerError := goalGrowthRatio - c.triggerRatio - utilization/gcGoalUtilization*(actualGrowthRatio-c.triggerRatio)

	// Finally, we adjust the trigger for next time by this error,
	// damped by the proportional gain.
	triggerRatio := c.triggerRatio + triggerGain*triggerError

	if debug.gcpacertrace > 0 {
		// Print controller state in terms of the design
		// document.
		H_m_prev := c.heapMarked
		h_t := c.triggerRatio
		H_T := c.trigger
		h_a := actualGrowthRatio
		H_a := c.heapLive
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

	return triggerRatio
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
		id := int32(fastrandn(uint32(gomaxprocs - 1)))
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

// findRunnableGCWorker returns a background mark worker for _p_ if it
// should be run. This must only be called when gcBlackenEnabled != 0.
func (c *gcControllerState) findRunnableGCWorker(_p_ *p) *g {
	if gcBlackenEnabled == 0 {
		throw("gcControllerState.findRunnable: blackening not enabled")
	}

	if !gcMarkWorkAvailable(_p_) {
		// No work to be done right now. This can happen at
		// the end of the mark phase when there are still
		// assists tapering off. Don't bother running a worker
		// now because it'll just return immediately.
		return nil
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
		return nil
	}

	decIfPositive := func(ptr *int64) bool {
		for {
			v := atomic.Loadint64(ptr)
			if v <= 0 {
				return false
			}

			if atomic.Casint64(ptr, v, v-1) {
				return true
			}
		}
	}

	if decIfPositive(&c.dedicatedMarkWorkersNeeded) {
		// This P is now dedicated to marking until the end of
		// the concurrent mark phase.
		_p_.gcMarkWorkerMode = gcMarkWorkerDedicatedMode
	} else if c.fractionalUtilizationGoal == 0 {
		// No need for fractional workers.
		gcBgMarkWorkerPool.push(&node.node)
		return nil
	} else {
		// Is this P behind on the fractional utilization
		// goal?
		//
		// This should be kept in sync with pollFractionalWorkerExit.
		delta := nanotime() - c.markStartTime
		if delta > 0 && float64(_p_.gcFractionalMarkTime)/float64(delta) > c.fractionalUtilizationGoal {
			// Nope. No need to run a fractional worker.
			gcBgMarkWorkerPool.push(&node.node)
			return nil
		}
		// Run a fractional worker.
		_p_.gcMarkWorkerMode = gcMarkWorkerFractionalMode
	}

	// Run the background mark worker.
	gp := node.gp.ptr()
	casgstatus(gp, _Gwaiting, _Grunnable)
	if trace.enabled {
		traceGoUnpark(gp, 0)
	}
	return gp
}

// commit sets the trigger ratio and updates everything
// derived from it: the absolute trigger, the heap goal, mark pacing,
// and sweep pacing.
//
// This can be called any time. If GC is the in the middle of a
// concurrent phase, it will adjust the pacing of that phase.
//
// This depends on gcPercent, gcController.heapMarked, and
// gcController.heapLive. These must be up to date.
//
// mheap_.lock must be held or the world must be stopped.
func (c *gcControllerState) commit(triggerRatio float64) {
	assertWorldStoppedOrLockHeld(&mheap_.lock)

	// Compute the next GC goal, which is when the allocated heap
	// has grown by GOGC/100 over the heap marked by the last
	// cycle.
	goal := ^uint64(0)
	if c.gcPercent >= 0 {
		goal = c.heapMarked + c.heapMarked*uint64(c.gcPercent)/100
	}

	// Set the trigger ratio, capped to reasonable bounds.
	if c.gcPercent >= 0 {
		scalingFactor := float64(c.gcPercent) / 100
		// Ensure there's always a little margin so that the
		// mutator assist ratio isn't infinity.
		maxTriggerRatio := 0.95 * scalingFactor
		if triggerRatio > maxTriggerRatio {
			triggerRatio = maxTriggerRatio
		}

		// If we let triggerRatio go too low, then if the application
		// is allocating very rapidly we might end up in a situation
		// where we're allocating black during a nearly always-on GC.
		// The result of this is a growing heap and ultimately an
		// increase in RSS. By capping us at a point >0, we're essentially
		// saying that we're OK using more CPU during the GC to prevent
		// this growth in RSS.
		//
		// The current constant was chosen empirically: given a sufficiently
		// fast/scalable allocator with 48 Ps that could drive the trigger ratio
		// to <0.05, this constant causes applications to retain the same peak
		// RSS compared to not having this allocator.
		minTriggerRatio := 0.6 * scalingFactor
		if triggerRatio < minTriggerRatio {
			triggerRatio = minTriggerRatio
		}
	} else if triggerRatio < 0 {
		// gcPercent < 0, so just make sure we're not getting a negative
		// triggerRatio. This case isn't expected to happen in practice,
		// and doesn't really matter because if gcPercent < 0 then we won't
		// ever consume triggerRatio further on in this function, but let's
		// just be defensive here; the triggerRatio being negative is almost
		// certainly undesirable.
		triggerRatio = 0
	}
	c.triggerRatio = triggerRatio

	// Compute the absolute GC trigger from the trigger ratio.
	//
	// We trigger the next GC cycle when the allocated heap has
	// grown by the trigger ratio over the marked heap size.
	trigger := ^uint64(0)
	if c.gcPercent >= 0 {
		trigger = uint64(float64(c.heapMarked) * (1 + triggerRatio))
		// Don't trigger below the minimum heap size.
		minTrigger := c.heapMinimum
		if !isSweepDone() {
			// Concurrent sweep happens in the heap growth
			// from gcController.heapLive to trigger, so ensure
			// that concurrent sweep has some heap growth
			// in which to perform sweeping before we
			// start the next GC cycle.
			sweepMin := atomic.Load64(&c.heapLive) + sweepMinHeapDistance
			if sweepMin > minTrigger {
				minTrigger = sweepMin
			}
		}
		if trigger < minTrigger {
			trigger = minTrigger
		}
		if int64(trigger) < 0 {
			print("runtime: heapGoal=", c.heapGoal, " heapMarked=", c.heapMarked, " gcController.heapLive=", c.heapLive, " initialHeapLive=", work.initialHeapLive, "triggerRatio=", triggerRatio, " minTrigger=", minTrigger, "\n")
			throw("trigger underflow")
		}
		if trigger > goal {
			// The trigger ratio is always less than GOGC/100, but
			// other bounds on the trigger may have raised it.
			// Push up the goal, too.
			goal = trigger
		}
	}

	// Commit to the trigger and goal.
	c.trigger = trigger
	atomic.Store64(&c.heapGoal, goal)
	if trace.enabled {
		traceHeapGoal()
	}

	// Update mark pacing.
	if gcphase != _GCoff {
		c.revise()
	}

	// Update sweep pacing.
	if isSweepDone() {
		mheap_.sweepPagesPerByte = 0
	} else {
		// Concurrent sweep needs to sweep all of the in-use
		// pages by the time the allocated heap reaches the GC
		// trigger. Compute the ratio of in-use pages to sweep
		// per byte allocated, accounting for the fact that
		// some might already be swept.
		heapLiveBasis := atomic.Load64(&c.heapLive)
		heapDistance := int64(trigger) - int64(heapLiveBasis)
		// Add a little margin so rounding errors and
		// concurrent sweep are less likely to leave pages
		// unswept when GC starts.
		heapDistance -= 1024 * 1024
		if heapDistance < _PageSize {
			// Avoid setting the sweep ratio extremely high
			heapDistance = _PageSize
		}
		pagesSwept := atomic.Load64(&mheap_.pagesSwept)
		pagesInUse := atomic.Load64(&mheap_.pagesInUse)
		sweepDistancePages := int64(pagesInUse) - int64(pagesSwept)
		if sweepDistancePages <= 0 {
			mheap_.sweepPagesPerByte = 0
		} else {
			mheap_.sweepPagesPerByte = float64(sweepDistancePages) / float64(heapDistance)
			mheap_.sweepHeapLiveBasis = heapLiveBasis
			// Write pagesSweptBasis last, since this
			// signals concurrent sweeps to recompute
			// their debt.
			atomic.Store64(&mheap_.pagesSweptBasis, pagesSwept)
		}
	}

	gcPaceScavenger()
}

// effectiveGrowthRatio returns the current effective heap growth
// ratio (GOGC/100) based on heapMarked from the previous GC and
// heapGoal for the current GC.
//
// This may differ from gcPercent/100 because of various upper and
// lower bounds on gcPercent. For example, if the heap is smaller than
// heapMinimum, this can be higher than gcPercent/100.
//
// mheap_.lock must be held or the world must be stopped.
func (c *gcControllerState) effectiveGrowthRatio() float64 {
	assertWorldStoppedOrLockHeld(&mheap_.lock)

	egogc := float64(atomic.Load64(&c.heapGoal)-c.heapMarked) / float64(c.heapMarked)
	if egogc < 0 {
		// Shouldn't happen, but just in case.
		egogc = 0
	}
	return egogc
}

// setGCPercent updates gcPercent and all related pacer state.
// Returns the old value of gcPercent.
//
// The world must be stopped, or mheap_.lock must be held.
func (c *gcControllerState) setGCPercent(in int32) int32 {
	assertWorldStoppedOrLockHeld(&mheap_.lock)

	out := c.gcPercent
	if in < 0 {
		in = -1
	}
	c.gcPercent = in
	c.heapMinimum = defaultHeapMinimum * uint64(c.gcPercent) / 100
	// Update pacing in response to gcPercent change.
	c.commit(c.triggerRatio)

	return out
}

//go:linkname setGCPercent runtime/debug.setGCPercent
func setGCPercent(in int32) (out int32) {
	// Run on the system stack since we grab the heap lock.
	systemstack(func() {
		lock(&mheap_.lock)
		out = gcController.setGCPercent(in)
		unlock(&mheap_.lock)
	})

	// If we just disabled GC, wait for any concurrent GC mark to
	// finish so we always return with no GC running.
	if in < 0 {
		gcWaitOnMark(atomic.Load(&work.cycles))
	}

	return out
}

func readGOGC() int32 {
	p := gogetenv("GOGC")
	if p == "off" {
		return -1
	}
	if n, ok := atoi32(p); ok {
		return n
	}
	return 100
}

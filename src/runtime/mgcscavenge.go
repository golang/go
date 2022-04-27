// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Scavenging free pages.
//
// This file implements scavenging (the release of physical pages backing mapped
// memory) of free and unused pages in the heap as a way to deal with page-level
// fragmentation and reduce the RSS of Go applications.
//
// Scavenging in Go happens on two fronts: there's the background
// (asynchronous) scavenger and the heap-growth (synchronous) scavenger.
//
// The former happens on a goroutine much like the background sweeper which is
// soft-capped at using scavengePercent of the mutator's time, based on
// order-of-magnitude estimates of the costs of scavenging. The background
// scavenger's primary goal is to bring the estimated heap RSS of the
// application down to a goal.
//
// That goal is defined as:
//   (retainExtraPercent+100) / 100 * (heapGoal / lastHeapGoal) * last_heap_inuse
//
// Essentially, we wish to have the application's RSS track the heap goal, but
// the heap goal is defined in terms of bytes of objects, rather than pages like
// RSS. As a result, we need to take into account for fragmentation internal to
// spans. heapGoal / lastHeapGoal defines the ratio between the current heap goal
// and the last heap goal, which tells us by how much the heap is growing and
// shrinking. We estimate what the heap will grow to in terms of pages by taking
// this ratio and multiplying it by heap_inuse at the end of the last GC, which
// allows us to account for this additional fragmentation. Note that this
// procedure makes the assumption that the degree of fragmentation won't change
// dramatically over the next GC cycle. Overestimating the amount of
// fragmentation simply results in higher memory use, which will be accounted
// for by the next pacing up date. Underestimating the fragmentation however
// could lead to performance degradation. Handling this case is not within the
// scope of the scavenger. Situations where the amount of fragmentation balloons
// over the course of a single GC cycle should be considered pathologies,
// flagged as bugs, and fixed appropriately.
//
// An additional factor of retainExtraPercent is added as a buffer to help ensure
// that there's more unscavenged memory to allocate out of, since each allocation
// out of scavenged memory incurs a potentially expensive page fault.
//
// The goal is updated after each GC and the scavenger's pacing parameters
// (which live in mheap_) are updated to match. The pacing parameters work much
// like the background sweeping parameters. The parameters define a line whose
// horizontal axis is time and vertical axis is estimated heap RSS, and the
// scavenger attempts to stay below that line at all times.
//
// The synchronous heap-growth scavenging happens whenever the heap grows in
// size, for some definition of heap-growth. The intuition behind this is that
// the application had to grow the heap because existing fragments were
// not sufficiently large to satisfy a page-level memory allocation, so we
// scavenge those fragments eagerly to offset the growth in RSS that results.

package runtime

import (
	"internal/goos"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

const (
	// The background scavenger is paced according to these parameters.
	//
	// scavengePercent represents the portion of mutator time we're willing
	// to spend on scavenging in percent.
	scavengePercent = 1 // 1%

	// retainExtraPercent represents the amount of memory over the heap goal
	// that the scavenger should keep as a buffer space for the allocator.
	//
	// The purpose of maintaining this overhead is to have a greater pool of
	// unscavenged memory available for allocation (since using scavenged memory
	// incurs an additional cost), to account for heap fragmentation and
	// the ever-changing layout of the heap.
	retainExtraPercent = 10

	// maxPagesPerPhysPage is the maximum number of supported runtime pages per
	// physical page, based on maxPhysPageSize.
	maxPagesPerPhysPage = maxPhysPageSize / pageSize

	// scavengeCostRatio is the approximate ratio between the costs of using previously
	// scavenged memory and scavenging memory.
	//
	// For most systems the cost of scavenging greatly outweighs the costs
	// associated with using scavenged memory, making this constant 0. On other systems
	// (especially ones where "sysUsed" is not just a no-op) this cost is non-trivial.
	//
	// This ratio is used as part of multiplicative factor to help the scavenger account
	// for the additional costs of using scavenged memory in its pacing.
	scavengeCostRatio = 0.7 * (goos.IsDarwin + goos.IsIos)

	// scavengeReservationShards determines the amount of memory the scavenger
	// should reserve for scavenging at a time. Specifically, the amount of
	// memory reserved is (heap size in bytes) / scavengeReservationShards.
	scavengeReservationShards = 64
)

// heapRetained returns an estimate of the current heap RSS.
func heapRetained() uint64 {
	return memstats.heap_sys.load() - atomic.Load64(&memstats.heap_released)
}

// gcPaceScavenger updates the scavenger's pacing, particularly
// its rate and RSS goal. For this, it requires the current heapGoal,
// and the heapGoal for the previous GC cycle.
//
// The RSS goal is based on the current heap goal with a small overhead
// to accommodate non-determinism in the allocator.
//
// The pacing is based on scavengePageRate, which applies to both regular and
// huge pages. See that constant for more information.
//
// Must be called whenever GC pacing is updated.
//
// mheap_.lock must be held or the world must be stopped.
func gcPaceScavenger(heapGoal, lastHeapGoal uint64) {
	assertWorldStoppedOrLockHeld(&mheap_.lock)

	// If we're called before the first GC completed, disable scavenging.
	// We never scavenge before the 2nd GC cycle anyway (we don't have enough
	// information about the heap yet) so this is fine, and avoids a fault
	// or garbage data later.
	if lastHeapGoal == 0 {
		atomic.Store64(&mheap_.scavengeGoal, ^uint64(0))
		return
	}
	// Compute our scavenging goal.
	goalRatio := float64(heapGoal) / float64(lastHeapGoal)
	retainedGoal := uint64(float64(memstats.last_heap_inuse) * goalRatio)
	// Add retainExtraPercent overhead to retainedGoal. This calculation
	// looks strange but the purpose is to arrive at an integer division
	// (e.g. if retainExtraPercent = 12.5, then we get a divisor of 8)
	// that also avoids the overflow from a multiplication.
	retainedGoal += retainedGoal / (1.0 / (retainExtraPercent / 100.0))
	// Align it to a physical page boundary to make the following calculations
	// a bit more exact.
	retainedGoal = (retainedGoal + uint64(physPageSize) - 1) &^ (uint64(physPageSize) - 1)

	// Represents where we are now in the heap's contribution to RSS in bytes.
	//
	// Guaranteed to always be a multiple of physPageSize on systems where
	// physPageSize <= pageSize since we map heap_sys at a rate larger than
	// any physPageSize and released memory in multiples of the physPageSize.
	//
	// However, certain functions recategorize heap_sys as other stats (e.g.
	// stack_sys) and this happens in multiples of pageSize, so on systems
	// where physPageSize > pageSize the calculations below will not be exact.
	// Generally this is OK since we'll be off by at most one regular
	// physical page.
	retainedNow := heapRetained()

	// If we're already below our goal, or within one page of our goal, then disable
	// the background scavenger. We disable the background scavenger if there's
	// less than one physical page of work to do because it's not worth it.
	if retainedNow <= retainedGoal || retainedNow-retainedGoal < uint64(physPageSize) {
		atomic.Store64(&mheap_.scavengeGoal, ^uint64(0))
		return
	}
	atomic.Store64(&mheap_.scavengeGoal, retainedGoal)
}

const (
	// It doesn't really matter what value we start at, but we can't be zero, because
	// that'll cause divide-by-zero issues. Pick something conservative which we'll
	// also use as a fallback.
	startingScavSleepRatio = 0.001

	// Spend at least 1 ms scavenging, otherwise the corresponding
	// sleep time to maintain our desired utilization is too low to
	// be reliable.
	minScavWorkTime = 1e6
)

// Sleep/wait state of the background scavenger.
var scavenger scavengerState

type scavengerState struct {
	// lock protects all fields below.
	lock mutex

	// g is the goroutine the scavenger is bound to.
	g *g

	// parked is whether or not the scavenger is parked.
	parked bool

	// timer is the timer used for the scavenger to sleep.
	timer *timer

	// sysmonWake signals to sysmon that it should wake the scavenger.
	sysmonWake atomic.Uint32

	// targetCPUFraction is the target CPU overhead for the scavenger.
	targetCPUFraction float64

	// sleepRatio is the ratio of time spent doing scavenging work to
	// time spent sleeping. This is used to decide how long the scavenger
	// should sleep for in between batches of work. It is set by
	// critSleepController in order to maintain a CPU overhead of
	// targetCPUFraction.
	//
	// Lower means more sleep, higher means more aggressive scavenging.
	sleepRatio float64

	// sleepController controls sleepRatio.
	//
	// See sleepRatio for more details.
	sleepController piController

	// cooldown is the time left in nanoseconds during which we avoid
	// using the controller and we hold sleepRatio at a conservative
	// value. Used if the controller's assumptions fail to hold.
	controllerCooldown int64

	// printControllerReset instructs printScavTrace to signal that
	// the controller was reset.
	printControllerReset bool

	// sleepStub is a stub used for testing to avoid actually having
	// the scavenger sleep.
	//
	// Unlike the other stubs, this is not populated if left nil
	// Instead, it is called when non-nil because any valid implementation
	// of this function basically requires closing over this scavenger
	// state, and allocating a closure is not allowed in the runtime as
	// a matter of policy.
	sleepStub func(n int64) int64

	// scavenge is a function that scavenges n bytes of memory.
	// Returns how many bytes of memory it actually scavenged, as
	// well as the time it took in nanoseconds. Usually mheap.pages.scavenge
	// with nanotime called around it, but stubbed out for testing.
	// Like mheap.pages.scavenge, if it scavenges less than n bytes of
	// memory, the caller may assume the heap is exhausted of scavengable
	// memory for now.
	//
	// If this is nil, it is populated with the real thing in init.
	scavenge func(n uintptr) (uintptr, int64)

	// shouldStop is a callback called in the work loop and provides a
	// point that can force the scavenger to stop early, for example because
	// the scavenge policy dictates too much has been scavenged already.
	//
	// If this is nil, it is populated with the real thing in init.
	shouldStop func() bool

	// gomaxprocs returns the current value of gomaxprocs. Stub for testing.
	//
	// If this is nil, it is populated with the real thing in init.
	gomaxprocs func() int32
}

// init initializes a scavenger state and wires to the current G.
//
// Must be called from a regular goroutine that can allocate.
func (s *scavengerState) init() {
	if s.g != nil {
		throw("scavenger state is already wired")
	}
	lockInit(&s.lock, lockRankScavenge)
	s.g = getg()

	s.timer = new(timer)
	s.timer.arg = s
	s.timer.f = func(s any, _ uintptr) {
		s.(*scavengerState).wake()
	}

	// input: fraction of CPU time actually used.
	// setpoint: ideal CPU fraction.
	// output: ratio of time worked to time slept (determines sleep time).
	//
	// The output of this controller is somewhat indirect to what we actually
	// want to achieve: how much time to sleep for. The reason for this definition
	// is to ensure that the controller's outputs have a direct relationship with
	// its inputs (as opposed to an inverse relationship), making it somewhat
	// easier to reason about for tuning purposes.
	s.sleepController = piController{
		// Tuned loosely via Ziegler-Nichols process.
		kp: 0.3375,
		ti: 3.2e6,
		tt: 1e9, // 1 second reset time.

		// These ranges seem wide, but we want to give the controller plenty of
		// room to hunt for the optimal value.
		min: 0.001,  // 1:1000
		max: 1000.0, // 1000:1
	}
	s.sleepRatio = startingScavSleepRatio

	// Install real functions if stubs aren't present.
	if s.scavenge == nil {
		s.scavenge = func(n uintptr) (uintptr, int64) {
			start := nanotime()
			r := mheap_.pages.scavenge(n)
			end := nanotime()
			if start >= end {
				return r, 0
			}
			return r, end - start
		}
	}
	if s.shouldStop == nil {
		s.shouldStop = func() bool {
			// If background scavenging is disabled or if there's no work to do just stop.
			return heapRetained() <= atomic.Load64(&mheap_.scavengeGoal)
		}
	}
	if s.gomaxprocs == nil {
		s.gomaxprocs = func() int32 {
			return gomaxprocs
		}
	}
}

// park parks the scavenger goroutine.
func (s *scavengerState) park() {
	lock(&s.lock)
	if getg() != s.g {
		throw("tried to park scavenger from another goroutine")
	}
	s.parked = true
	goparkunlock(&s.lock, waitReasonGCScavengeWait, traceEvGoBlock, 2)
}

// ready signals to sysmon that the scavenger should be awoken.
func (s *scavengerState) ready() {
	s.sysmonWake.Store(1)
}

// wake immediately unparks the scavenger if necessary.
//
// Safe to run without a P.
func (s *scavengerState) wake() {
	lock(&s.lock)
	if s.parked {
		// Unset sysmonWake, since the scavenger is now being awoken.
		s.sysmonWake.Store(0)

		// s.parked is unset to prevent a double wake-up.
		s.parked = false

		// Ready the goroutine by injecting it. We use injectglist instead
		// of ready or goready in order to allow us to run this function
		// without a P. injectglist also avoids placing the goroutine in
		// the current P's runnext slot, which is desirable to prevent
		// the scavenger from interfering with user goroutine scheduling
		// too much.
		var list gList
		list.push(s.g)
		injectglist(&list)
	}
	unlock(&s.lock)
}

// sleep puts the scavenger to sleep based on the amount of time that it worked
// in nanoseconds.
//
// Note that this function should only be called by the scavenger.
//
// The scavenger may be woken up earlier by a pacing change, and it may not go
// to sleep at all if there's a pending pacing change.
func (s *scavengerState) sleep(worked float64) {
	lock(&s.lock)
	if getg() != s.g {
		throw("tried to sleep scavenger from another goroutine")
	}

	if worked < minScavWorkTime {
		// This means there wasn't enough work to actually fill up minScavWorkTime.
		// That's fine; we shouldn't try to do anything with this information
		// because it's going result in a short enough sleep request that things
		// will get messy. Just assume we did at least this much work.
		// All this means is that we'll sleep longer than we otherwise would have.
		worked = minScavWorkTime
	}

	// Multiply the critical time by 1 + the ratio of the costs of using
	// scavenged memory vs. scavenging memory. This forces us to pay down
	// the cost of reusing this memory eagerly by sleeping for a longer period
	// of time and scavenging less frequently. More concretely, we avoid situations
	// where we end up scavenging so often that we hurt allocation performance
	// because of the additional overheads of using scavenged memory.
	worked *= 1 + scavengeCostRatio

	// sleepTime is the amount of time we're going to sleep, based on the amount
	// of time we worked, and the sleepRatio.
	sleepTime := int64(worked / s.sleepRatio)

	var slept int64
	if s.sleepStub == nil {
		// Set the timer.
		//
		// This must happen here instead of inside gopark
		// because we can't close over any variables without
		// failing escape analysis.
		start := nanotime()
		resetTimer(s.timer, start+sleepTime)

		// Mark ourselves as asleep and go to sleep.
		s.parked = true
		goparkunlock(&s.lock, waitReasonSleep, traceEvGoSleep, 2)

		// How long we actually slept for.
		slept = nanotime() - start

		lock(&s.lock)
		// Stop the timer here because s.wake is unable to do it for us.
		// We don't really care if we succeed in stopping the timer. One
		// reason we might fail is that we've already woken up, but the timer
		// might be in the process of firing on some other P; essentially we're
		// racing with it. That's totally OK. Double wake-ups are perfectly safe.
		stopTimer(s.timer)
		unlock(&s.lock)
	} else {
		unlock(&s.lock)
		slept = s.sleepStub(sleepTime)
	}

	// Stop here if we're cooling down from the controller.
	if s.controllerCooldown > 0 {
		// worked and slept aren't exact measures of time, but it's OK to be a bit
		// sloppy here. We're just hoping we're avoiding some transient bad behavior.
		t := slept + int64(worked)
		if t > s.controllerCooldown {
			s.controllerCooldown = 0
		} else {
			s.controllerCooldown -= t
		}
		return
	}

	// idealFraction is the ideal % of overall application CPU time that we
	// spend scavenging.
	idealFraction := float64(scavengePercent) / 100.0

	// Calculate the CPU time spent.
	//
	// This may be slightly inaccurate with respect to GOMAXPROCS, but we're
	// recomputing this often enough relative to GOMAXPROCS changes in general
	// (it only changes when the world is stopped, and not during a GC) that
	// that small inaccuracy is in the noise.
	cpuFraction := worked / ((float64(slept) + worked) * float64(s.gomaxprocs()))

	// Update the critSleepRatio, adjusting until we reach our ideal fraction.
	var ok bool
	s.sleepRatio, ok = s.sleepController.next(cpuFraction, idealFraction, float64(slept)+worked)
	if !ok {
		// The core assumption of the controller, that we can get a proportional
		// response, broke down. This may be transient, so temporarily switch to
		// sleeping a fixed, conservative amount.
		s.sleepRatio = startingScavSleepRatio
		s.controllerCooldown = 5e9 // 5 seconds.

		// Signal the scav trace printer to output this.
		s.controllerFailed()
	}
}

// controllerFailed indicates that the scavenger's scheduling
// controller failed.
func (s *scavengerState) controllerFailed() {
	lock(&s.lock)
	s.printControllerReset = true
	unlock(&s.lock)
}

// run is the body of the main scavenging loop.
//
// Returns the number of bytes released and the estimated time spent
// releasing those bytes.
//
// Must be run on the scavenger goroutine.
func (s *scavengerState) run() (released uintptr, worked float64) {
	lock(&s.lock)
	if getg() != s.g {
		throw("tried to run scavenger from another goroutine")
	}
	unlock(&s.lock)

	for worked < minScavWorkTime {
		// If something from outside tells us to stop early, stop.
		if s.shouldStop() {
			break
		}

		// scavengeQuantum is the amount of memory we try to scavenge
		// in one go. A smaller value means the scavenger is more responsive
		// to the scheduler in case of e.g. preemption. A larger value means
		// that the overheads of scavenging are better amortized, so better
		// scavenging throughput.
		//
		// The current value is chosen assuming a cost of ~10µs/physical page
		// (this is somewhat pessimistic), which implies a worst-case latency of
		// about 160µs for 4 KiB physical pages. The current value is biased
		// toward latency over throughput.
		const scavengeQuantum = 64 << 10

		// Accumulate the amount of time spent scavenging.
		r, duration := s.scavenge(scavengeQuantum)

		// On some platforms we may see end >= start if the time it takes to scavenge
		// memory is less than the minimum granularity of its clock (e.g. Windows) or
		// due to clock bugs.
		//
		// In this case, just assume scavenging takes 10 µs per regular physical page
		// (determined empirically), and conservatively ignore the impact of huge pages
		// on timing.
		const approxWorkedNSPerPhysicalPage = 10e3
		if duration == 0 {
			worked += approxWorkedNSPerPhysicalPage * float64(r/physPageSize)
		} else {
			// TODO(mknyszek): If duration is small compared to worked, it could be
			// rounded down to zero. Probably not a problem in practice because the
			// values are all within a few orders of magnitude of each other but maybe
			// worth worrying about.
			worked += float64(duration)
		}
		released += r

		// scavenge does not return until it either finds the requisite amount of
		// memory to scavenge, or exhausts the heap. If we haven't found enough
		// to scavenge, then the heap must be exhausted.
		if r < scavengeQuantum {
			break
		}
		// When using fake time just do one loop.
		if faketime != 0 {
			break
		}
	}
	if released > 0 && released < physPageSize {
		// If this happens, it means that we may have attempted to release part
		// of a physical page, but the likely effect of that is that it released
		// the whole physical page, some of which may have still been in-use.
		// This could lead to memory corruption. Throw.
		throw("released less than one physical page of memory")
	}
	return
}

// Background scavenger.
//
// The background scavenger maintains the RSS of the application below
// the line described by the proportional scavenging statistics in
// the mheap struct.
func bgscavenge(c chan int) {
	scavenger.init()

	c <- 1
	scavenger.park()

	for {
		released, workTime := scavenger.run()
		if released == 0 {
			scavenger.park()
			continue
		}
		atomic.Xadduintptr(&mheap_.pages.scav.released, released)
		scavenger.sleep(workTime)
	}
}

// scavenge scavenges nbytes worth of free pages, starting with the
// highest address first. Successive calls continue from where it left
// off until the heap is exhausted. Call scavengeStartGen to bring it
// back to the top of the heap.
//
// Returns the amount of memory scavenged in bytes.
//
// scavenge always tries to scavenge nbytes worth of memory, and will
// only fail to do so if the heap is exhausted for now.
func (p *pageAlloc) scavenge(nbytes uintptr) uintptr {
	var (
		addrs addrRange
		gen   uint32
	)
	released := uintptr(0)
	for released < nbytes {
		if addrs.size() == 0 {
			if addrs, gen = p.scavengeReserve(); addrs.size() == 0 {
				break
			}
		}
		systemstack(func() {
			r, a := p.scavengeOne(addrs, nbytes-released)
			released += r
			addrs = a
		})
	}
	// Only unreserve the space which hasn't been scavenged or searched
	// to ensure we always make progress.
	p.scavengeUnreserve(addrs, gen)
	return released
}

// printScavTrace prints a scavenge trace line to standard error.
//
// released should be the amount of memory released since the last time this
// was called, and forced indicates whether the scavenge was forced by the
// application.
//
// scavenger.lock must be held.
func printScavTrace(gen uint32, released uintptr, forced bool) {
	assertLockHeld(&scavenger.lock)

	printlock()
	print("scav ", gen, " ",
		released>>10, " KiB work, ",
		atomic.Load64(&memstats.heap_released)>>10, " KiB total, ",
		(atomic.Load64(&memstats.heap_inuse)*100)/heapRetained(), "% util",
	)
	if forced {
		print(" (forced)")
	} else if scavenger.printControllerReset {
		print(" [controller reset]")
		scavenger.printControllerReset = false
	}
	println()
	printunlock()
}

// scavengeStartGen starts a new scavenge generation, resetting
// the scavenger's search space to the full in-use address space.
//
// p.mheapLock must be held.
//
// Must run on the system stack because p.mheapLock must be held.
//
//go:systemstack
func (p *pageAlloc) scavengeStartGen() {
	assertLockHeld(p.mheapLock)

	lock(&p.scav.lock)
	if debug.scavtrace > 0 {
		printScavTrace(p.scav.gen, atomic.Loaduintptr(&p.scav.released), false)
	}
	p.inUse.cloneInto(&p.scav.inUse)

	// Pick the new starting address for the scavenger cycle.
	var startAddr offAddr
	if p.scav.scavLWM.lessThan(p.scav.freeHWM) {
		// The "free" high watermark exceeds the "scavenged" low watermark,
		// so there are free scavengable pages in parts of the address space
		// that the scavenger already searched, the high watermark being the
		// highest one. Pick that as our new starting point to ensure we
		// see those pages.
		startAddr = p.scav.freeHWM
	} else {
		// The "free" high watermark does not exceed the "scavenged" low
		// watermark. This means the allocator didn't free any memory in
		// the range we scavenged last cycle, so we might as well continue
		// scavenging from where we were.
		startAddr = p.scav.scavLWM
	}
	p.scav.inUse.removeGreaterEqual(startAddr.addr())

	// reservationBytes may be zero if p.inUse.totalBytes is small, or if
	// scavengeReservationShards is large. This case is fine as the scavenger
	// will simply be turned off, but it does mean that scavengeReservationShards,
	// in concert with pallocChunkBytes, dictates the minimum heap size at which
	// the scavenger triggers. In practice this minimum is generally less than an
	// arena in size, so virtually every heap has the scavenger on.
	p.scav.reservationBytes = alignUp(p.inUse.totalBytes, pallocChunkBytes) / scavengeReservationShards
	p.scav.gen++
	atomic.Storeuintptr(&p.scav.released, 0)
	p.scav.freeHWM = minOffAddr
	p.scav.scavLWM = maxOffAddr
	unlock(&p.scav.lock)
}

// scavengeReserve reserves a contiguous range of the address space
// for scavenging. The maximum amount of space it reserves is proportional
// to the size of the heap. The ranges are reserved from the high addresses
// first.
//
// Returns the reserved range and the scavenge generation number for it.
func (p *pageAlloc) scavengeReserve() (addrRange, uint32) {
	lock(&p.scav.lock)
	gen := p.scav.gen

	// Start by reserving the minimum.
	r := p.scav.inUse.removeLast(p.scav.reservationBytes)

	// Return early if the size is zero; we don't want to use
	// the bogus address below.
	if r.size() == 0 {
		unlock(&p.scav.lock)
		return r, gen
	}

	// The scavenger requires that base be aligned to a
	// palloc chunk because that's the unit of operation for
	// the scavenger, so align down, potentially extending
	// the range.
	newBase := alignDown(r.base.addr(), pallocChunkBytes)

	// Remove from inUse however much extra we just pulled out.
	p.scav.inUse.removeGreaterEqual(newBase)
	unlock(&p.scav.lock)

	r.base = offAddr{newBase}
	return r, gen
}

// scavengeUnreserve returns an unscavenged portion of a range that was
// previously reserved with scavengeReserve.
func (p *pageAlloc) scavengeUnreserve(r addrRange, gen uint32) {
	if r.size() == 0 {
		return
	}
	if r.base.addr()%pallocChunkBytes != 0 {
		throw("unreserving unaligned region")
	}
	lock(&p.scav.lock)
	if gen == p.scav.gen {
		p.scav.inUse.add(r)
	}
	unlock(&p.scav.lock)
}

// scavengeOne walks over address range work until it finds
// a contiguous run of pages to scavenge. It will try to scavenge
// at most max bytes at once, but may scavenge more to avoid
// breaking huge pages. Once it scavenges some memory it returns
// how much it scavenged in bytes.
//
// Returns the number of bytes scavenged and the part of work
// which was not yet searched.
//
// work's base address must be aligned to pallocChunkBytes.
//
// Must run on the systemstack because it acquires p.mheapLock.
//
//go:systemstack
func (p *pageAlloc) scavengeOne(work addrRange, max uintptr) (uintptr, addrRange) {
	// Defensively check if we've received an empty address range.
	// If so, just return.
	if work.size() == 0 {
		// Nothing to do.
		return 0, work
	}
	// Check the prerequisites of work.
	if work.base.addr()%pallocChunkBytes != 0 {
		throw("scavengeOne called with unaligned work region")
	}
	// Calculate the maximum number of pages to scavenge.
	//
	// This should be alignUp(max, pageSize) / pageSize but max can and will
	// be ^uintptr(0), so we need to be very careful not to overflow here.
	// Rather than use alignUp, calculate the number of pages rounded down
	// first, then add back one if necessary.
	maxPages := max / pageSize
	if max%pageSize != 0 {
		maxPages++
	}

	// Calculate the minimum number of pages we can scavenge.
	//
	// Because we can only scavenge whole physical pages, we must
	// ensure that we scavenge at least minPages each time, aligned
	// to minPages*pageSize.
	minPages := physPageSize / pageSize
	if minPages < 1 {
		minPages = 1
	}

	// Fast path: check the chunk containing the top-most address in work.
	if r, w := p.scavengeOneFast(work, minPages, maxPages); r != 0 {
		return r, w
	} else {
		work = w
	}

	// findCandidate finds the next scavenge candidate in work optimistically.
	//
	// Returns the candidate chunk index and true on success, and false on failure.
	//
	// The heap need not be locked.
	findCandidate := func(work addrRange) (chunkIdx, bool) {
		// Iterate over this work's chunks.
		for i := chunkIndex(work.limit.addr() - 1); i >= chunkIndex(work.base.addr()); i-- {
			// If this chunk is totally in-use or has no unscavenged pages, don't bother
			// doing a more sophisticated check.
			//
			// Note we're accessing the summary and the chunks without a lock, but
			// that's fine. We're being optimistic anyway.

			// Check quickly if there are enough free pages at all.
			if p.summary[len(p.summary)-1][i].max() < uint(minPages) {
				continue
			}

			// Run over the chunk looking harder for a candidate. Again, we could
			// race with a lot of different pieces of code, but we're just being
			// optimistic. Make sure we load the l2 pointer atomically though, to
			// avoid races with heap growth. It may or may not be possible to also
			// see a nil pointer in this case if we do race with heap growth, but
			// just defensively ignore the nils. This operation is optimistic anyway.
			l2 := (*[1 << pallocChunksL2Bits]pallocData)(atomic.Loadp(unsafe.Pointer(&p.chunks[i.l1()])))
			if l2 != nil && l2[i.l2()].hasScavengeCandidate(minPages) {
				return i, true
			}
		}
		return 0, false
	}

	// Slow path: iterate optimistically over the in-use address space
	// looking for any free and unscavenged page. If we think we see something,
	// lock and verify it!
	for work.size() != 0 {

		// Search for the candidate.
		candidateChunkIdx, ok := findCandidate(work)
		if !ok {
			// We didn't find a candidate, so we're done.
			work.limit = work.base
			break
		}

		// Lock, so we can verify what we found.
		lock(p.mheapLock)

		// Find, verify, and scavenge if we can.
		chunk := p.chunkOf(candidateChunkIdx)
		base, npages := chunk.findScavengeCandidate(pallocChunkPages-1, minPages, maxPages)
		if npages > 0 {
			work.limit = offAddr{p.scavengeRangeLocked(candidateChunkIdx, base, npages)}
			unlock(p.mheapLock)
			return uintptr(npages) * pageSize, work
		}
		unlock(p.mheapLock)

		// We were fooled, so let's continue from where we left off.
		work.limit = offAddr{chunkBase(candidateChunkIdx)}
	}
	return 0, work
}

// scavengeOneFast is the fast path for scavengeOne, which just checks the top
// chunk of work for some pages to scavenge.
//
// Must run on the system stack because it acquires the heap lock.
//
//go:systemstack
func (p *pageAlloc) scavengeOneFast(work addrRange, minPages, maxPages uintptr) (uintptr, addrRange) {
	maxAddr := work.limit.addr() - 1
	maxChunk := chunkIndex(maxAddr)

	lock(p.mheapLock)
	if p.summary[len(p.summary)-1][maxChunk].max() >= uint(minPages) {
		// We only bother looking for a candidate if there at least
		// minPages free pages at all.
		base, npages := p.chunkOf(maxChunk).findScavengeCandidate(chunkPageIndex(maxAddr), minPages, maxPages)

		// If we found something, scavenge it and return!
		if npages != 0 {
			work.limit = offAddr{p.scavengeRangeLocked(maxChunk, base, npages)}
			unlock(p.mheapLock)
			return uintptr(npages) * pageSize, work
		}
	}
	unlock(p.mheapLock)

	// Update the limit to reflect the fact that we checked maxChunk already.
	work.limit = offAddr{chunkBase(maxChunk)}
	return 0, work
}

// scavengeRangeLocked scavenges the given region of memory.
// The region of memory is described by its chunk index (ci),
// the starting page index of the region relative to that
// chunk (base), and the length of the region in pages (npages).
//
// Returns the base address of the scavenged region.
//
// p.mheapLock must be held. Unlocks p.mheapLock but reacquires
// it before returning. Must be run on the systemstack as a result.
//
//go:systemstack
func (p *pageAlloc) scavengeRangeLocked(ci chunkIdx, base, npages uint) uintptr {
	assertLockHeld(p.mheapLock)

	// Compute the full address for the start of the range.
	addr := chunkBase(ci) + uintptr(base)*pageSize

	// Mark the range we're about to scavenge as allocated, because
	// we don't want any allocating goroutines to grab it while
	// the scavenging is in progress.
	if scav := p.allocRange(addr, uintptr(npages)); scav != 0 {
		throw("double scavenge")
	}

	// With that done, it's safe to unlock.
	unlock(p.mheapLock)

	// Update the scavenge low watermark.
	lock(&p.scav.lock)
	if oAddr := (offAddr{addr}); oAddr.lessThan(p.scav.scavLWM) {
		p.scav.scavLWM = oAddr
	}
	unlock(&p.scav.lock)

	if !p.test {
		// Only perform the actual scavenging if we're not in a test.
		// It's dangerous to do so otherwise.
		sysUnused(unsafe.Pointer(addr), uintptr(npages)*pageSize)

		// Update global accounting only when not in test, otherwise
		// the runtime's accounting will be wrong.
		nbytes := int64(npages) * pageSize
		atomic.Xadd64(&memstats.heap_released, nbytes)

		// Update consistent accounting too.
		stats := memstats.heapStats.acquire()
		atomic.Xaddint64(&stats.committed, -nbytes)
		atomic.Xaddint64(&stats.released, nbytes)
		memstats.heapStats.release()
	}

	// Relock the heap, because now we need to make these pages
	// available allocation. Free them back to the page allocator.
	lock(p.mheapLock)
	p.free(addr, uintptr(npages), true)

	// Mark the range as scavenged.
	p.chunkOf(ci).scavenged.setRange(base, npages)
	return addr
}

// fillAligned returns x but with all zeroes in m-aligned
// groups of m bits set to 1 if any bit in the group is non-zero.
//
// For example, fillAligned(0x0100a3, 8) == 0xff00ff.
//
// Note that if m == 1, this is a no-op.
//
// m must be a power of 2 <= maxPagesPerPhysPage.
func fillAligned(x uint64, m uint) uint64 {
	apply := func(x uint64, c uint64) uint64 {
		// The technique used it here is derived from
		// https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
		// and extended for more than just bytes (like nibbles
		// and uint16s) by using an appropriate constant.
		//
		// To summarize the technique, quoting from that page:
		// "[It] works by first zeroing the high bits of the [8]
		// bytes in the word. Subsequently, it adds a number that
		// will result in an overflow to the high bit of a byte if
		// any of the low bits were initially set. Next the high
		// bits of the original word are ORed with these values;
		// thus, the high bit of a byte is set iff any bit in the
		// byte was set. Finally, we determine if any of these high
		// bits are zero by ORing with ones everywhere except the
		// high bits and inverting the result."
		return ^((((x & c) + c) | x) | c)
	}
	// Transform x to contain a 1 bit at the top of each m-aligned
	// group of m zero bits.
	switch m {
	case 1:
		return x
	case 2:
		x = apply(x, 0x5555555555555555)
	case 4:
		x = apply(x, 0x7777777777777777)
	case 8:
		x = apply(x, 0x7f7f7f7f7f7f7f7f)
	case 16:
		x = apply(x, 0x7fff7fff7fff7fff)
	case 32:
		x = apply(x, 0x7fffffff7fffffff)
	case 64: // == maxPagesPerPhysPage
		x = apply(x, 0x7fffffffffffffff)
	default:
		throw("bad m value")
	}
	// Now, the top bit of each m-aligned group in x is set
	// that group was all zero in the original x.

	// From each group of m bits subtract 1.
	// Because we know only the top bits of each
	// m-aligned group are set, we know this will
	// set each group to have all the bits set except
	// the top bit, so just OR with the original
	// result to set all the bits.
	return ^((x - (x >> (m - 1))) | x)
}

// hasScavengeCandidate returns true if there's any min-page-aligned groups of
// min pages of free-and-unscavenged memory in the region represented by this
// pallocData.
//
// min must be a non-zero power of 2 <= maxPagesPerPhysPage.
func (m *pallocData) hasScavengeCandidate(min uintptr) bool {
	if min&(min-1) != 0 || min == 0 {
		print("runtime: min = ", min, "\n")
		throw("min must be a non-zero power of 2")
	} else if min > maxPagesPerPhysPage {
		print("runtime: min = ", min, "\n")
		throw("min too large")
	}

	// The goal of this search is to see if the chunk contains any free and unscavenged memory.
	for i := len(m.scavenged) - 1; i >= 0; i-- {
		// 1s are scavenged OR non-free => 0s are unscavenged AND free
		//
		// TODO(mknyszek): Consider splitting up fillAligned into two
		// functions, since here we technically could get by with just
		// the first half of its computation. It'll save a few instructions
		// but adds some additional code complexity.
		x := fillAligned(m.scavenged[i]|m.pallocBits[i], uint(min))

		// Quickly skip over chunks of non-free or scavenged pages.
		if x != ^uint64(0) {
			return true
		}
	}
	return false
}

// findScavengeCandidate returns a start index and a size for this pallocData
// segment which represents a contiguous region of free and unscavenged memory.
//
// searchIdx indicates the page index within this chunk to start the search, but
// note that findScavengeCandidate searches backwards through the pallocData. As a
// a result, it will return the highest scavenge candidate in address order.
//
// min indicates a hard minimum size and alignment for runs of pages. That is,
// findScavengeCandidate will not return a region smaller than min pages in size,
// or that is min pages or greater in size but not aligned to min. min must be
// a non-zero power of 2 <= maxPagesPerPhysPage.
//
// max is a hint for how big of a region is desired. If max >= pallocChunkPages, then
// findScavengeCandidate effectively returns entire free and unscavenged regions.
// If max < pallocChunkPages, it may truncate the returned region such that size is
// max. However, findScavengeCandidate may still return a larger region if, for
// example, it chooses to preserve huge pages, or if max is not aligned to min (it
// will round up). That is, even if max is small, the returned size is not guaranteed
// to be equal to max. max is allowed to be less than min, in which case it is as if
// max == min.
func (m *pallocData) findScavengeCandidate(searchIdx uint, min, max uintptr) (uint, uint) {
	if min&(min-1) != 0 || min == 0 {
		print("runtime: min = ", min, "\n")
		throw("min must be a non-zero power of 2")
	} else if min > maxPagesPerPhysPage {
		print("runtime: min = ", min, "\n")
		throw("min too large")
	}
	// max may not be min-aligned, so we might accidentally truncate to
	// a max value which causes us to return a non-min-aligned value.
	// To prevent this, align max up to a multiple of min (which is always
	// a power of 2). This also prevents max from ever being less than
	// min, unless it's zero, so handle that explicitly.
	if max == 0 {
		max = min
	} else {
		max = alignUp(max, min)
	}

	i := int(searchIdx / 64)
	// Start by quickly skipping over blocks of non-free or scavenged pages.
	for ; i >= 0; i-- {
		// 1s are scavenged OR non-free => 0s are unscavenged AND free
		x := fillAligned(m.scavenged[i]|m.pallocBits[i], uint(min))
		if x != ^uint64(0) {
			break
		}
	}
	if i < 0 {
		// Failed to find any free/unscavenged pages.
		return 0, 0
	}
	// We have something in the 64-bit chunk at i, but it could
	// extend further. Loop until we find the extent of it.

	// 1s are scavenged OR non-free => 0s are unscavenged AND free
	x := fillAligned(m.scavenged[i]|m.pallocBits[i], uint(min))
	z1 := uint(sys.LeadingZeros64(^x))
	run, end := uint(0), uint(i)*64+(64-z1)
	if x<<z1 != 0 {
		// After shifting out z1 bits, we still have 1s,
		// so the run ends inside this word.
		run = uint(sys.LeadingZeros64(x << z1))
	} else {
		// After shifting out z1 bits, we have no more 1s.
		// This means the run extends to the bottom of the
		// word so it may extend into further words.
		run = 64 - z1
		for j := i - 1; j >= 0; j-- {
			x := fillAligned(m.scavenged[j]|m.pallocBits[j], uint(min))
			run += uint(sys.LeadingZeros64(x))
			if x != 0 {
				// The run stopped in this word.
				break
			}
		}
	}

	// Split the run we found if it's larger than max but hold on to
	// our original length, since we may need it later.
	size := run
	if size > uint(max) {
		size = uint(max)
	}
	start := end - size

	// Each huge page is guaranteed to fit in a single palloc chunk.
	//
	// TODO(mknyszek): Support larger huge page sizes.
	// TODO(mknyszek): Consider taking pages-per-huge-page as a parameter
	// so we can write tests for this.
	if physHugePageSize > pageSize && physHugePageSize > physPageSize {
		// We have huge pages, so let's ensure we don't break one by scavenging
		// over a huge page boundary. If the range [start, start+size) overlaps with
		// a free-and-unscavenged huge page, we want to grow the region we scavenge
		// to include that huge page.

		// Compute the huge page boundary above our candidate.
		pagesPerHugePage := uintptr(physHugePageSize / pageSize)
		hugePageAbove := uint(alignUp(uintptr(start), pagesPerHugePage))

		// If that boundary is within our current candidate, then we may be breaking
		// a huge page.
		if hugePageAbove <= end {
			// Compute the huge page boundary below our candidate.
			hugePageBelow := uint(alignDown(uintptr(start), pagesPerHugePage))

			if hugePageBelow >= end-run {
				// We're in danger of breaking apart a huge page since start+size crosses
				// a huge page boundary and rounding down start to the nearest huge
				// page boundary is included in the full run we found. Include the entire
				// huge page in the bound by rounding down to the huge page size.
				size = size + (start - hugePageBelow)
				start = hugePageBelow
			}
		}
	}
	return start, size
}

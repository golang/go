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
// Before we consider what this looks like, we need to split the world into two
// halves. One in which a memory limit is not set, and one in which it is.
//
// For the former, the goal is defined as:
//   (retainExtraPercent+100) / 100 * (heapGoal / lastHeapGoal) * lastHeapInUse
//
// Essentially, we wish to have the application's RSS track the heap goal, but
// the heap goal is defined in terms of bytes of objects, rather than pages like
// RSS. As a result, we need to take into account for fragmentation internal to
// spans. heapGoal / lastHeapGoal defines the ratio between the current heap goal
// and the last heap goal, which tells us by how much the heap is growing and
// shrinking. We estimate what the heap will grow to in terms of pages by taking
// this ratio and multiplying it by heapInUse at the end of the last GC, which
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
// If a memory limit is set, then we wish to pick a scavenge goal that maintains
// that memory limit. For that, we look at total memory that has been committed
// (memstats.mappedReady) and try to bring that down below the limit. In this case,
// we want to give buffer space in the *opposite* direction. When the application
// is close to the limit, we want to make sure we push harder to keep it under, so
// if we target below the memory limit, we ensure that the background scavenger is
// giving the situation the urgency it deserves.
//
// In this case, the goal is defined as:
//    (100-reduceExtraPercent) / 100 * memoryLimit
//
// We compute both of these goals, and check whether either of them have been met.
// The background scavenger continues operating as long as either one of the goals
// has not been met.
//
// The goals are updated after each GC.
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
	// This constant is used when we do not have a memory limit set.
	//
	// The purpose of maintaining this overhead is to have a greater pool of
	// unscavenged memory available for allocation (since using scavenged memory
	// incurs an additional cost), to account for heap fragmentation and
	// the ever-changing layout of the heap.
	retainExtraPercent = 10

	// reduceExtraPercent represents the amount of memory under the limit
	// that the scavenger should target. For example, 5 means we target 95%
	// of the limit.
	//
	// The purpose of shooting lower than the limit is to ensure that, once
	// close to the limit, the scavenger is working hard to maintain it. If
	// we have a memory limit set but are far away from it, there's no harm
	// in leaving up to 100-retainExtraPercent live, and it's more efficient
	// anyway, for the same reasons that retainExtraPercent exists.
	reduceExtraPercent = 5

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
)

// heapRetained returns an estimate of the current heap RSS.
func heapRetained() uint64 {
	return gcController.heapInUse.load() + gcController.heapFree.load()
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
func gcPaceScavenger(memoryLimit int64, heapGoal, lastHeapGoal uint64) {
	assertWorldStoppedOrLockHeld(&mheap_.lock)

	// As described at the top of this file, there are two scavenge goals here: one
	// for gcPercent and one for memoryLimit. Let's handle the latter first because
	// it's simpler.

	// We want to target retaining (100-reduceExtraPercent)% of the heap.
	memoryLimitGoal := uint64(float64(memoryLimit) * (100.0 - reduceExtraPercent))

	// mappedReady is comparable to memoryLimit, and represents how much total memory
	// the Go runtime has committed now (estimated).
	mappedReady := gcController.mappedReady.Load()

	// If we're below the goal already indicate that we don't need the background
	// scavenger for the memory limit. This may seems worrisome at first, but note
	// that the allocator will assist the background scavenger in the face of a memory
	// limit, so we'll be safe even if we stop the scavenger when we shouldn't have.
	if mappedReady <= memoryLimitGoal {
		scavenge.memoryLimitGoal.Store(^uint64(0))
	} else {
		scavenge.memoryLimitGoal.Store(memoryLimitGoal)
	}

	// Now handle the gcPercent goal.

	// If we're called before the first GC completed, disable scavenging.
	// We never scavenge before the 2nd GC cycle anyway (we don't have enough
	// information about the heap yet) so this is fine, and avoids a fault
	// or garbage data later.
	if lastHeapGoal == 0 {
		scavenge.gcPercentGoal.Store(^uint64(0))
		return
	}
	// Compute our scavenging goal.
	goalRatio := float64(heapGoal) / float64(lastHeapGoal)
	gcPercentGoal := uint64(float64(memstats.lastHeapInUse) * goalRatio)
	// Add retainExtraPercent overhead to retainedGoal. This calculation
	// looks strange but the purpose is to arrive at an integer division
	// (e.g. if retainExtraPercent = 12.5, then we get a divisor of 8)
	// that also avoids the overflow from a multiplication.
	gcPercentGoal += gcPercentGoal / (1.0 / (retainExtraPercent / 100.0))
	// Align it to a physical page boundary to make the following calculations
	// a bit more exact.
	gcPercentGoal = (gcPercentGoal + uint64(physPageSize) - 1) &^ (uint64(physPageSize) - 1)

	// Represents where we are now in the heap's contribution to RSS in bytes.
	//
	// Guaranteed to always be a multiple of physPageSize on systems where
	// physPageSize <= pageSize since we map new heap memory at a size larger than
	// any physPageSize and released memory in multiples of the physPageSize.
	//
	// However, certain functions recategorize heap memory as other stats (e.g.
	// stacks) and this happens in multiples of pageSize, so on systems
	// where physPageSize > pageSize the calculations below will not be exact.
	// Generally this is OK since we'll be off by at most one regular
	// physical page.
	heapRetainedNow := heapRetained()

	// If we're already below our goal, or within one page of our goal, then indicate
	// that we don't need the background scavenger for maintaining a memory overhead
	// proportional to the heap goal.
	if heapRetainedNow <= gcPercentGoal || heapRetainedNow-gcPercentGoal < uint64(physPageSize) {
		scavenge.gcPercentGoal.Store(^uint64(0))
	} else {
		scavenge.gcPercentGoal.Store(gcPercentGoal)
	}
}

var scavenge struct {
	// gcPercentGoal is the amount of retained heap memory (measured by
	// heapRetained) that the runtime will try to maintain by returning
	// memory to the OS. This goal is derived from gcController.gcPercent
	// by choosing to retain enough memory to allocate heap memory up to
	// the heap goal.
	gcPercentGoal atomic.Uint64

	// memoryLimitGoal is the amount of memory retained by the runtime (
	// measured by gcController.mappedReady) that the runtime will try to
	// maintain by returning memory to the OS. This goal is derived from
	// gcController.memoryLimit by choosing to target the memory limit or
	// some lower target to keep the scavenger working.
	memoryLimitGoal atomic.Uint64
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
			r := mheap_.pages.scavenge(n, nil)
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
			return heapRetained() <= scavenge.gcPercentGoal.Load() &&
				(!go119MemoryLimitSupport ||
					gcController.mappedReady.Load() <= scavenge.memoryLimitGoal.Load())
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
func (p *pageAlloc) scavenge(nbytes uintptr, shouldStop func() bool) uintptr {
	released := uintptr(0)
	for released < nbytes {
		ci, pageIdx := p.scav.index.find()
		if ci == 0 {
			break
		}
		systemstack(func() {
			released += p.scavengeOne(ci, pageIdx, nbytes-released)
		})
		if shouldStop != nil && shouldStop() {
			break
		}
	}
	return released
}

// printScavTrace prints a scavenge trace line to standard error.
//
// released should be the amount of memory released since the last time this
// was called, and forced indicates whether the scavenge was forced by the
// application.
//
// scavenger.lock must be held.
func printScavTrace(released uintptr, forced bool) {
	assertLockHeld(&scavenger.lock)

	printlock()
	print("scav ",
		released>>10, " KiB work, ",
		gcController.heapReleased.load()>>10, " KiB total, ",
		(gcController.heapInUse.load()*100)/heapRetained(), "% util",
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

// scavengeOne walks over the chunk at chunk index ci and searches for
// a contiguous run of pages to scavenge. It will try to scavenge
// at most max bytes at once, but may scavenge more to avoid
// breaking huge pages. Once it scavenges some memory it returns
// how much it scavenged in bytes.
//
// searchIdx is the page index to start searching from in ci.
//
// Returns the number of bytes scavenged.
//
// Must run on the systemstack because it acquires p.mheapLock.
//
//go:systemstack
func (p *pageAlloc) scavengeOne(ci chunkIdx, searchIdx uint, max uintptr) uintptr {
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

	lock(p.mheapLock)
	if p.summary[len(p.summary)-1][ci].max() >= uint(minPages) {
		// We only bother looking for a candidate if there at least
		// minPages free pages at all.
		base, npages := p.chunkOf(ci).findScavengeCandidate(searchIdx, minPages, maxPages)

		// If we found something, scavenge it and return!
		if npages != 0 {
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

			if !p.test {
				// Only perform the actual scavenging if we're not in a test.
				// It's dangerous to do so otherwise.
				sysUnused(unsafe.Pointer(addr), uintptr(npages)*pageSize)

				// Update global accounting only when not in test, otherwise
				// the runtime's accounting will be wrong.
				nbytes := int64(npages) * pageSize
				gcController.heapReleased.add(nbytes)
				gcController.heapFree.add(-nbytes)

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
			unlock(p.mheapLock)

			return uintptr(npages) * pageSize
		}
	}
	// Mark this chunk as having no free pages.
	p.scav.index.clear(ci)
	unlock(p.mheapLock)

	return 0
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

// scavengeIndex is a structure for efficiently managing which pageAlloc chunks have
// memory available to scavenge.
type scavengeIndex struct {
	// chunks is a bitmap representing the entire address space. Each bit represents
	// a single chunk, and a 1 value indicates the presence of pages available for
	// scavenging. Updates to the bitmap are serialized by the pageAlloc lock.
	//
	// The underlying storage of chunks is platform dependent and may not even be
	// totally mapped read/write. min and max reflect the extent that is safe to access.
	// min is inclusive, max is exclusive.
	//
	// searchAddr is the maximum address (in the offset address space, so we have a linear
	// view of the address space; see mranges.go:offAddr) containing memory available to
	// scavenge. It is a hint to the find operation to avoid O(n^2) behavior in repeated lookups.
	//
	// searchAddr is always inclusive and should be the base address of the highest runtime
	// page available for scavenging.
	//
	// searchAddr is managed by both find and mark.
	//
	// Normally, find monotonically decreases searchAddr as it finds no more free pages to
	// scavenge. However, mark, when marking a new chunk at an index greater than the current
	// searchAddr, sets searchAddr to the *negative* index into chunks of that page. The trick here
	// is that concurrent calls to find will fail to monotonically decrease searchAddr, and so they
	// won't barge over new memory becoming available to scavenge. Furthermore, this ensures
	// that some future caller of find *must* observe the new high index. That caller
	// (or any other racing with it), then makes searchAddr positive before continuing, bringing
	// us back to our monotonically decreasing steady-state.
	//
	// A pageAlloc lock serializes updates between min, max, and searchAddr, so abs(searchAddr)
	// is always guaranteed to be >= min and < max (converted to heap addresses).
	//
	// TODO(mknyszek): Ideally we would use something bigger than a uint8 for faster
	// iteration like uint32, but we lack the bit twiddling intrinsics. We'd need to either
	// copy them from math/bits or fix the fact that we can't import math/bits' code from
	// the runtime due to compiler instrumentation.
	searchAddr atomicOffAddr
	chunks     []atomic.Uint8
	minHeapIdx atomic.Int32
	min, max   atomic.Int32
}

// find returns the highest chunk index that may contain pages available to scavenge.
// It also returns an offset to start searching in the highest chunk.
func (s *scavengeIndex) find() (chunkIdx, uint) {
	searchAddr, marked := s.searchAddr.Load()
	if searchAddr == minOffAddr.addr() {
		// We got a cleared search addr.
		return 0, 0
	}

	// Starting from searchAddr's chunk, and moving down to minHeapIdx,
	// iterate until we find a chunk with pages to scavenge.
	min := s.minHeapIdx.Load()
	searchChunk := chunkIndex(uintptr(searchAddr))
	start := int32(searchChunk / 8)
	for i := start; i >= min; i-- {
		// Skip over irrelevant address space.
		chunks := s.chunks[i].Load()
		if chunks == 0 {
			continue
		}
		// Note that we can't have 8 leading zeroes here because
		// we necessarily skipped that case. So, what's left is
		// an index. If there are no zeroes, we want the 7th
		// index, if 1 zero, the 6th, and so on.
		n := 7 - sys.LeadingZeros8(chunks)
		ci := chunkIdx(uint(i)*8 + uint(n))
		if searchChunk == ci {
			return ci, chunkPageIndex(uintptr(searchAddr))
		}
		// Try to reduce searchAddr to newSearchAddr.
		newSearchAddr := chunkBase(ci) + pallocChunkBytes - pageSize
		if marked {
			// Attempt to be the first one to decrease the searchAddr
			// after an increase. If we fail, that means there was another
			// increase, or somebody else got to it before us. Either way,
			// it doesn't matter. We may lose some performance having an
			// incorrect search address, but it's far more important that
			// we don't miss updates.
			s.searchAddr.StoreUnmark(searchAddr, newSearchAddr)
		} else {
			// Decrease searchAddr.
			s.searchAddr.StoreMin(newSearchAddr)
		}
		return ci, pallocChunkPages - 1
	}
	// Clear searchAddr, because we've exhausted the heap.
	s.searchAddr.Clear()
	return 0, 0
}

// mark sets the inclusive range of chunks between indices start and end as
// containing pages available to scavenge.
//
// Must be serialized with other mark, markRange, and clear calls.
func (s *scavengeIndex) mark(base, limit uintptr) {
	start, end := chunkIndex(base), chunkIndex(limit-pageSize)
	if start == end {
		// Within a chunk.
		mask := uint8(1 << (start % 8))
		s.chunks[start/8].Or(mask)
	} else if start/8 == end/8 {
		// Within the same byte in the index.
		mask := uint8(uint16(1<<(end-start+1))-1) << (start % 8)
		s.chunks[start/8].Or(mask)
	} else {
		// Crosses multiple bytes in the index.
		startAligned := chunkIdx(alignUp(uintptr(start), 8))
		endAligned := chunkIdx(alignDown(uintptr(end), 8))

		// Do the end of the first byte first.
		if width := startAligned - start; width > 0 {
			mask := uint8(uint16(1<<width)-1) << (start % 8)
			s.chunks[start/8].Or(mask)
		}
		// Do the middle aligned sections that take up a whole
		// byte.
		for ci := startAligned; ci < endAligned; ci += 8 {
			s.chunks[ci/8].Store(^uint8(0))
		}
		// Do the end of the last byte.
		//
		// This width check doesn't match the one above
		// for start because aligning down into the endAligned
		// block means we always have at least one chunk in this
		// block (note that end is *inclusive*). This also means
		// that if end == endAligned+n, then what we really want
		// is to fill n+1 chunks, i.e. width n+1. By induction,
		// this is true for all n.
		if width := end - endAligned + 1; width > 0 {
			mask := uint8(uint16(1<<width) - 1)
			s.chunks[end/8].Or(mask)
		}
	}
	newSearchAddr := limit - pageSize
	searchAddr, _ := s.searchAddr.Load()
	// N.B. Because mark is serialized, it's not necessary to do a
	// full CAS here. mark only ever increases searchAddr, while
	// find only ever decreases it. Since we only ever race with
	// decreases, even if the value we loaded is stale, the actual
	// value will never be larger.
	if (offAddr{searchAddr}).lessThan(offAddr{newSearchAddr}) {
		s.searchAddr.StoreMarked(newSearchAddr)
	}
}

// clear sets the chunk at index ci as not containing pages available to scavenge.
//
// Must be serialized with other mark, markRange, and clear calls.
func (s *scavengeIndex) clear(ci chunkIdx) {
	s.chunks[ci/8].And(^uint8(1 << (ci % 8)))
}

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
// That goal is defined as (retainExtraPercent+100) / 100 * next_gc.
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

const (
	// The background scavenger is paced according to these parameters.
	//
	// scavengePercent represents the portion of mutator time we're willing
	// to spend on scavenging in percent.
	//
	// scavengePageLatency is a worst-case estimate (order-of-magnitude) of
	// the time it takes to scavenge one (regular-sized) page of memory.
	// scavengeHugePageLatency is the same but for huge pages.
	//
	// scavengePagePeriod is derived from scavengePercent and scavengePageLatency,
	// and represents the average time between scavenging one page that we're
	// aiming for. scavengeHugePagePeriod is the same but for huge pages.
	// These constants are core to the scavenge pacing algorithm.
	scavengePercent         = 1    // 1%
	scavengePageLatency     = 10e3 // 10µs
	scavengeHugePageLatency = 10e3 // 10µs
	scavengePagePeriod      = scavengePageLatency / (scavengePercent / 100.0)
	scavengeHugePagePeriod  = scavengePageLatency / (scavengePercent / 100.0)

	// retainExtraPercent represents the amount of memory over the heap goal
	// that the scavenger should keep as a buffer space for the allocator.
	//
	// The purpose of maintaining this overhead is to have a greater pool of
	// unscavenged memory available for allocation (since using scavenged memory
	// incurs an additional cost), to account for heap fragmentation and
	// the ever-changing layout of the heap.
	retainExtraPercent = 10
)

// heapRetained returns an estimate of the current heap RSS.
//
// mheap_.lock must be held or the world must be stopped.
func heapRetained() uint64 {
	return memstats.heap_sys - memstats.heap_released
}

// gcPaceScavenger updates the scavenger's pacing, particularly
// its rate and RSS goal.
//
// The RSS goal is based on the current heap goal with a small overhead
// to accomodate non-determinism in the allocator.
//
// The pacing is based on scavengePageRate, which applies to both regular and
// huge pages. See that constant for more information.
//
// mheap_.lock must be held or the world must be stopped.
func gcPaceScavenger() {
	// Compute our scavenging goal and align it to a physical page boundary
	// to make the following calculations more exact.
	retainedGoal := memstats.next_gc
	// Add retainExtraPercent overhead to retainedGoal. This calculation
	// looks strange but the purpose is to arrive at an integer division
	// (e.g. if retainExtraPercent = 12.5, then we get a divisor of 8)
	// that also avoids the overflow from a multiplication.
	retainedGoal += retainedGoal / (1.0 / (retainExtraPercent / 100.0))
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

	// If we're already below our goal, publish the goal in case it changed
	// then disable the background scavenger.
	if retainedNow <= retainedGoal {
		mheap_.scavengeRetainedGoal = retainedGoal
		mheap_.scavengeBytesPerNS = 0
		return
	}

	// Now we start to compute the total amount of work necessary and the total
	// amount of time we're willing to give the scavenger to complete this work.
	// This will involve calculating how much of the work consists of huge pages
	// and how much consists of regular pages since the former can let us scavenge
	// more memory in the same time.
	totalWork := retainedNow - retainedGoal

	// On systems without huge page support, all work is regular work.
	regularWork := totalWork
	hugeTime := uint64(0)

	// On systems where we have huge pages, we want to do as much of the
	// scavenging work as possible on huge pages, because the costs are the
	// same per page, but we can give back more more memory in a shorter
	// period of time.
	if physHugePageSize != 0 {
		// Start by computing the amount of free memory we have in huge pages
		// in total. Trivially, this is all the huge page work we need to do.
		hugeWork := uint64(mheap_.free.unscavHugePages * physHugePageSize)

		// ...but it could turn out that there's more huge work to do than
		// total work, so cap it at total work. This might happen for very large
		// heaps where the additional factor of retainExtraPercent can make it so
		// that there are free chunks of memory larger than a huge page that we don't want
		// to scavenge.
		if hugeWork >= totalWork {
			hugePages := totalWork / uint64(physHugePageSize)
			hugeWork = hugePages * uint64(physHugePageSize)
		}
		// Everything that's not huge work is regular work. At this point we
		// know huge work so we can calculate how much time that will take
		// based on scavengePageRate (which applies to pages of any size).
		regularWork = totalWork - hugeWork
		hugeTime = hugeWork / uint64(physHugePageSize) * scavengeHugePagePeriod
	}
	// Finally, we can compute how much time it'll take to do the regular work
	// and the total time to do all the work.
	regularTime := regularWork / uint64(physPageSize) * scavengePagePeriod
	totalTime := hugeTime + regularTime

	now := nanotime()

	lock(&scavenge.lock)

	// Update all the pacing parameters in mheap with scavenge.lock held,
	// so that scavenge.gen is kept in sync with the updated values.
	mheap_.scavengeRetainedGoal = retainedGoal
	mheap_.scavengeRetainedBasis = retainedNow
	mheap_.scavengeTimeBasis = now
	mheap_.scavengeBytesPerNS = float64(totalWork) / float64(totalTime)
	scavenge.gen++ // increase scavenge generation

	// Wake up background scavenger if needed, since the pacing was just updated.
	wakeScavengerLocked()

	unlock(&scavenge.lock)
}

// State of the background scavenger.
var scavenge struct {
	lock   mutex
	g      *g
	parked bool
	timer  *timer
	gen    uint32 // read with either lock or mheap_.lock, write with both
}

// wakeScavengerLocked unparks the scavenger if necessary. It must be called
// after any pacing update.
//
// scavenge.lock must be held.
func wakeScavengerLocked() {
	if scavenge.parked {
		// Try to stop the timer but we don't really care if we succeed.
		// It's possible that either a timer was never started, or that
		// we're racing with it.
		// In the case that we're racing with there's the low chance that
		// we experience a spurious wake-up of the scavenger, but that's
		// totally safe.
		stopTimer(scavenge.timer)

		// Unpark the goroutine and tell it that there may have been a pacing
		// change.
		scavenge.parked = false
		ready(scavenge.g, 0, true)
	}
}

// scavengeSleep attempts to put the scavenger to sleep for ns.
// It also checks to see if gen != scavenge.gen before going to sleep,
// and aborts if true (meaning an update had occurred).
//
// Note that this function should only be called by the scavenger.
//
// The scavenger may be woken up earlier by a pacing change, and it may not go
// to sleep at all if there's a pending pacing change.
//
// Returns false if awoken early (i.e. true means a complete sleep).
func scavengeSleep(gen uint32, ns int64) bool {
	lock(&scavenge.lock)

	// If there was an update, just abort the sleep.
	if scavenge.gen != gen {
		unlock(&scavenge.lock)
		return false
	}

	// Set the timer.
	now := nanotime()
	scavenge.timer.when = now + ns
	startTimer(scavenge.timer)

	// Park the goroutine. It's fine that we don't publish the
	// fact that the timer was set; even if the timer wakes up
	// and fire scavengeReady before we park, it'll block on
	// scavenge.lock.
	scavenge.parked = true
	goparkunlock(&scavenge.lock, waitReasonSleep, traceEvGoSleep, 2)

	// Return true if we completed the full sleep.
	return (nanotime() - now) >= ns
}

// Background scavenger.
//
// The background scavenger maintains the RSS of the application below
// the line described by the proportional scavenging statistics in
// the mheap struct.
func bgscavenge(c chan int) {
	scavenge.g = getg()

	lock(&scavenge.lock)
	scavenge.parked = true

	scavenge.timer = new(timer)
	scavenge.timer.f = func(_ interface{}, _ uintptr) {
		lock(&scavenge.lock)
		wakeScavengerLocked()
		unlock(&scavenge.lock)
	}

	c <- 1
	goparkunlock(&scavenge.lock, waitReasonGCScavengeWait, traceEvGoBlock, 1)

	// Parameters for sleeping.
	//
	// If we end up doing more work than we need, we should avoid spinning
	// until we have more work to do: instead, we know exactly how much time
	// until more work will need to be done, so we sleep.
	//
	// We should avoid sleeping for less than minSleepNS because Gosched()
	// overheads among other things will work out better in that case.
	//
	// There's no reason to set a maximum on sleep time because we'll always
	// get woken up earlier if there's any kind of update that could change
	// the scavenger's pacing.
	//
	// retryDelayNS tracks how much to sleep next time we fail to do any
	// useful work.
	const minSleepNS = int64(100 * 1000) // 100 µs

	retryDelayNS := minSleepNS

	for {
		released := uintptr(0)
		park := false
		ttnext := int64(0)
		gen := uint32(0)

		// Run on the system stack since we grab the heap lock,
		// and a stack growth with the heap lock means a deadlock.
		systemstack(func() {
			lock(&mheap_.lock)

			gen = scavenge.gen

			// If background scavenging is disabled or if there's no work to do just park.
			retained := heapRetained()
			if mheap_.scavengeBytesPerNS == 0 || retained <= mheap_.scavengeRetainedGoal {
				unlock(&mheap_.lock)
				park = true
				return
			}

			// Calculate how big we want the retained heap to be
			// at this point in time.
			//
			// The formula is for that of a line, y = b - mx
			// We want y (want),
			//   m = scavengeBytesPerNS (> 0)
			//   x = time between scavengeTimeBasis and now
			//   b = scavengeRetainedBasis
			rate := mheap_.scavengeBytesPerNS
			tdist := nanotime() - mheap_.scavengeTimeBasis
			rdist := uint64(rate * float64(tdist))
			want := mheap_.scavengeRetainedBasis - rdist

			// If we're above the line, scavenge to get below the
			// line.
			if retained > want {
				released = mheap_.scavengeLocked(uintptr(retained - want))
			}
			unlock(&mheap_.lock)

			// If we over-scavenged a bit, calculate how much time it'll
			// take at the current rate for us to make that up. We definitely
			// won't have any work to do until at least that amount of time
			// passes.
			if released > uintptr(retained-want) {
				extra := released - uintptr(retained-want)
				ttnext = int64(float64(extra) / rate)
			}
		})

		if park {
			lock(&scavenge.lock)
			scavenge.parked = true
			goparkunlock(&scavenge.lock, waitReasonGCScavengeWait, traceEvGoBlock, 1)
			continue
		}

		if debug.gctrace > 0 {
			if released > 0 {
				print("scvg: ", released>>20, " MB released\n")
			}
			print("scvg: inuse: ", memstats.heap_inuse>>20, ", idle: ", memstats.heap_idle>>20, ", sys: ", memstats.heap_sys>>20, ", released: ", memstats.heap_released>>20, ", consumed: ", (memstats.heap_sys-memstats.heap_released)>>20, " (MB)\n")
		}

		if released == 0 {
			// If we were unable to release anything this may be because there's
			// no free memory available to scavenge. Go to sleep and try again.
			if scavengeSleep(gen, retryDelayNS) {
				// If we successfully slept through the delay, back off exponentially.
				retryDelayNS *= 2
			}
			continue
		}
		retryDelayNS = minSleepNS

		if ttnext > 0 && ttnext > minSleepNS {
			// If there's an appreciable amount of time until the next scavenging
			// goal, just sleep. We'll get woken up if anything changes and this
			// way we avoid spinning.
			scavengeSleep(gen, ttnext)
			continue
		}

		// Give something else a chance to run, no locks are held.
		Gosched()
	}
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// maxWhen is the maximum value for timer's when field.
const maxWhen = 1<<63 - 1

// verifyTimers can be set to true to add debugging checks that the
// timer heaps are valid.
const verifyTimers = false

type timing struct {
	// The when field of the first entry on the timer heap.
	// This is 0 if the timer heap is empty.
	timer0When atomic.Uint64

	// The earliest known nextwhen field of a timer with
	// timerModifiedEarlier status. Because the timer may have been
	// modified again, there need not be any timer with this value.
	// This is 0 if there are no timerModifiedEarlier timers.
	timerModifiedEarliest atomic.Uint64

	// Lock for timers. We normally access the timers while running
	// on this P, but the scheduler can also do it from a different P.
	timersLock mutex

	// Actions to take at some time. This is used to implement the
	// standard library's time package.
	// Must hold timersLock to access.
	timers []*timer

	// Number of timers in P's heap.
	numTimers atomic.Uint32

	// Number of timerDeleted timers in P's heap.
	deletedTimers atomic.Uint32

	// Race context used while executing timer functions.
	timerRaceCtx uintptr

	pp *p
}

func (ti *timing) init(pp *p) {
	ti.pp = pp
	lockInit(&ti.timersLock, lockRankTimers)
	ti.timerRaceCtx = racegostart(abi.FuncPCABIInternal(ti.runTimer) + sys.PCQuantum)
}

// destroy move all runnable timers to the caller process timing queue.
// caller must be whom held sched.lock and the world must be stopped.
func (ti *timing) destroy() {
	if len(ti.timers) > 0 {
		plocal := getg().m.p.ptr()
		// The world is stopped, but we acquire timersLock to
		// protect against sysmon calling timeSleepUntil.
		// This is the only case where we hold the timersLock of
		// more than one P, so there are no deadlock concerns.
		lock(&plocal.timing.timersLock)
		lock(&ti.timersLock)
		plocal.timing.moveTimers(ti.timers)
		ti.timers = nil
		ti.numTimers = 0
		ti.deletedTimers = 0
		ti.timer0When.Store(0)
		unlock(&ti.timersLock)
		unlock(&plocal.timing.timersLock)
	}
	if raceenabled {
		if ti.timerRaceCtx != 0 {
			// The race detector code uses a callback to fetch
			// the proc context, so arrange for that callback
			// to see the right thing.
			// This hack only works because we are the only
			// thread running.
			mp := getg().m
			phold := mp.p.ptr()
			mp.p.set(ti.pp)

			racectxend(ti.timerRaceCtx)
			ti.timerRaceCtx = 0

			mp.p.set(phold)
		}
	}
}

// addTimer adds t to the timing's heap.
// The caller must have locked the ti.timersLock.
func (ti *timing) addTimer(t *timer) {
	// Timers rely on the network poller, so make sure the poller
	// has started.
	if netpollInited == 0 {
		netpollGenericInit()
	}

	if t.pp != 0 {
		throw("timing.addTimer: P already set in timer")
	}
	t.pp.set(ti.pp)
	i := len(ti.timers)
	ti.timers = append(ti.timers, t)
	ti.siftupTimer(i)
	if t == ti.timers[0] {
		ti.timer0When.Store(uint64(t.when))
	}
	ti.numTimers.Add(1)
}

// deleteTimer removes timer i from the current the timing.
// We are locked on the P when this is called.
// It returns the smallest changed index in ti.timers.
// The caller must have locked the ti.timersLock.
func (ti *timing) deleteTimer(i int) int {
	t := ti.timers[i]
	if t.pp.ptr() != ti.pp {
		throw("timing.deleteTimer: wrong P")
	} else {
		t.pp = 0
	}
	last := len(ti.timers) - 1
	if i != last {
		ti.timers[i] = ti.timers[last]
	}
	ti.timers[last] = nil
	ti.timers = ti.timers[:last]
	smallestChanged := i
	if i != last {
		// Moving to i may have moved the last timer to a new parent,
		// so sift up to preserve the heap guarantee.
		smallestChanged = ti.siftupTimer(i)
		ti.siftdownTimer(i)
	}
	if i == 0 {
		ti.updateTimer0When()
	}
	n := ti.numTimers.Add(-1)
	if n == 0 {
		// If there are no timers, then clearly none are modified.
		ti.timerModifiedEarliest.Store(0)
	}
	return smallestChanged
}

// deleteTimer0 removes timer 0 from the current timing.
// We are locked on the P when this is called.
// It reports whether it saw no problems due to races.
// The caller must have locked the ti.timersLock.
func (ti *timing) deleteTimer0() {
	t := ti.timers[0]
	if t.pp.ptr() != ti.pp {
		throw("timing.deleteTimer0: wrong P")
	} else {
		t.pp = 0
	}
	last := len(ti.timers) - 1
	if last > 0 {
		ti.timers[0] = ti.timers[last]
	}
	ti.timers[last] = nil
	ti.timers = ti.timers[:last]
	if last > 0 {
		ti.siftdownTimer(0)
	}
	ti.updateTimer0When()
	n := ti.numTimers.Add(-1)
	if n == 0 {
		// If there are no timers, then clearly none are modified.
		ti.timerModifiedEarliest.Store(0)
	}
}

// cleanTimers cleans up the head of the timer queue. This speeds up
// programs that create and delete timers; leaving them in the heap
// slows down timer.add(). Reports whether no timer problems were found.
// The caller must have locked the ti.timersLock
func (ti *timing) cleanTimers() {
	gp := getg()
	for {
		if len(ti.timers) == 0 {
			return
		}

		// This loop can theoretically run for a while, and because
		// it is holding timersLock it cannot be preempted.
		// If someone is trying to preempt us, just return.
		// We can clean the timers later.
		if gp.preemptStop {
			return
		}

		t := ti.timers[0]
		if t.pp.ptr() != ti.pp {
			throw("timing.cleanTimers: bad p")
		}
		switch s := t.status.Load(); s {
		case timerDeleted:
			if !t.status.CompareAndSwap(s, timerRemoving) {
				continue
			}
			ti.deleteTimer0()
			if !t.status.CompareAndSwap(timerRemoving, timerRemoved) {
				badTimer()
			}
			ti.deletedTimers.Add(-1)
		case timerModifiedEarlier, timerModifiedLater:
			if !t.status.CompareAndSwap(s, timerMoving) {
				continue
			}
			// Now we can change the when field.
			t.when = t.nextwhen
			// Move t to the right position.
			ti.deleteTimer0()
			ti.addTimer(t)
			if !t.status.CompareAndSwap(timerMoving, timerWaiting) {
				badTimer()
			}
		default:
			// Head of timers does not need adjustment.
			return
		}
	}
}

// moveTimers moves a slice of timers to the timing.
// The slice has been taken from a different P.
// This is currently called when the world is stopped, but the caller
// is expected to have locked the ti.timersLock
func (ti *timing) moveTimers(timers []*timer) {
	for _, t := range timers {
	loop:
		for {
			switch s := t.status.Load(); s {
			case timerWaiting:
				if !t.status.CompareAndSwap(s, timerMoving) {
					continue
				}
				t.pp = 0
				ti.addTimer(t)
				if !t.status.CompareAndSwap(timerMoving, timerWaiting) {
					badTimer()
				}
				break loop
			case timerModifiedEarlier, timerModifiedLater:
				if !t.status.CompareAndSwap(s, timerMoving) {
					continue
				}
				t.when = t.nextwhen
				t.pp = 0
				ti.addTimer(t)
				if !t.status.CompareAndSwap(timerMoving, timerWaiting) {
					badTimer()
				}
				break loop
			case timerDeleted:
				if !t.status.CompareAndSwap(s, timerRemoved) {
					continue
				}
				t.pp = 0
				// We no longer need this timer in the heap.
				break loop
			case timerModifying:
				// Loop until the modification is complete.
				osyield()
			case timerNoStatus, timerRemoved:
				// We should not see these status values in a timers heap.
				badTimer()
			case timerRunning, timerRemoving, timerMoving:
				// Some other P thinks it owns this timer,
				// which should not happen.
				badTimer()
			default:
				badTimer()
			}
		}
	}
}

// adjustTimers looks through the timers in the current P's heap for
// any timers that have been modified to run earlier, and puts them in
// the correct place in the heap. While looking for those timers,
// it also moves timers that have been modified to run later,
// and removes deleted timers. The caller must have locked the timers for pp.
func (ti *timing) adjustTimers(now int64) {
	// If we haven't yet reached the time of the first timerModifiedEarlier
	// timer, don't do anything. This speeds up programs that adjust
	// a lot of timers back and forth if the timers rarely expire.
	// We'll postpone looking through all the adjusted timers until
	// one would actually expire.
	first := ti.timerModifiedEarliest.Load()
	if first == 0 || int64(first) > now {
		if verifyTimers {
			ti.verifyTimerHeap()
		}
		return
	}

	// We are going to clear all timerModifiedEarlier timers.
	ti.timerModifiedEarliest.Store(0)

	var moved []*timer
	for i := 0; i < len(ti.timers); i++ {
		t := ti.timers[i]
		if t.pp.ptr() != ti.pp {
			throw("timing.adjustTimers: bad p")
		}
		switch s := t.status.Load(); s {
		case timerDeleted:
			if t.status.CompareAndSwap(s, timerRemoving) {
				changed := ti.deleteTimer(i)
				if !t.status.CompareAndSwap(timerRemoving, timerRemoved) {
					badTimer()
				}
				ti.deletedTimers.Add(-1)
				// Go back to the earliest changed heap entry.
				// "- 1" because the loop will add 1.
				i = changed - 1
			}
		case timerModifiedEarlier, timerModifiedLater:
			if t.status.CompareAndSwap(s, timerMoving) {
				// Now we can change the when field.
				t.when = t.nextwhen
				// Take t off the heap, and hold onto it.
				// We don't add it back yet because the
				// heap manipulation could cause our
				// loop to skip some other timer.
				changed := ti.deleteTimer(i)
				moved = append(moved, t)
				// Go back to the earliest changed heap entry.
				// "- 1" because the loop will add 1.
				i = changed - 1
			}
		case timerNoStatus, timerRunning, timerRemoving, timerRemoved, timerMoving:
			badTimer()
		case timerWaiting:
			// OK, nothing to do.
		case timerModifying:
			// Check again after modification is complete.
			osyield()
			i--
		default:
			badTimer()
		}
	}

	if len(moved) > 0 {
		ti.addAdjustedTimers(moved)
	}

	if verifyTimers {
		ti.verifyTimerHeap()
	}
}

// addAdjustedTimers adds any timers we adjusted in adjustTimers
// back to the timer heap.
func (ti *timing) addAdjustedTimers(moved []*timer) {
	for _, t := range moved {
		ti.addTimer(t)
		if !t.status.CompareAndSwap(timerMoving, timerWaiting) {
			badTimer()
		}
	}
}

// noBarrierWakeTime looks at timers and returns the time when we
// should wake up the netpoller. It returns 0 if there are no timers.
// This function is invoked when dropping a P, and must run without
// any write barriers.
//
//go:nowritebarrierrec
func (ti *timing) noBarrierWakeTime() int64 {
	next := int64(ti.timer0When.Load())
	nextAdj := int64(ti.timerModifiedEarliest.Load())
	if next == 0 || (nextAdj != 0 && nextAdj < next) {
		next = nextAdj
	}
	return next
}

// runTimer examines the first timer in timers. If it is ready based on now,
// it runs the timer and removes or updates it.
// Returns 0 if it ran a timer, -1 if there are no more timers, or the time
// when the first timer should run.
// The caller must have locked the timers for pp.
// If a timer is run, this will temporarily unlock the timers.
//
//go:systemstack
func (ti *timing) runTimer(now int64) int64 {
	for {
		t := ti.timers[0]
		if t.pp.ptr() != ti.pp {
			throw("timing.runTimer: bad p")
		}
		switch s := t.status.Load(); s {
		case timerWaiting:
			if t.when > now {
				// Not ready to run.
				return t.when
			}

			if !t.status.CompareAndSwap(s, timerRunning) {
				continue
			}
			// Note that runOneTimer may temporarily unlock
			// timing.timersLock.
			ti.runOneTimer(t, now)
			return 0

		case timerDeleted:
			if !t.status.CompareAndSwap(s, timerRemoving) {
				continue
			}
			ti.deleteTimer0()
			if !t.status.CompareAndSwap(timerRemoving, timerRemoved) {
				badTimer()
			}
			ti.deletedTimers.Add(-1)
			if len(ti.timers) == 0 {
				return -1
			}

		case timerModifiedEarlier, timerModifiedLater:
			if !t.status.CompareAndSwap(s, timerMoving) {
				continue
			}
			t.when = t.nextwhen
			ti.deleteTimer0()
			ti.addTimer(t)
			if !t.status.CompareAndSwap(timerMoving, timerWaiting) {
				badTimer()
			}

		case timerModifying:
			// Wait for modification to complete.
			osyield()

		case timerNoStatus, timerRemoved:
			// Should not see a new or inactive timer on the heap.
			badTimer()
		case timerRunning, timerRemoving, timerMoving:
			// These should only be set when timers are locked,
			// and we didn't do it.
			badTimer()
		default:
			badTimer()
		}
	}
}

// runOneTimer runs a single timer.
// The caller must have locked the ti.timersLock.
// This will temporarily unlock the timers while running the timer function.
//
//go:systemstack
func (ti *timing) runOneTimer(t *timer, now int64) {
	if raceenabled {
		ppcur := getg().m.p.ptr()
		raceacquirectx(ppcur.timing.timerRaceCtx, unsafe.Pointer(t))
	}

	f := t.f
	arg := t.arg
	seq := t.seq

	if t.period > 0 {
		// Leave in heap but adjust next time to fire.
		delta := t.when - now
		t.when += t.period * (1 + -delta/t.period)
		if t.when < 0 { // check for overflow.
			t.when = maxWhen
		}
		ti.siftdownTimer(0)
		if !t.status.CompareAndSwap(timerRunning, timerWaiting) {
			badTimer()
		}
		ti.updateTimer0When()
	} else {
		// Remove from heap.
		ti.deleteTimer0()
		if !t.status.CompareAndSwap(timerRunning, timerNoStatus) {
			badTimer()
		}
	}

	if raceenabled {
		// Temporarily use the current P's timing.timerRaceCtx for g0.
		gp := getg()
		if gp.racectx != 0 {
			throw("timing.runOneTimer: unexpected racectx")
		}
		gp.racectx = gp.m.p.ptr().timing.timerRaceCtx
	}

	unlock(&ti.timersLock)

	f(arg, seq)

	lock(&ti.timersLock)

	if raceenabled {
		gp := getg()
		gp.racectx = 0
	}
}

// clearDeletedTimers removes all deleted timers from the P's timer heap.
// This is used to avoid clogging up the heap if the program
// starts a lot of long-running timers and then stops them.
// For example, this can happen via context.WithTimeout.
//
// This is the only function that walks through the entire timer heap,
// other than moveTimers which only runs when the world is stopped.
//
// The caller must have locked the timers for pp.
func (ti *timing) clearDeletedTimers() {
	// We are going to clear all timerModifiedEarlier timers.
	// Do this now in case new ones show up while we are looping.
	ti.timerModifiedEarliest.Store(0)

	cdel := int32(0)
	to := 0
	changedHeap := false
	timers := ti.timers
nextTimer:
	for _, t := range timers {
		for {
			switch s := t.status.Load(); s {
			case timerWaiting:
				if changedHeap {
					timers[to] = t
					ti.siftupTimer(to)
				}
				to++
				continue nextTimer
			case timerModifiedEarlier, timerModifiedLater:
				if t.status.CompareAndSwap(s, timerMoving) {
					t.when = t.nextwhen
					timers[to] = t
					ti.siftupTimer(to)
					to++
					changedHeap = true
					if !t.status.CompareAndSwap(timerMoving, timerWaiting) {
						badTimer()
					}
					continue nextTimer
				}
			case timerDeleted:
				if t.status.CompareAndSwap(s, timerRemoving) {
					t.pp = 0
					cdel++
					if !t.status.CompareAndSwap(timerRemoving, timerRemoved) {
						badTimer()
					}
					changedHeap = true
					continue nextTimer
				}
			case timerModifying:
				// Loop until modification complete.
				osyield()
			case timerNoStatus, timerRemoved:
				// We should not see these status values in a timer heap.
				badTimer()
			case timerRunning, timerRemoving, timerMoving:
				// Some other P thinks it owns this timer,
				// which should not happen.
				badTimer()
			default:
				badTimer()
			}
		}
	}

	// Set remaining slots in timers slice to nil,
	// so that the timer values can be garbage collected.
	for i := to; i < len(timers); i++ {
		timers[i] = nil
	}

	ti.deletedTimers.Add(-cdel)
	ti.numTimers.Add(-cdel)

	timers = timers[:to]
	ti.timers = timers
	ti.updateTimer0When()

	if verifyTimers {
		ti.verifyTimerHeap()
	}
}

// verifyTimerHeap verifies that the timer heap is in a valid state.
// This is only for debugging, and is only called if verifyTimers is true.
// The caller must have locked the timers.
func (ti *timing) verifyTimerHeap() {
	for i, t := range ti.timers {
		if i == 0 {
			// First timer has no parent.
			continue
		}

		// The heap is 4-ary. See siftupTimer and siftdownTimer.
		p := (i - 1) / 4
		if t.when < ti.timers[p].when {
			print("bad timer heap at ", i, ": ", p, ": ", ti.timers[p].when, ", ", i, ": ", t.when, "\n")
			throw("bad timer heap")
		}
	}
	if numTimers := int(ti.numTimers.Load()); len(ti.timers) != numTimers {
		println("timer heap len", len(ti.timers), "!= numTimers", numTimers)
		throw("bad timer heap len")
	}
}

// updateTimer0When sets the P's timer0When field.
// The caller must have locked the timers for pp.
func (ti *timing) updateTimer0When() {
	if len(ti.timers) == 0 {
		ti.timer0When.Store(0)
	} else {
		ti.timer0When.Store(uint64(ti.timers[0].when))
	}
}

// updateTimerModifiedEarliest updates the recorded nextwhen field of the
// earlier timerModifiedEarier value.
// The timers for pp will not be locked.
func (ti *timing) updateTimerModifiedEarliest(nextwhen int64) {
	for {
		old :=ti.timerModifiedEarliest.Load()
		if old != 0 && int64(old) < nextwhen {
			return
		}
		if ti.timerModifiedEarliest.CompareAndSwap(old, uint64(nextwhen)) {
			return
		}
	}
}

// checkTimers runs any timers that are ready.
// If now is not 0 it is the current time.
// It returns the passed time or the current time if now was passed as 0.
// and the time when the next timer should run or 0 if there is no next timer,
// and reports whether it ran any timers.
// If the time when the next timer should run is not 0,
// it is always larger than the returned time.
// We pass now in and out to avoid extra calls of nanotime.
//
//go:yeswritebarrierrec
func (ti *timing) checkTimers(now int64) (rnow, pollUntil int64, ran bool) {
	// If it's not yet time for the first timer, or the first adjusted
	// timer, then there is nothing to do.
	next := int64(ti.timer0When.Load())
	nextAdj := int64(ti.timerModifiedEarliest.Load())
	if next == 0 || (nextAdj != 0 && nextAdj < next) {
		next = nextAdj
	}

	if next == 0 {
		// No timers to run or adjust.
		return now, 0, false
	}

	if now == 0 {
		now = nanotime()
	}
	if now < next {
		// Next timer is not ready to run, but keep going
		// if we would clear deleted timers.
		// This corresponds to the condition below where
		// we decide whether to call clearDeletedTimers.
		if ti.pp != getg().m.p.ptr() || int(ti.deletedTimers.Load()) <= int(ti.numTimers.Load()/4) {
			return now, next, false
		}
	}

	lock(&ti.timersLock)

	if len(ti.timers) > 0 {
		ti.adjustTimers(now)
		for len(ti.timers) > 0 {
			// Note that runTimer may temporarily unlock
			// ti.timersLock.
			if tw := ti.runTimer(now); tw != 0 {
				if tw > 0 {
					pollUntil = tw
				}
				break
			}
			ran = true
		}
	}

	// If this is the local P, and there are a lot of deleted timers,
	// clear them out. We only do this for the local P to reduce
	// lock contention on timersLock.
	if ti.pp == getg().m.p.ptr() && int(ti.deletedTimers.Load()) > len(ti.timers)/4 {
		ti.clearDeletedTimers()
	}

	unlock(&ti.timersLock)

	return now, pollUntil, ran
}

// updateTimerPMask clears pp's timer mask if it has no timers on its heap.
//
// Ideally, the timer mask would be kept immediately consistent on any timer
// operations. Unfortunately, updating a shared global data structure in the
// timer hot path adds too much overhead in applications frequently switching
// between no timers and some timers.
//
// As a compromise, the timer mask is updated only on pidleget / pidleput. A
// running P (returned by pidleget) may add a timer at any time, so its mask
// must be set. An idle P (passed to pidleput) cannot add new timers while
// idle, so if it has no timers at that time, its mask may be cleared.
//
// Thus, we get the following effects on timer-stealing in findrunnable:
//
//   - Idle Ps with no timers when they go idle are never checked in findrunnable
//     (for work- or timer-stealing; this is the ideal case).
//   - Running Ps must always be checked.
//   - Idle Ps whose timers are stolen must continue to be checked until they run
//     again, even after timer expiration.
//
// When the P starts running again, the mask should be set, as a timer may be
// added at any time.
//
// TODO(prattmic): Additional targeted updates may improve the above cases.
// e.g., updating the mask when stealing a timer.
func (ti *timing) updateTimerPMask() {
	if ti.numTimers.Load() > 0 {
		return
	}

	// Looks like there are no timers, however another P may transiently
	// decrement numTimers when handling a timerModified timer in
	// checkTimers. We must take timersLock to serialize with these changes.
	lock(&ti.timersLock)
	if ti.numTimers.Load() == 0 {
		timerpMask.clear(ti.pp.id)
	}
	unlock(&ti.timersLock)
}

// Heap maintenance algorithms.
// These algorithms check for slice index errors manually.
// Slice index error can happen if the program is using racy
// access to timers. We don't want to panic here, because
// it will cause the program to crash with a mysterious
// "panic holding locks" message. Instead, we panic while not
// holding a lock.

// siftupTimer puts the timer at position i in the right place
// in the heap by moving it up toward the top of the heap.
// It returns the smallest changed index.
func (ti *timing) siftupTimer(i int) int {
	var t []*timer = ti.timers

	if i >= len(t) {
		badTimer()
	}
	when := t[i].when
	if when <= 0 {
		badTimer()
	}
	tmp := t[i]
	for i > 0 {
		p := (i - 1) / 4 // parent
		if when >= t[p].when {
			break
		}
		t[i] = t[p]
		i = p
	}
	if tmp != t[i] {
		t[i] = tmp
	}
	return i
}

// siftdownTimer puts the timer at position i in the right place
// in the heap by moving it down toward the bottom of the heap.
func (ti *timing) siftdownTimer(i int) {
	var t []*timer = ti.timers

	n := len(t)
	if i >= n {
		badTimer()
	}
	when := t[i].when
	if when <= 0 {
		badTimer()
	}
	tmp := t[i]
	for {
		c := i*4 + 1 // left child
		c3 := c + 2  // mid child
		if c >= n {
			break
		}
		w := t[c].when
		if c+1 < n && t[c+1].when < w {
			w = t[c+1].when
			c++
		}
		if c3 < n {
			w3 := t[c3].when
			if c3+1 < n && t[c3+1].when < w3 {
				w3 = t[c3+1].when
				c3++
			}
			if w3 < w {
				w = w3
				c = c3
			}
		}
		if w >= when {
			break
		}
		t[i] = t[c]
		i = c
	}
	if tmp != t[i] {
		t[i] = tmp
	}
}

// badTimer is called if the timer data structures have been corrupted,
// presumably due to racy use by the program. We panic here rather than
// panicing due to invalid slice access while holding locks.
// See issue #25686.
func badTimer() {
	throw("timer data corruption")
}

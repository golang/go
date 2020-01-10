// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Time-related runtime and pieces of package time.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// Package time knows the layout of this structure.
// If this struct changes, adjust ../time/sleep.go:/runtimeTimer.
type timer struct {
	// If this timer is on a heap, which P's heap it is on.
	// puintptr rather than *p to match uintptr in the versions
	// of this struct defined in other packages.
	pp puintptr

	// Timer wakes up at when, and then at when+period, ... (period > 0 only)
	// each time calling f(arg, now) in the timer goroutine, so f must be
	// a well-behaved function and not block.
	when   int64
	period int64
	f      func(interface{}, uintptr)
	arg    interface{}
	seq    uintptr

	// What to set the when field to in timerModifiedXX status.
	nextwhen int64

	// The status field holds one of the values below.
	status uint32
}

// Code outside this file has to be careful in using a timer value.
//
// The pp, status, and nextwhen fields may only be used by code in this file.
//
// Code that creates a new timer value can set the when, period, f,
// arg, and seq fields.
// A new timer value may be passed to addtimer (called by time.startTimer).
// After doing that no fields may be touched.
//
// An active timer (one that has been passed to addtimer) may be
// passed to deltimer (time.stopTimer), after which it is no longer an
// active timer. It is an inactive timer.
// In an inactive timer the period, f, arg, and seq fields may be modified,
// but not the when field.
// It's OK to just drop an inactive timer and let the GC collect it.
// It's not OK to pass an inactive timer to addtimer.
// Only newly allocated timer values may be passed to addtimer.
//
// An active timer may be passed to modtimer. No fields may be touched.
// It remains an active timer.
//
// An inactive timer may be passed to resettimer to turn into an
// active timer with an updated when field.
// It's OK to pass a newly allocated timer value to resettimer.
//
// Timer operations are addtimer, deltimer, modtimer, resettimer,
// cleantimers, adjusttimers, and runtimer.
//
// We don't permit calling addtimer/deltimer/modtimer/resettimer simultaneously,
// but adjusttimers and runtimer can be called at the same time as any of those.
//
// Active timers live in heaps attached to P, in the timers field.
// Inactive timers live there too temporarily, until they are removed.
//
// addtimer:
//   timerNoStatus   -> timerWaiting
//   anything else   -> panic: invalid value
// deltimer:
//   timerWaiting    -> timerDeleted
//   timerModifiedXX -> timerDeleted
//   timerNoStatus   -> do nothing
//   timerDeleted    -> do nothing
//   timerRemoving   -> do nothing
//   timerRemoved    -> do nothing
//   timerRunning    -> wait until status changes
//   timerMoving     -> wait until status changes
//   timerModifying  -> panic: concurrent deltimer/modtimer calls
// modtimer:
//   timerWaiting    -> timerModifying -> timerModifiedXX
//   timerModifiedXX -> timerModifying -> timerModifiedYY
//   timerNoStatus   -> timerWaiting
//   timerRemoved    -> timerWaiting
//   timerRunning    -> wait until status changes
//   timerMoving     -> wait until status changes
//   timerRemoving   -> wait until status changes
//   timerDeleted    -> panic: concurrent modtimer/deltimer calls
//   timerModifying  -> panic: concurrent modtimer calls
// resettimer:
//   timerNoStatus   -> timerWaiting
//   timerRemoved    -> timerWaiting
//   timerDeleted    -> timerModifying -> timerModifiedXX
//   timerRemoving   -> wait until status changes
//   timerRunning    -> wait until status changes
//   timerWaiting    -> panic: resettimer called on active timer
//   timerMoving     -> panic: resettimer called on active timer
//   timerModifiedXX -> panic: resettimer called on active timer
//   timerModifying  -> panic: resettimer called on active timer
// cleantimers (looks in P's timer heap):
//   timerDeleted    -> timerRemoving -> timerRemoved
//   timerModifiedXX -> timerMoving -> timerWaiting
// adjusttimers (looks in P's timer heap):
//   timerDeleted    -> timerRemoving -> timerRemoved
//   timerModifiedXX -> timerMoving -> timerWaiting
// runtimer (looks in P's timer heap):
//   timerNoStatus   -> panic: uninitialized timer
//   timerWaiting    -> timerWaiting or
//   timerWaiting    -> timerRunning -> timerNoStatus or
//   timerWaiting    -> timerRunning -> timerWaiting
//   timerModifying  -> wait until status changes
//   timerModifiedXX -> timerMoving -> timerWaiting
//   timerDeleted    -> timerRemoving -> timerRemoved
//   timerRunning    -> panic: concurrent runtimer calls
//   timerRemoved    -> panic: inconsistent timer heap
//   timerRemoving   -> panic: inconsistent timer heap
//   timerMoving     -> panic: inconsistent timer heap

// Values for the timer status field.
const (
	// Timer has no status set yet.
	timerNoStatus = iota

	// Waiting for timer to fire.
	// The timer is in some P's heap.
	timerWaiting

	// Running the timer function.
	// A timer will only have this status briefly.
	timerRunning

	// The timer is deleted and should be removed.
	// It should not be run, but it is still in some P's heap.
	timerDeleted

	// The timer is being removed.
	// The timer will only have this status briefly.
	timerRemoving

	// The timer has been stopped.
	// It is not in any P's heap.
	timerRemoved

	// The timer is being modified.
	// The timer will only have this status briefly.
	timerModifying

	// The timer has been modified to an earlier time.
	// The new when value is in the nextwhen field.
	// The timer is in some P's heap, possibly in the wrong place.
	timerModifiedEarlier

	// The timer has been modified to the same or a later time.
	// The new when value is in the nextwhen field.
	// The timer is in some P's heap, possibly in the wrong place.
	timerModifiedLater

	// The timer has been modified and is being moved.
	// The timer will only have this status briefly.
	timerMoving
)

// maxWhen is the maximum value for timer's when field.
const maxWhen = 1<<63 - 1

// verifyTimers can be set to true to add debugging checks that the
// timer heaps are valid.
const verifyTimers = false

// Package time APIs.
// Godoc uses the comments in package time, not these.

// time.now is implemented in assembly.

// timeSleep puts the current goroutine to sleep for at least ns nanoseconds.
//go:linkname timeSleep time.Sleep
func timeSleep(ns int64) {
	if ns <= 0 {
		return
	}

	gp := getg()
	t := gp.timer
	if t == nil {
		t = new(timer)
		gp.timer = t
	}
	t.f = goroutineReady
	t.arg = gp
	t.nextwhen = nanotime() + ns
	gopark(resetForSleep, unsafe.Pointer(t), waitReasonSleep, traceEvGoSleep, 1)
}

// resetForSleep is called after the goroutine is parked for timeSleep.
// We can't call resettimer in timeSleep itself because if this is a short
// sleep and there are many goroutines then the P can wind up running the
// timer function, goroutineReady, before the goroutine has been parked.
func resetForSleep(gp *g, ut unsafe.Pointer) bool {
	t := (*timer)(ut)
	resettimer(t, t.nextwhen)
	return true
}

// startTimer adds t to the timer heap.
//go:linkname startTimer time.startTimer
func startTimer(t *timer) {
	if raceenabled {
		racerelease(unsafe.Pointer(t))
	}
	addtimer(t)
}

// stopTimer stops a timer.
// It reports whether t was stopped before being run.
//go:linkname stopTimer time.stopTimer
func stopTimer(t *timer) bool {
	return deltimer(t)
}

// resetTimer resets an inactive timer, adding it to the heap.
//go:linkname resetTimer time.resetTimer
func resetTimer(t *timer, when int64) {
	if raceenabled {
		racerelease(unsafe.Pointer(t))
	}
	resettimer(t, when)
}

// Go runtime.

// Ready the goroutine arg.
func goroutineReady(arg interface{}, seq uintptr) {
	goready(arg.(*g), 0)
}

// addtimer adds a timer to the current P.
// This should only be called with a newly created timer.
// That avoids the risk of changing the when field of a timer in some P's heap,
// which could cause the heap to become unsorted.
func addtimer(t *timer) {
	// when must never be negative; otherwise runtimer will overflow
	// during its delta calculation and never expire other runtime timers.
	if t.when < 0 {
		t.when = maxWhen
	}
	if t.status != timerNoStatus {
		badTimer()
	}
	t.status = timerWaiting

	addInitializedTimer(t)
}

// addInitializedTimer adds an initialized timer to the current P.
func addInitializedTimer(t *timer) {
	when := t.when

	pp := getg().m.p.ptr()
	lock(&pp.timersLock)
	ok := cleantimers(pp) && doaddtimer(pp, t)
	unlock(&pp.timersLock)
	if !ok {
		badTimer()
	}

	wakeNetPoller(when)
}

// doaddtimer adds t to the current P's heap.
// It reports whether it saw no problems due to races.
// The caller must have locked the timers for pp.
func doaddtimer(pp *p, t *timer) bool {
	// Timers rely on the network poller, so make sure the poller
	// has started.
	if netpollInited == 0 {
		netpollGenericInit()
	}

	if t.pp != 0 {
		throw("doaddtimer: P already set in timer")
	}
	t.pp.set(pp)
	i := len(pp.timers)
	pp.timers = append(pp.timers, t)
	return siftupTimer(pp.timers, i)
}

// deltimer deletes the timer t. It may be on some other P, so we can't
// actually remove it from the timers heap. We can only mark it as deleted.
// It will be removed in due course by the P whose heap it is on.
// Reports whether the timer was removed before it was run.
func deltimer(t *timer) bool {
	for {
		switch s := atomic.Load(&t.status); s {
		case timerWaiting, timerModifiedLater:
			tpp := t.pp.ptr()
			if atomic.Cas(&t.status, s, timerDeleted) {
				atomic.Xadd(&tpp.deletedTimers, 1)
				// Timer was not yet run.
				return true
			}
		case timerModifiedEarlier:
			tpp := t.pp.ptr()
			if atomic.Cas(&t.status, s, timerModifying) {
				atomic.Xadd(&tpp.adjustTimers, -1)
				if !atomic.Cas(&t.status, timerModifying, timerDeleted) {
					badTimer()
				}
				atomic.Xadd(&tpp.deletedTimers, 1)
				// Timer was not yet run.
				return true
			}
		case timerDeleted, timerRemoving, timerRemoved:
			// Timer was already run.
			return false
		case timerRunning, timerMoving:
			// The timer is being run or moved, by a different P.
			// Wait for it to complete.
			osyield()
		case timerNoStatus:
			// Removing timer that was never added or
			// has already been run. Also see issue 21874.
			return false
		case timerModifying:
			// Simultaneous calls to deltimer and modtimer.
			badTimer()
		default:
			badTimer()
		}
	}
}

// dodeltimer removes timer i from the current P's heap.
// We are locked on the P when this is called.
// It reports whether it saw no problems due to races.
// The caller must have locked the timers for pp.
func dodeltimer(pp *p, i int) bool {
	if t := pp.timers[i]; t.pp.ptr() != pp {
		throw("dodeltimer: wrong P")
	} else {
		t.pp = 0
	}
	last := len(pp.timers) - 1
	if i != last {
		pp.timers[i] = pp.timers[last]
	}
	pp.timers[last] = nil
	pp.timers = pp.timers[:last]
	ok := true
	if i != last {
		// Moving to i may have moved the last timer to a new parent,
		// so sift up to preserve the heap guarantee.
		if !siftupTimer(pp.timers, i) {
			ok = false
		}
		if !siftdownTimer(pp.timers, i) {
			ok = false
		}
	}
	return ok
}

// dodeltimer0 removes timer 0 from the current P's heap.
// We are locked on the P when this is called.
// It reports whether it saw no problems due to races.
// The caller must have locked the timers for pp.
func dodeltimer0(pp *p) bool {
	if t := pp.timers[0]; t.pp.ptr() != pp {
		throw("dodeltimer0: wrong P")
	} else {
		t.pp = 0
	}
	last := len(pp.timers) - 1
	if last > 0 {
		pp.timers[0] = pp.timers[last]
	}
	pp.timers[last] = nil
	pp.timers = pp.timers[:last]
	ok := true
	if last > 0 {
		ok = siftdownTimer(pp.timers, 0)
	}
	return ok
}

// modtimer modifies an existing timer.
// This is called by the netpoll code.
func modtimer(t *timer, when, period int64, f func(interface{}, uintptr), arg interface{}, seq uintptr) {
	if when < 0 {
		when = maxWhen
	}

	status := uint32(timerNoStatus)
	wasRemoved := false
loop:
	for {
		switch status = atomic.Load(&t.status); status {
		case timerWaiting, timerModifiedEarlier, timerModifiedLater:
			if atomic.Cas(&t.status, status, timerModifying) {
				break loop
			}
		case timerNoStatus, timerRemoved:
			// Timer was already run and t is no longer in a heap.
			// Act like addtimer.
			if atomic.Cas(&t.status, status, timerWaiting) {
				wasRemoved = true
				break loop
			}
		case timerRunning, timerRemoving, timerMoving:
			// The timer is being run or moved, by a different P.
			// Wait for it to complete.
			osyield()
		case timerDeleted:
			// Simultaneous calls to modtimer and deltimer.
			badTimer()
		case timerModifying:
			// Multiple simultaneous calls to modtimer.
			badTimer()
		default:
			badTimer()
		}
	}

	t.period = period
	t.f = f
	t.arg = arg
	t.seq = seq

	if wasRemoved {
		t.when = when
		addInitializedTimer(t)
	} else {
		// The timer is in some other P's heap, so we can't change
		// the when field. If we did, the other P's heap would
		// be out of order. So we put the new when value in the
		// nextwhen field, and let the other P set the when field
		// when it is prepared to resort the heap.
		t.nextwhen = when

		newStatus := uint32(timerModifiedLater)
		if when < t.when {
			newStatus = timerModifiedEarlier
		}

		// Update the adjustTimers field.  Subtract one if we
		// are removing a timerModifiedEarlier, add one if we
		// are adding a timerModifiedEarlier.
		tpp := t.pp.ptr()
		adjust := int32(0)
		if status == timerModifiedEarlier {
			adjust--
		}
		if newStatus == timerModifiedEarlier {
			adjust++
		}
		if adjust != 0 {
			atomic.Xadd(&tpp.adjustTimers, adjust)
		}

		// Set the new status of the timer.
		if !atomic.Cas(&t.status, timerModifying, newStatus) {
			badTimer()
		}

		// If the new status is earlier, wake up the poller.
		if newStatus == timerModifiedEarlier {
			wakeNetPoller(when)
		}
	}
}

// resettimer resets an existing inactive timer to turn it into an active timer,
// with a new time for when the timer should fire.
// This should be called instead of addtimer if the timer value has been,
// or may have been, used previously.
func resettimer(t *timer, when int64) {
	if when < 0 {
		when = maxWhen
	}

	for {
		switch s := atomic.Load(&t.status); s {
		case timerNoStatus, timerRemoved:
			if atomic.Cas(&t.status, s, timerWaiting) {
				t.when = when
				addInitializedTimer(t)
				return
			}
		case timerDeleted:
			tpp := t.pp.ptr()
			if atomic.Cas(&t.status, s, timerModifying) {
				t.nextwhen = when
				newStatus := uint32(timerModifiedLater)
				if when < t.when {
					newStatus = timerModifiedEarlier
					atomic.Xadd(&t.pp.ptr().adjustTimers, 1)
				}
				if !atomic.Cas(&t.status, timerModifying, newStatus) {
					badTimer()
				}
				atomic.Xadd(&tpp.deletedTimers, -1)
				if newStatus == timerModifiedEarlier {
					wakeNetPoller(when)
				}
				return
			}
		case timerRemoving:
			// Wait for the removal to complete.
			osyield()
		case timerRunning:
			// Even though the timer should not be active,
			// we can see timerRunning if the timer function
			// permits some other goroutine to call resettimer.
			// Wait until the run is complete.
			osyield()
		case timerWaiting, timerModifying, timerModifiedEarlier, timerModifiedLater, timerMoving:
			// Called resettimer on active timer.
			badTimer()
		default:
			badTimer()
		}
	}
}

// cleantimers cleans up the head of the timer queue. This speeds up
// programs that create and delete timers; leaving them in the heap
// slows down addtimer. Reports whether no timer problems were found.
// The caller must have locked the timers for pp.
func cleantimers(pp *p) bool {
	for {
		if len(pp.timers) == 0 {
			return true
		}
		t := pp.timers[0]
		if t.pp.ptr() != pp {
			throw("cleantimers: bad p")
		}
		switch s := atomic.Load(&t.status); s {
		case timerDeleted:
			if !atomic.Cas(&t.status, s, timerRemoving) {
				continue
			}
			if !dodeltimer0(pp) {
				return false
			}
			if !atomic.Cas(&t.status, timerRemoving, timerRemoved) {
				return false
			}
			atomic.Xadd(&pp.deletedTimers, -1)
		case timerModifiedEarlier, timerModifiedLater:
			if !atomic.Cas(&t.status, s, timerMoving) {
				continue
			}
			// Now we can change the when field.
			t.when = t.nextwhen
			// Move t to the right position.
			if !dodeltimer0(pp) {
				return false
			}
			if !doaddtimer(pp, t) {
				return false
			}
			if s == timerModifiedEarlier {
				atomic.Xadd(&pp.adjustTimers, -1)
			}
			if !atomic.Cas(&t.status, timerMoving, timerWaiting) {
				return false
			}
		default:
			// Head of timers does not need adjustment.
			return true
		}
	}
}

// moveTimers moves a slice of timers to pp. The slice has been taken
// from a different P.
// This is currently called when the world is stopped, but the caller
// is expected to have locked the timers for pp.
func moveTimers(pp *p, timers []*timer) {
	for _, t := range timers {
	loop:
		for {
			switch s := atomic.Load(&t.status); s {
			case timerWaiting:
				t.pp = 0
				if !doaddtimer(pp, t) {
					badTimer()
				}
				break loop
			case timerModifiedEarlier, timerModifiedLater:
				if !atomic.Cas(&t.status, s, timerMoving) {
					continue
				}
				t.when = t.nextwhen
				t.pp = 0
				if !doaddtimer(pp, t) {
					badTimer()
				}
				if !atomic.Cas(&t.status, timerMoving, timerWaiting) {
					badTimer()
				}
				break loop
			case timerDeleted:
				if !atomic.Cas(&t.status, s, timerRemoved) {
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

// adjusttimers looks through the timers in the current P's heap for
// any timers that have been modified to run earlier, and puts them in
// the correct place in the heap. While looking for those timers,
// it also moves timers that have been modified to run later,
// and removes deleted timers. The caller must have locked the timers for pp.
func adjusttimers(pp *p) {
	if len(pp.timers) == 0 {
		return
	}
	if atomic.Load(&pp.adjustTimers) == 0 {
		if verifyTimers {
			verifyTimerHeap(pp.timers)
		}
		return
	}
	var moved []*timer
loop:
	for i := 0; i < len(pp.timers); i++ {
		t := pp.timers[i]
		if t.pp.ptr() != pp {
			throw("adjusttimers: bad p")
		}
		switch s := atomic.Load(&t.status); s {
		case timerDeleted:
			if atomic.Cas(&t.status, s, timerRemoving) {
				if !dodeltimer(pp, i) {
					badTimer()
				}
				if !atomic.Cas(&t.status, timerRemoving, timerRemoved) {
					badTimer()
				}
				atomic.Xadd(&pp.deletedTimers, -1)
				// Look at this heap position again.
				i--
			}
		case timerModifiedEarlier, timerModifiedLater:
			if atomic.Cas(&t.status, s, timerMoving) {
				// Now we can change the when field.
				t.when = t.nextwhen
				// Take t off the heap, and hold onto it.
				// We don't add it back yet because the
				// heap manipulation could cause our
				// loop to skip some other timer.
				if !dodeltimer(pp, i) {
					badTimer()
				}
				moved = append(moved, t)
				if s == timerModifiedEarlier {
					if n := atomic.Xadd(&pp.adjustTimers, -1); int32(n) <= 0 {
						break loop
					}
				}
				// Look at this heap position again.
				i--
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
		addAdjustedTimers(pp, moved)
	}

	if verifyTimers {
		verifyTimerHeap(pp.timers)
	}
}

// addAdjustedTimers adds any timers we adjusted in adjusttimers
// back to the timer heap.
func addAdjustedTimers(pp *p, moved []*timer) {
	for _, t := range moved {
		if !doaddtimer(pp, t) {
			badTimer()
		}
		if !atomic.Cas(&t.status, timerMoving, timerWaiting) {
			badTimer()
		}
	}
}

// nobarrierWakeTime looks at P's timers and returns the time when we
// should wake up the netpoller. It returns 0 if there are no timers.
// This function is invoked when dropping a P, and must run without
// any write barriers. Therefore, if there are any timers that needs
// to be moved earlier, it conservatively returns the current time.
// The netpoller M will wake up and adjust timers before sleeping again.
//go:nowritebarrierrec
func nobarrierWakeTime(pp *p) int64 {
	lock(&pp.timersLock)
	ret := int64(0)
	if len(pp.timers) > 0 {
		if atomic.Load(&pp.adjustTimers) > 0 {
			ret = nanotime()
		} else {
			ret = pp.timers[0].when
		}
	}
	unlock(&pp.timersLock)
	return ret
}

// runtimer examines the first timer in timers. If it is ready based on now,
// it runs the timer and removes or updates it.
// Returns 0 if it ran a timer, -1 if there are no more timers, or the time
// when the first timer should run.
// The caller must have locked the timers for pp.
// If a timer is run, this will temporarily unlock the timers.
//go:systemstack
func runtimer(pp *p, now int64) int64 {
	for {
		t := pp.timers[0]
		if t.pp.ptr() != pp {
			throw("runtimer: bad p")
		}
		switch s := atomic.Load(&t.status); s {
		case timerWaiting:
			if t.when > now {
				// Not ready to run.
				return t.when
			}

			if !atomic.Cas(&t.status, s, timerRunning) {
				continue
			}
			// Note that runOneTimer may temporarily unlock
			// pp.timersLock.
			runOneTimer(pp, t, now)
			return 0

		case timerDeleted:
			if !atomic.Cas(&t.status, s, timerRemoving) {
				continue
			}
			if !dodeltimer0(pp) {
				badTimer()
			}
			if !atomic.Cas(&t.status, timerRemoving, timerRemoved) {
				badTimer()
			}
			atomic.Xadd(&pp.deletedTimers, -1)
			if len(pp.timers) == 0 {
				return -1
			}

		case timerModifiedEarlier, timerModifiedLater:
			if !atomic.Cas(&t.status, s, timerMoving) {
				continue
			}
			t.when = t.nextwhen
			if !dodeltimer0(pp) {
				badTimer()
			}
			if !doaddtimer(pp, t) {
				badTimer()
			}
			if s == timerModifiedEarlier {
				atomic.Xadd(&pp.adjustTimers, -1)
			}
			if !atomic.Cas(&t.status, timerMoving, timerWaiting) {
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
// The caller must have locked the timers for pp.
// This will temporarily unlock the timers while running the timer function.
//go:systemstack
func runOneTimer(pp *p, t *timer, now int64) {
	if raceenabled {
		ppcur := getg().m.p.ptr()
		if ppcur.timerRaceCtx == 0 {
			ppcur.timerRaceCtx = racegostart(funcPC(runtimer) + sys.PCQuantum)
		}
		raceacquirectx(ppcur.timerRaceCtx, unsafe.Pointer(t))
	}

	f := t.f
	arg := t.arg
	seq := t.seq

	if t.period > 0 {
		// Leave in heap but adjust next time to fire.
		delta := t.when - now
		t.when += t.period * (1 + -delta/t.period)
		if !siftdownTimer(pp.timers, 0) {
			badTimer()
		}
		if !atomic.Cas(&t.status, timerRunning, timerWaiting) {
			badTimer()
		}
	} else {
		// Remove from heap.
		if !dodeltimer0(pp) {
			badTimer()
		}
		if !atomic.Cas(&t.status, timerRunning, timerNoStatus) {
			badTimer()
		}
	}

	if raceenabled {
		// Temporarily use the current P's racectx for g0.
		gp := getg()
		if gp.racectx != 0 {
			throw("runOneTimer: unexpected racectx")
		}
		gp.racectx = gp.m.p.ptr().timerRaceCtx
	}

	unlock(&pp.timersLock)

	f(arg, seq)

	lock(&pp.timersLock)

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
func clearDeletedTimers(pp *p) {
	cdel := int32(0)
	cearlier := int32(0)
	to := 0
	changedHeap := false
	timers := pp.timers
nextTimer:
	for _, t := range timers {
		for {
			switch s := atomic.Load(&t.status); s {
			case timerWaiting:
				if changedHeap {
					timers[to] = t
					siftupTimer(timers, to)
				}
				to++
				continue nextTimer
			case timerModifiedEarlier, timerModifiedLater:
				if atomic.Cas(&t.status, s, timerMoving) {
					t.when = t.nextwhen
					timers[to] = t
					siftupTimer(timers, to)
					to++
					changedHeap = true
					if !atomic.Cas(&t.status, timerMoving, timerWaiting) {
						badTimer()
					}
					if s == timerModifiedEarlier {
						cearlier++
					}
					continue nextTimer
				}
			case timerDeleted:
				if atomic.Cas(&t.status, s, timerRemoving) {
					t.pp = 0
					cdel++
					if !atomic.Cas(&t.status, timerRemoving, timerRemoved) {
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

	timers = timers[:to]
	if verifyTimers {
		verifyTimerHeap(timers)
	}
	pp.timers = timers
	atomic.Xadd(&pp.deletedTimers, -cdel)
	atomic.Xadd(&pp.adjustTimers, -cearlier)
}

// verifyTimerHeap verifies that the timer heap is in a valid state.
// This is only for debugging, and is only called if verifyTimers is true.
// The caller must have locked the timers.
func verifyTimerHeap(timers []*timer) {
	for i, t := range timers {
		if i == 0 {
			// First timer has no parent.
			continue
		}

		// The heap is 4-ary. See siftupTimer and siftdownTimer.
		p := (i - 1) / 4
		if t.when < timers[p].when {
			print("bad timer heap at ", i, ": ", p, ": ", timers[p].when, ", ", i, ": ", t.when, "\n")
			throw("bad timer heap")
		}
	}
}

func timejump() *p {
	if faketime == 0 {
		return nil
	}

	// Nothing is running, so we can look at all the P's.
	// Determine a timer bucket with minimum when.
	var (
		minT    *timer
		minWhen int64
		minP    *p
	)
	for _, pp := range allp {
		if pp.status != _Pidle && pp.status != _Pdead {
			throw("non-idle P in timejump")
		}
		if len(pp.timers) == 0 {
			continue
		}
		c := pp.adjustTimers
		for _, t := range pp.timers {
			switch s := atomic.Load(&t.status); s {
			case timerWaiting:
				if minT == nil || t.when < minWhen {
					minT = t
					minWhen = t.when
					minP = pp
				}
			case timerModifiedEarlier, timerModifiedLater:
				if minT == nil || t.nextwhen < minWhen {
					minT = t
					minWhen = t.nextwhen
					minP = pp
				}
				if s == timerModifiedEarlier {
					c--
				}
			case timerRunning, timerModifying, timerMoving:
				badTimer()
			}
			// The timers are sorted, so we only have to check
			// the first timer for each P, unless there are
			// some timerModifiedEarlier timers. The number
			// of timerModifiedEarlier timers is in the adjustTimers
			// field, used to initialize c, above.
			if c == 0 {
				break
			}
		}
	}

	if minT == nil || minWhen <= faketime {
		return nil
	}

	faketime = minWhen
	return minP
}

// timeSleepUntil returns the time when the next timer should fire.
// This is only called by sysmon.
func timeSleepUntil() int64 {
	next := int64(maxWhen)

	// Prevent allp slice changes. This is like retake.
	lock(&allpLock)
	for _, pp := range allp {
		if pp == nil {
			// This can happen if procresize has grown
			// allp but not yet created new Ps.
			continue
		}

		lock(&pp.timersLock)
		c := atomic.Load(&pp.adjustTimers)
		for _, t := range pp.timers {
			switch s := atomic.Load(&t.status); s {
			case timerWaiting:
				if t.when < next {
					next = t.when
				}
			case timerModifiedEarlier, timerModifiedLater:
				if t.nextwhen < next {
					next = t.nextwhen
				}
				if s == timerModifiedEarlier {
					c--
				}
			}
			// The timers are sorted, so we only have to check
			// the first timer for each P, unless there are
			// some timerModifiedEarlier timers. The number
			// of timerModifiedEarlier timers is in the adjustTimers
			// field, used to initialize c, above.
			//
			// We don't worry about cases like timerModifying.
			// New timers can show up at any time,
			// so this function is necessarily imprecise.
			// Do a signed check here since we aren't
			// synchronizing the read of pp.adjustTimers
			// with the check of a timer status.
			if int32(c) <= 0 {
				break
			}
		}
		unlock(&pp.timersLock)
	}
	unlock(&allpLock)

	return next
}

// Heap maintenance algorithms.
// These algorithms check for slice index errors manually.
// Slice index error can happen if the program is using racy
// access to timers. We don't want to panic here, because
// it will cause the program to crash with a mysterious
// "panic holding locks" message. Instead, we panic while not
// holding a lock.

func siftupTimer(t []*timer, i int) bool {
	if i >= len(t) {
		return false
	}
	when := t[i].when
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
	return true
}

func siftdownTimer(t []*timer, i int) bool {
	n := len(t)
	if i >= n {
		return false
	}
	when := t[i].when
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
	return true
}

// badTimer is called if the timer data structures have been corrupted,
// presumably due to racy use by the program. We panic here rather than
// panicing due to invalid slice access while holding locks.
// See issue #25686.
func badTimer() {
	panic(errorString("racy use of timers"))
}

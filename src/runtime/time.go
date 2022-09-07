// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Time-related runtime and pieces of package time.

package runtime

import (
	"internal/abi"
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
	//
	// when must be positive on an active timer.
	when   int64
	period int64
	f      func(any, uintptr)
	arg    any
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
//   timerWaiting         -> timerModifying -> timerDeleted
//   timerModifiedEarlier -> timerModifying -> timerDeleted
//   timerModifiedLater   -> timerModifying -> timerDeleted
//   timerNoStatus        -> do nothing
//   timerDeleted         -> do nothing
//   timerRemoving        -> do nothing
//   timerRemoved         -> do nothing
//   timerRunning         -> wait until status changes
//   timerMoving          -> wait until status changes
//   timerModifying       -> wait until status changes
// modtimer:
//   timerWaiting    -> timerModifying -> timerModifiedXX
//   timerModifiedXX -> timerModifying -> timerModifiedYY
//   timerNoStatus   -> timerModifying -> timerWaiting
//   timerRemoved    -> timerModifying -> timerWaiting
//   timerDeleted    -> timerModifying -> timerModifiedXX
//   timerRunning    -> wait until status changes
//   timerMoving     -> wait until status changes
//   timerRemoving   -> wait until status changes
//   timerModifying  -> wait until status changes
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
//
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
	if t.nextwhen < 0 { // check for overflow.
		t.nextwhen = maxWhen
	}
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
//
//go:linkname startTimer time.startTimer
func startTimer(t *timer) {
	if raceenabled {
		racerelease(unsafe.Pointer(t))
	}
	addtimer(t)
}

// stopTimer stops a timer.
// It reports whether t was stopped before being run.
//
//go:linkname stopTimer time.stopTimer
func stopTimer(t *timer) bool {
	return deltimer(t)
}

// resetTimer resets an inactive timer, adding it to the heap.
//
// Reports whether the timer was modified before it was run.
//
//go:linkname resetTimer time.resetTimer
func resetTimer(t *timer, when int64) bool {
	if raceenabled {
		racerelease(unsafe.Pointer(t))
	}
	return resettimer(t, when)
}

// modTimer modifies an existing timer.
//
//go:linkname modTimer time.modTimer
func modTimer(t *timer, when, period int64, f func(any, uintptr), arg any, seq uintptr) {
	modtimer(t, when, period, f, arg, seq)
}

// Go runtime.

// Ready the goroutine arg.
func goroutineReady(arg any, seq uintptr) {
	goready(arg.(*g), 0)
}

// addtimer adds a timer to the current P.
// This should only be called with a newly created timer.
// That avoids the risk of changing the when field of a timer in some P's heap,
// which could cause the heap to become unsorted.
func addtimer(t *timer) {
	// when must be positive. A negative value will cause runtimer to
	// overflow during its delta calculation and never expire other runtime
	// timers. Zero will cause checkTimers to fail to notice the timer.
	if t.when <= 0 {
		throw("timer when must be positive")
	}
	if t.period < 0 {
		throw("timer period must be non-negative")
	}
	if t.status != timerNoStatus {
		throw("addtimer called with initialized timer")
	}
	t.status = timerWaiting

	when := t.when

	// Disable preemption while using pp to avoid changing another P's heap.
	mp := acquirem()

	pp := getg().m.p.ptr()
	lock(&pp.timersLock)
	cleantimers(pp)
	doaddtimer(pp, t)
	unlock(&pp.timersLock)

	wakeNetPoller(when)

	releasem(mp)
}

// doaddtimer adds t to the current P's heap.
// The caller must have locked the timers for pp.
func doaddtimer(pp *p, t *timer) {
	// Timers rely on the network poller, so make sure the poller
	// has started.
	if netpollInited.Load() == 0 {
		netpollGenericInit()
	}

	if t.pp != 0 {
		throw("doaddtimer: P already set in timer")
	}
	t.pp.set(pp)
	i := len(pp.timers)
	pp.timers = append(pp.timers, t)
	siftupTimer(pp.timers, i)
	if t == pp.timers[0] {
		pp.timer0When.Store(t.when)
	}
	pp.numTimers.Add(1)
}

// deltimer deletes the timer t. It may be on some other P, so we can't
// actually remove it from the timers heap. We can only mark it as deleted.
// It will be removed in due course by the P whose heap it is on.
// Reports whether the timer was removed before it was run.
func deltimer(t *timer) bool {
	for {
		switch s := atomic.Load(&t.status); s {
		case timerWaiting, timerModifiedLater:
			// Prevent preemption while the timer is in timerModifying.
			// This could lead to a self-deadlock. See #38070.
			mp := acquirem()
			if atomic.Cas(&t.status, s, timerModifying) {
				// Must fetch t.pp before changing status,
				// as cleantimers in another goroutine
				// can clear t.pp of a timerDeleted timer.
				tpp := t.pp.ptr()
				if !atomic.Cas(&t.status, timerModifying, timerDeleted) {
					badTimer()
				}
				releasem(mp)
				tpp.deletedTimers.Add(1)
				// Timer was not yet run.
				return true
			} else {
				releasem(mp)
			}
		case timerModifiedEarlier:
			// Prevent preemption while the timer is in timerModifying.
			// This could lead to a self-deadlock. See #38070.
			mp := acquirem()
			if atomic.Cas(&t.status, s, timerModifying) {
				// Must fetch t.pp before setting status
				// to timerDeleted.
				tpp := t.pp.ptr()
				if !atomic.Cas(&t.status, timerModifying, timerDeleted) {
					badTimer()
				}
				releasem(mp)
				tpp.deletedTimers.Add(1)
				// Timer was not yet run.
				return true
			} else {
				releasem(mp)
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
			// Wait for the other call to complete.
			osyield()
		default:
			badTimer()
		}
	}
}

// dodeltimer removes timer i from the current P's heap.
// We are locked on the P when this is called.
// It returns the smallest changed index in pp.timers.
// The caller must have locked the timers for pp.
func dodeltimer(pp *p, i int) int {
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
	smallestChanged := i
	if i != last {
		// Moving to i may have moved the last timer to a new parent,
		// so sift up to preserve the heap guarantee.
		smallestChanged = siftupTimer(pp.timers, i)
		siftdownTimer(pp.timers, i)
	}
	if i == 0 {
		updateTimer0When(pp)
	}
	n := pp.numTimers.Add(-1)
	if n == 0 {
		// If there are no timers, then clearly none are modified.
		pp.timerModifiedEarliest.Store(0)
	}
	return smallestChanged
}

// dodeltimer0 removes timer 0 from the current P's heap.
// We are locked on the P when this is called.
// It reports whether it saw no problems due to races.
// The caller must have locked the timers for pp.
func dodeltimer0(pp *p) {
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
	if last > 0 {
		siftdownTimer(pp.timers, 0)
	}
	updateTimer0When(pp)
	n := pp.numTimers.Add(-1)
	if n == 0 {
		// If there are no timers, then clearly none are modified.
		pp.timerModifiedEarliest.Store(0)
	}
}

// modtimer modifies an existing timer.
// This is called by the netpoll code or time.Ticker.Reset or time.Timer.Reset.
// Reports whether the timer was modified before it was run.
func modtimer(t *timer, when, period int64, f func(any, uintptr), arg any, seq uintptr) bool {
	if when <= 0 {
		throw("timer when must be positive")
	}
	if period < 0 {
		throw("timer period must be non-negative")
	}

	status := uint32(timerNoStatus)
	wasRemoved := false
	var pending bool
	var mp *m
loop:
	for {
		switch status = atomic.Load(&t.status); status {
		case timerWaiting, timerModifiedEarlier, timerModifiedLater:
			// Prevent preemption while the timer is in timerModifying.
			// This could lead to a self-deadlock. See #38070.
			mp = acquirem()
			if atomic.Cas(&t.status, status, timerModifying) {
				pending = true // timer not yet run
				break loop
			}
			releasem(mp)
		case timerNoStatus, timerRemoved:
			// Prevent preemption while the timer is in timerModifying.
			// This could lead to a self-deadlock. See #38070.
			mp = acquirem()

			// Timer was already run and t is no longer in a heap.
			// Act like addtimer.
			if atomic.Cas(&t.status, status, timerModifying) {
				wasRemoved = true
				pending = false // timer already run or stopped
				break loop
			}
			releasem(mp)
		case timerDeleted:
			// Prevent preemption while the timer is in timerModifying.
			// This could lead to a self-deadlock. See #38070.
			mp = acquirem()
			if atomic.Cas(&t.status, status, timerModifying) {
				t.pp.ptr().deletedTimers.Add(-1)
				pending = false // timer already stopped
				break loop
			}
			releasem(mp)
		case timerRunning, timerRemoving, timerMoving:
			// The timer is being run or moved, by a different P.
			// Wait for it to complete.
			osyield()
		case timerModifying:
			// Multiple simultaneous calls to modtimer.
			// Wait for the other call to complete.
			osyield()
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
		pp := getg().m.p.ptr()
		lock(&pp.timersLock)
		doaddtimer(pp, t)
		unlock(&pp.timersLock)
		if !atomic.Cas(&t.status, timerModifying, timerWaiting) {
			badTimer()
		}
		releasem(mp)
		wakeNetPoller(when)
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

		tpp := t.pp.ptr()

		if newStatus == timerModifiedEarlier {
			updateTimerModifiedEarliest(tpp, when)
		}

		// Set the new status of the timer.
		if !atomic.Cas(&t.status, timerModifying, newStatus) {
			badTimer()
		}
		releasem(mp)

		// If the new status is earlier, wake up the poller.
		if newStatus == timerModifiedEarlier {
			wakeNetPoller(when)
		}
	}

	return pending
}

// resettimer resets the time when a timer should fire.
// If used for an inactive timer, the timer will become active.
// This should be called instead of addtimer if the timer value has been,
// or may have been, used previously.
// Reports whether the timer was modified before it was run.
func resettimer(t *timer, when int64) bool {
	return modtimer(t, when, t.period, t.f, t.arg, t.seq)
}

// cleantimers cleans up the head of the timer queue. This speeds up
// programs that create and delete timers; leaving them in the heap
// slows down addtimer. Reports whether no timer problems were found.
// The caller must have locked the timers for pp.
func cleantimers(pp *p) {
	gp := getg()
	for {
		if len(pp.timers) == 0 {
			return
		}

		// This loop can theoretically run for a while, and because
		// it is holding timersLock it cannot be preempted.
		// If someone is trying to preempt us, just return.
		// We can clean the timers later.
		if gp.preemptStop {
			return
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
			dodeltimer0(pp)
			if !atomic.Cas(&t.status, timerRemoving, timerRemoved) {
				badTimer()
			}
			pp.deletedTimers.Add(-1)
		case timerModifiedEarlier, timerModifiedLater:
			if !atomic.Cas(&t.status, s, timerMoving) {
				continue
			}
			// Now we can change the when field.
			t.when = t.nextwhen
			// Move t to the right position.
			dodeltimer0(pp)
			doaddtimer(pp, t)
			if !atomic.Cas(&t.status, timerMoving, timerWaiting) {
				badTimer()
			}
		default:
			// Head of timers does not need adjustment.
			return
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
				if !atomic.Cas(&t.status, s, timerMoving) {
					continue
				}
				t.pp = 0
				doaddtimer(pp, t)
				if !atomic.Cas(&t.status, timerMoving, timerWaiting) {
					badTimer()
				}
				break loop
			case timerModifiedEarlier, timerModifiedLater:
				if !atomic.Cas(&t.status, s, timerMoving) {
					continue
				}
				t.when = t.nextwhen
				t.pp = 0
				doaddtimer(pp, t)
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
func adjusttimers(pp *p, now int64) {
	// If we haven't yet reached the time of the first timerModifiedEarlier
	// timer, don't do anything. This speeds up programs that adjust
	// a lot of timers back and forth if the timers rarely expire.
	// We'll postpone looking through all the adjusted timers until
	// one would actually expire.
	first := pp.timerModifiedEarliest.Load()
	if first == 0 || first > now {
		if verifyTimers {
			verifyTimerHeap(pp)
		}
		return
	}

	// We are going to clear all timerModifiedEarlier timers.
	pp.timerModifiedEarliest.Store(0)

	var moved []*timer
	for i := 0; i < len(pp.timers); i++ {
		t := pp.timers[i]
		if t.pp.ptr() != pp {
			throw("adjusttimers: bad p")
		}
		switch s := atomic.Load(&t.status); s {
		case timerDeleted:
			if atomic.Cas(&t.status, s, timerRemoving) {
				changed := dodeltimer(pp, i)
				if !atomic.Cas(&t.status, timerRemoving, timerRemoved) {
					badTimer()
				}
				pp.deletedTimers.Add(-1)
				// Go back to the earliest changed heap entry.
				// "- 1" because the loop will add 1.
				i = changed - 1
			}
		case timerModifiedEarlier, timerModifiedLater:
			if atomic.Cas(&t.status, s, timerMoving) {
				// Now we can change the when field.
				t.when = t.nextwhen
				// Take t off the heap, and hold onto it.
				// We don't add it back yet because the
				// heap manipulation could cause our
				// loop to skip some other timer.
				changed := dodeltimer(pp, i)
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
		addAdjustedTimers(pp, moved)
	}

	if verifyTimers {
		verifyTimerHeap(pp)
	}
}

// addAdjustedTimers adds any timers we adjusted in adjusttimers
// back to the timer heap.
func addAdjustedTimers(pp *p, moved []*timer) {
	for _, t := range moved {
		doaddtimer(pp, t)
		if !atomic.Cas(&t.status, timerMoving, timerWaiting) {
			badTimer()
		}
	}
}

// nobarrierWakeTime looks at P's timers and returns the time when we
// should wake up the netpoller. It returns 0 if there are no timers.
// This function is invoked when dropping a P, and must run without
// any write barriers.
//
//go:nowritebarrierrec
func nobarrierWakeTime(pp *p) int64 {
	next := pp.timer0When.Load()
	nextAdj := pp.timerModifiedEarliest.Load()
	if next == 0 || (nextAdj != 0 && nextAdj < next) {
		next = nextAdj
	}
	return next
}

// runtimer examines the first timer in timers. If it is ready based on now,
// it runs the timer and removes or updates it.
// Returns 0 if it ran a timer, -1 if there are no more timers, or the time
// when the first timer should run.
// The caller must have locked the timers for pp.
// If a timer is run, this will temporarily unlock the timers.
//
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
			dodeltimer0(pp)
			if !atomic.Cas(&t.status, timerRemoving, timerRemoved) {
				badTimer()
			}
			pp.deletedTimers.Add(-1)
			if len(pp.timers) == 0 {
				return -1
			}

		case timerModifiedEarlier, timerModifiedLater:
			if !atomic.Cas(&t.status, s, timerMoving) {
				continue
			}
			t.when = t.nextwhen
			dodeltimer0(pp)
			doaddtimer(pp, t)
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
//
//go:systemstack
func runOneTimer(pp *p, t *timer, now int64) {
	if raceenabled {
		ppcur := getg().m.p.ptr()
		if ppcur.timerRaceCtx == 0 {
			ppcur.timerRaceCtx = racegostart(abi.FuncPCABIInternal(runtimer) + sys.PCQuantum)
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
		if t.when < 0 { // check for overflow.
			t.when = maxWhen
		}
		siftdownTimer(pp.timers, 0)
		if !atomic.Cas(&t.status, timerRunning, timerWaiting) {
			badTimer()
		}
		updateTimer0When(pp)
	} else {
		// Remove from heap.
		dodeltimer0(pp)
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
	// We are going to clear all timerModifiedEarlier timers.
	// Do this now in case new ones show up while we are looping.
	pp.timerModifiedEarliest.Store(0)

	cdel := int32(0)
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

	pp.deletedTimers.Add(-cdel)
	pp.numTimers.Add(-cdel)

	timers = timers[:to]
	pp.timers = timers
	updateTimer0When(pp)

	if verifyTimers {
		verifyTimerHeap(pp)
	}
}

// verifyTimerHeap verifies that the timer heap is in a valid state.
// This is only for debugging, and is only called if verifyTimers is true.
// The caller must have locked the timers.
func verifyTimerHeap(pp *p) {
	for i, t := range pp.timers {
		if i == 0 {
			// First timer has no parent.
			continue
		}

		// The heap is 4-ary. See siftupTimer and siftdownTimer.
		p := (i - 1) / 4
		if t.when < pp.timers[p].when {
			print("bad timer heap at ", i, ": ", p, ": ", pp.timers[p].when, ", ", i, ": ", t.when, "\n")
			throw("bad timer heap")
		}
	}
	if numTimers := int(pp.numTimers.Load()); len(pp.timers) != numTimers {
		println("timer heap len", len(pp.timers), "!= numTimers", numTimers)
		throw("bad timer heap len")
	}
}

// updateTimer0When sets the P's timer0When field.
// The caller must have locked the timers for pp.
func updateTimer0When(pp *p) {
	if len(pp.timers) == 0 {
		pp.timer0When.Store(0)
	} else {
		pp.timer0When.Store(pp.timers[0].when)
	}
}

// updateTimerModifiedEarliest updates the recorded nextwhen field of the
// earlier timerModifiedEarier value.
// The timers for pp will not be locked.
func updateTimerModifiedEarliest(pp *p, nextwhen int64) {
	for {
		old := pp.timerModifiedEarliest.Load()
		if old != 0 && int64(old) < nextwhen {
			return
		}

		if pp.timerModifiedEarliest.CompareAndSwap(old, nextwhen) {
			return
		}
	}
}

// timeSleepUntil returns the time when the next timer should fire. Returns
// maxWhen if there are no timers.
// This is only called by sysmon and checkdead.
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

		w := pp.timer0When.Load()
		if w != 0 && w < next {
			next = w
		}

		w = pp.timerModifiedEarliest.Load()
		if w != 0 && w < next {
			next = w
		}
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

// siftupTimer puts the timer at position i in the right place
// in the heap by moving it up toward the top of the heap.
// It returns the smallest changed index.
func siftupTimer(t []*timer, i int) int {
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
func siftdownTimer(t []*timer, i int) {
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

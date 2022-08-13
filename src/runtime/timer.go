// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Timer-related runtime and pieces of package time.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

// Code outside this file has to be careful in using a timer value.
//
// The pp, status, and nextwhen fields may only be used by code in this file.
//
// Code that creates a new timer value can set the when, period, f,
// arg, and seq fields.
// A new timer value may call timer.add (called by time.startTimer).
// After doing that no fields may be touched.
//
// An active timer (one that has been called timer.add()) may be
// passed to timer.delete() (time.stopTimer), after which it is no longer an
// active timer. It is an inactive timer.
// In an inactive timer the period, f, arg, and seq fields may be modified,
// but not the when field.
// It's OK to just drop an inactive timer and let the GC collect it.
// It's not OK to call an inactive timer add() method.
// Only newly allocated timer can call add() method.
//
// An active timer may be passed to timer.modify(). No fields may be touched.
// It remains an active timer.
//
// An inactive timer may be passed to timer.reset() to turn into an
// active timer with an updated when field.
// It's OK to pass a newly allocated timer value to timer.reset().
//
// We don't permit calling timer.add()/timer.delete()/timer.modify()/timer.reset() simultaneously,
// but timing.adjustTimers() and timing.runTimer() can be called at the same time as any of those.
//
// Active timers live in heaps attached to P, in the timers field.
// Inactive timers live there too temporarily, until they are removed.
//

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
	status atomic.Uint32
}

// start adds the timer to the current P timing.
// This should only be called with a newly created timer.
// That avoids the risk of changing the when field of a timer in some P's heap,
// which could cause the heap to become unsorted.
func (t *timer) start() {
	// when must be positive. A negative value will cause timing.runTimer to
	// overflow during its delta calculation and never expire other runtime
	// timers. Zero will cause checkTimers to fail to notice the timer.
	if t.when <= 0 {
		throw("timer.add: when must be positive")
	}
	if t.period < 0 {
		throw("timer.add: period must be non-negative")
	}
	if t.status.Load() != timerNoStatus {
		throw("timer.add: called with initialized timer")
	}
	t.status.Store(timerWaiting)

	when := t.when

	// Disable preemption while using pp to avoid changing another P's heap.
	mp := acquirem()

	pp := getg().m.p.ptr()
	lock(&pp.timing.timersLock)
	pp.timing.cleanTimers()
	pp.timing.addTimer(t)
	unlock(&pp.timing.timersLock)

	wakeNetPoller(when)

	releasem(mp)
}

// stop stop the timer t. It may be on some other P, so we can't
// actually remove it from the timers heap. We can only mark it as deleted.
// It will be removed in due course by the P whose heap it is on.
// Reports whether the timer was removed before it was run.
func (t *timer) stop() bool {
	for {
		switch s := t.status.Load(); s {
		case timerWaiting, timerModifiedLater:
			// Prevent preemption while the timer is in timerModifying.
			// This could lead to a self-deadlock. See #38070.
			mp := acquirem()
			if t.status.CompareAndSwap(s, timerModifying) {
				// Must fetch t.pp before changing status,
				// as cleanTimers in another goroutine
				// can clear t.pp of a timerDeleted timer.
				tpp := t.pp.ptr()
				if !t.status.CompareAndSwap(timerModifying, timerDeleted) {
					badTimer()
				}
				releasem(mp)
				tpp.timing.deletedTimers.Add(1)
				// Timer was not yet run.
				return true
			} else {
				releasem(mp)
			}
		case timerModifiedEarlier:
			// Prevent preemption while the timer is in timerModifying.
			// This could lead to a self-deadlock. See #38070.
			mp := acquirem()
			if t.status.CompareAndSwap(s, timerModifying) {
				// Must fetch t.pp before setting status
				// to timerDeleted.
				tpp := t.pp.ptr()
				if !t.status.CompareAndSwap(timerModifying, timerDeleted) {
					badTimer()
				}
				releasem(mp)
				tpp.timing.deletedTimers.Add(1)
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
			// Simultaneous calls to timer.delete() and timer.modify().
			// Wait for the other call to complete.
			osyield()
		default:
			badTimer()
		}
	}
}

// reset resets the time when a timer should fire.
// If used for an inactive timer, the timer will become active.
// This should be called instead of add() method if the timer value has been,
// or may have been, used previously.
// Reports whether the timer was modified before it was run.
func (t *timer) reset(when int64) bool {
	return t.modify(when, t.period, t.f, t.arg, t.seq)
}

// modify modifies an existing timer.
// This is called by the netpoll code or time.Ticker.Reset or time.Timer.Reset.
// Reports whether the timer was modified before it was run.
func (t *timer) modify(when, period int64, f func(any, uintptr), arg any, seq uintptr) bool {
	if when <= 0 {
		throw("timer.modify: when must be positive")
	}
	if period < 0 {
		throw("timer.modify: period must be non-negative")
	}

	var wasRemoved = false
	var pending bool
	var mp *m
loop:
	for {
		var status = t.status.Load()
		switch status {
		case timerWaiting, timerModifiedEarlier, timerModifiedLater:
			// Prevent preemption while the timer is in timerModifying.
			// This could lead to a self-deadlock. See #38070.
			mp = acquirem()
			if t.status.CompareAndSwap(status, timerModifying) {
				pending = true // timer not yet run
				break loop
			}
			releasem(mp)
		case timerNoStatus, timerRemoved:
			// Prevent preemption while the timer is in timerModifying.
			// This could lead to a self-deadlock. See #38070.
			mp = acquirem()

			// Timer was already run and t is no longer in a heap.
			// Act like timer.add().
			if t.status.CompareAndSwap(status, timerModifying) {
				wasRemoved = true
				pending = false // timer already run or stopped
				break loop
			}
			releasem(mp)
		case timerDeleted:
			// Prevent preemption while the timer is in timerModifying.
			// This could lead to a self-deadlock. See #38070.
			mp = acquirem()
			if t.status.CompareAndSwap(status, timerModifying) {
				tpp := t.pp.ptr()
				tpp.timing.deletedTimers.Add(-1)
				pending = false // timer already stopped
				break loop
			}
			releasem(mp)
		case timerRunning, timerRemoving, timerMoving:
			// The timer is being run or moved, by a different P.
			// Wait for it to complete.
			osyield()
		case timerModifying:
			// Multiple simultaneous calls to timer.modify().
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
		lock(&pp.timing.timersLock)
		pp.timing.addTimer(t)
		unlock(&pp.timing.timersLock)
		if !t.status.CompareAndSwap(timerModifying, timerWaiting) {
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
			tpp.timing.updateTimerModifiedEarliest(when)
		}

		// Set the new status of the timer.
		if !t.status.CompareAndSwap(timerModifying, newStatus) {
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

// startTimer adds t to the timer heap.
//
//go:linkname startTimer time.startTimer
func startTimer(t *timer) {
	if raceenabled {
		racerelease(unsafe.Pointer(t))
	}
	t.start()
}

// stopTimer stops a timer.
// It reports whether t was stopped before being run.
//
//go:linkname stopTimer time.stopTimer
func stopTimer(t *timer) bool {
	return t.stop()
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
	return t.reset(when)
}

// modTimer modifies an existing timer.
//
//go:linkname modTimer time.modTimer
func modTimer(t *timer, when, period int64, f func(any, uintptr), arg any, seq uintptr) {
	t.modify(when, period, f, arg, seq)
}

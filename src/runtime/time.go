// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Time-related runtime and pieces of package time.

package runtime

import (
	"internal/cpu"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// Temporary scaffolding while the new timer code is added.
const oldTimers = true

// Package time knows the layout of this structure.
// If this struct changes, adjust ../time/sleep.go:/runtimeTimer.
type timer struct {
	tb *timersBucket // the bucket the timer lives in (oldTimers)
	i  int           // heap index (oldTimers)

	// If this timer is on a heap, which P's heap it is on.
	// puintptr rather than *p to match uintptr in the versions
	// of this struct defined in other packages. (!oldTimers)
	pp puintptr

	// Timer wakes up at when, and then at when+period, ... (period > 0 only)
	// each time calling f(arg, now) in the timer goroutine, so f must be
	// a well-behaved function and not block.
	when   int64
	period int64
	f      func(interface{}, uintptr)
	arg    interface{}
	seq    uintptr

	// What to set the when field to in timerModifiedXX status. (!oldTimers)
	nextwhen int64

	// The status field holds one of the values below. (!oldTimers)
	status uint32
}

// timersLen is the length of timers array.
//
// Ideally, this would be set to GOMAXPROCS, but that would require
// dynamic reallocation
//
// The current value is a compromise between memory usage and performance
// that should cover the majority of GOMAXPROCS values used in the wild.
const timersLen = 64

// timers contains "per-P" timer heaps.
//
// Timers are queued into timersBucket associated with the current P,
// so each P may work with its own timers independently of other P instances.
//
// Each timersBucket may be associated with multiple P
// if GOMAXPROCS > timersLen.
var timers [timersLen]struct {
	timersBucket

	// The padding should eliminate false sharing
	// between timersBucket values.
	pad [cpu.CacheLinePadSize - unsafe.Sizeof(timersBucket{})%cpu.CacheLinePadSize]byte
}

func (t *timer) assignBucket() *timersBucket {
	id := uint8(getg().m.p.ptr().id) % timersLen
	t.tb = &timers[id].timersBucket
	return t.tb
}

//go:notinheap
type timersBucket struct {
	lock         mutex
	gp           *g
	created      bool
	sleeping     bool
	rescheduling bool
	sleepUntil   int64
	waitnote     note
	t            []*timer
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

// Package time APIs.
// Godoc uses the comments in package time, not these.

// time.now is implemented in assembly.

// timeSleep puts the current goroutine to sleep for at least ns nanoseconds.
//go:linkname timeSleep time.Sleep
func timeSleep(ns int64) {
	if oldTimers {
		timeSleepOld(ns)
		return
	}

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

func timeSleepOld(ns int64) {
	if ns <= 0 {
		return
	}

	gp := getg()
	t := gp.timer
	if t == nil {
		t = new(timer)
		gp.timer = t
	}
	*t = timer{}
	t.when = nanotime() + ns
	t.f = goroutineReady
	t.arg = gp
	tb := t.assignBucket()
	lock(&tb.lock)
	if !tb.addtimerLocked(t) {
		unlock(&tb.lock)
		badTimer()
	}
	goparkunlock(&tb.lock, waitReasonSleep, traceEvGoSleep, 3)
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
	if oldTimers {
		addtimerOld(t)
		return
	}

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

func addtimerOld(t *timer) {
	tb := t.assignBucket()
	lock(&tb.lock)
	ok := tb.addtimerLocked(t)
	unlock(&tb.lock)
	if !ok {
		badTimer()
	}
}

// Add a timer to the heap and start or kick timerproc if the new timer is
// earlier than any of the others.
// Timers are locked.
// Returns whether all is well: false if the data structure is corrupt
// due to user-level races.
func (tb *timersBucket) addtimerLocked(t *timer) bool {
	// when must never be negative; otherwise timerproc will overflow
	// during its delta calculation and never expire other runtime timers.
	if t.when < 0 {
		t.when = 1<<63 - 1
	}
	t.i = len(tb.t)
	tb.t = append(tb.t, t)
	if !siftupTimer(tb.t, t.i) {
		return false
	}
	if t.i == 0 {
		// siftup moved to top: new earliest deadline.
		if tb.sleeping && tb.sleepUntil > t.when {
			tb.sleeping = false
			notewakeup(&tb.waitnote)
		}
		if tb.rescheduling {
			tb.rescheduling = false
			goready(tb.gp, 0)
		}
		if !tb.created {
			tb.created = true
			go timerproc(tb)
		}
	}
	return true
}

// deltimer deletes the timer t. It may be on some other P, so we can't
// actually remove it from the timers heap. We can only mark it as deleted.
// It will be removed in due course by the P whose heap it is on.
// Reports whether the timer was removed before it was run.
func deltimer(t *timer) bool {
	if oldTimers {
		return deltimerOld(t)
	}

	for {
		switch s := atomic.Load(&t.status); s {
		case timerWaiting, timerModifiedLater:
			if atomic.Cas(&t.status, s, timerDeleted) {
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

func deltimerOld(t *timer) bool {
	if t.tb == nil {
		// t.tb can be nil if the user created a timer
		// directly, without invoking startTimer e.g
		//    time.Ticker{C: c}
		// In this case, return early without any deletion.
		// See Issue 21874.
		return false
	}

	tb := t.tb

	lock(&tb.lock)
	removed, ok := tb.deltimerLocked(t)
	unlock(&tb.lock)
	if !ok {
		badTimer()
	}
	return removed
}

func (tb *timersBucket) deltimerLocked(t *timer) (removed, ok bool) {
	// t may not be registered anymore and may have
	// a bogus i (typically 0, if generated by Go).
	// Verify it before proceeding.
	i := t.i
	last := len(tb.t) - 1
	if i < 0 || i > last || tb.t[i] != t {
		return false, true
	}
	if i != last {
		tb.t[i] = tb.t[last]
		tb.t[i].i = i
	}
	tb.t[last] = nil
	tb.t = tb.t[:last]
	ok = true
	if i != last {
		if !siftupTimer(tb.t, i) {
			ok = false
		}
		if !siftdownTimer(tb.t, i) {
			ok = false
		}
	}
	return true, ok
}

// modtimer modifies an existing timer.
// This is called by the netpoll code.
func modtimer(t *timer, when, period int64, f func(interface{}, uintptr), arg interface{}, seq uintptr) {
	if oldTimers {
		modtimerOld(t, when, period, f, arg, seq)
		return
	}

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
			wasRemoved = true
			atomic.Store(&t.status, timerWaiting)
			break loop
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

func modtimerOld(t *timer, when, period int64, f func(interface{}, uintptr), arg interface{}, seq uintptr) {
	tb := t.tb

	lock(&tb.lock)
	_, ok := tb.deltimerLocked(t)
	if ok {
		t.when = when
		t.period = period
		t.f = f
		t.arg = arg
		t.seq = seq
		ok = tb.addtimerLocked(t)
	}
	unlock(&tb.lock)
	if !ok {
		badTimer()
	}
}

// resettimer resets an existing inactive timer to turn it into an active timer,
// with a new time for when the timer should fire.
// This should be called instead of addtimer if the timer value has been,
// or may have been, used previously.
func resettimer(t *timer, when int64) {
	if oldTimers {
		resettimerOld(t, when)
		return
	}

	if when < 0 {
		when = maxWhen
	}

	for {
		switch s := atomic.Load(&t.status); s {
		case timerNoStatus, timerRemoved:
			atomic.Store(&t.status, timerWaiting)
			t.when = when
			addInitializedTimer(t)
			return
		case timerDeleted:
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

func resettimerOld(t *timer, when int64) {
	t.when = when
	addtimer(t)
}

// Timerproc runs the time-driven events.
// It sleeps until the next event in the tb heap.
// If addtimer inserts a new earlier event, it wakes timerproc early.
func timerproc(tb *timersBucket) {
	tb.gp = getg()
	for {
		lock(&tb.lock)
		tb.sleeping = false
		now := nanotime()
		delta := int64(-1)
		for {
			if len(tb.t) == 0 {
				delta = -1
				break
			}
			t := tb.t[0]
			delta = t.when - now
			if delta > 0 {
				break
			}
			ok := true
			if t.period > 0 {
				// leave in heap but adjust next time to fire
				t.when += t.period * (1 + -delta/t.period)
				if !siftdownTimer(tb.t, 0) {
					ok = false
				}
			} else {
				// remove from heap
				last := len(tb.t) - 1
				if last > 0 {
					tb.t[0] = tb.t[last]
					tb.t[0].i = 0
				}
				tb.t[last] = nil
				tb.t = tb.t[:last]
				if last > 0 {
					if !siftdownTimer(tb.t, 0) {
						ok = false
					}
				}
				t.i = -1 // mark as removed
			}
			f := t.f
			arg := t.arg
			seq := t.seq
			unlock(&tb.lock)
			if !ok {
				badTimer()
			}
			if raceenabled {
				raceacquire(unsafe.Pointer(t))
			}
			f(arg, seq)
			lock(&tb.lock)
		}
		if delta < 0 || faketime > 0 {
			// No timers left - put goroutine to sleep.
			tb.rescheduling = true
			goparkunlock(&tb.lock, waitReasonTimerGoroutineIdle, traceEvGoBlock, 1)
			continue
		}
		// At least one timer pending. Sleep until then.
		tb.sleeping = true
		tb.sleepUntil = now + delta
		noteclear(&tb.waitnote)
		unlock(&tb.lock)
		notetsleepg(&tb.waitnote, delta)
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
// This is currently called when the world is stopped, but it could
// work as long as the timers for pp are locked.
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
		return
	}
	var moved []*timer
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
				if !atomic.Cas(&t.status, timerMoving, timerWaiting) {
					badTimer()
				}
				if s == timerModifiedEarlier {
					if n := atomic.Xadd(&pp.adjustTimers, -1); int32(n) <= 0 {
						addAdjustedTimers(pp, moved)
						return
					}
				}
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
}

// addAdjustedTimers adds any timers we adjusted in adjusttimers
// back to the timer heap.
func addAdjustedTimers(pp *p, moved []*timer) {
	for _, t := range moved {
	loop:
		for {
			switch s := atomic.Load(&t.status); s {
			case timerWaiting:
				// This is the normal case.
				if !doaddtimer(pp, t) {
					badTimer()
				}
				break loop
			case timerDeleted:
				// Timer has been deleted since we adjusted it.
				// This timer is already out of the heap.
				if !atomic.Cas(&t.status, s, timerRemoved) {
					badTimer()
				}
				break loop
			case timerModifiedEarlier, timerModifiedLater:
				// Timer has been modified again since
				// we adjusted it.
				if atomic.Cas(&t.status, s, timerMoving) {
					t.when = t.nextwhen
					if !doaddtimer(pp, t) {
						badTimer()
					}
					if !atomic.Cas(&t.status, timerMoving, timerWaiting) {
						badTimer()
					}
					if s == timerModifiedEarlier {
						atomic.Xadd(&pp.adjustTimers, -1)
					}
				}
				break loop
			case timerNoStatus, timerRunning, timerRemoving, timerRemoved, timerMoving:
				badTimer()
			case timerModifying:
				// Wait and try again.
				osyield()
				continue
			}
		}
	}
}

// runtimer examines the first timer in timers. If it is ready based on now,
// it runs the timer and removes or updates it.
// Returns 0 if it ran a timer, -1 if there are no more timers, or the time
// when the first timer should run.
// The caller must have locked the timers for pp.
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
func runOneTimer(pp *p, t *timer, now int64) {
	if raceenabled {
		if pp.timerRaceCtx == 0 {
			pp.timerRaceCtx = racegostart(funcPC(runtimer) + sys.PCQuantum)
		}
		raceacquirectx(pp.timerRaceCtx, unsafe.Pointer(t))
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
		// Temporarily use the P's racectx for g0.
		gp := getg()
		if gp.racectx != 0 {
			throw("runOneTimer: unexpected racectx")
		}
		gp.racectx = pp.timerRaceCtx
	}

	// Note that since timers are locked here, f may not call
	// addtimer or resettimer.

	f(arg, seq)

	if raceenabled {
		gp := getg()
		gp.racectx = 0
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

func timejumpOld() *g {
	if faketime == 0 {
		return nil
	}

	for i := range timers {
		lock(&timers[i].lock)
	}
	gp := timejumpLocked()
	for i := range timers {
		unlock(&timers[i].lock)
	}

	return gp
}

func timejumpLocked() *g {
	// Determine a timer bucket with minimum when.
	var minT *timer
	for i := range timers {
		tb := &timers[i]
		if !tb.created || len(tb.t) == 0 {
			continue
		}
		t := tb.t[0]
		if minT == nil || t.when < minT.when {
			minT = t
		}
	}
	if minT == nil || minT.when <= faketime {
		return nil
	}

	faketime = minT.when
	tb := minT.tb
	if !tb.rescheduling {
		return nil
	}
	tb.rescheduling = false
	return tb.gp
}

func timeSleepUntil() int64 {
	if oldTimers {
		return timeSleepUntilOld()
	}

	next := int64(maxWhen)

	for _, pp := range allp {
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

	return next
}

func timeSleepUntilOld() int64 {
	next := int64(1<<63 - 1)

	// Determine minimum sleepUntil across all the timer buckets.
	//
	// The function can not return a precise answer,
	// as another timer may pop in as soon as timers have been unlocked.
	// So lock the timers one by one instead of all at once.
	for i := range timers {
		tb := &timers[i]

		lock(&tb.lock)
		if tb.sleeping && tb.sleepUntil < next {
			next = tb.sleepUntil
		}
		unlock(&tb.lock)
	}

	return next
}

// Heap maintenance algorithms.
// These algorithms check for slice index errors manually.
// Slice index error can happen if the program is using racy
// access to timers. We don't want to panic here, because
// it will cause the program to crash with a mysterious
// "panic holding locks" message. Instead, we panic while not
// holding a lock.
// The races can occur despite the bucket locks because assignBucket
// itself is called without locks, so racy calls can cause a timer to
// change buckets while executing these functions.

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
		t[i].i = i
		i = p
	}
	if tmp != t[i] {
		t[i] = tmp
		t[i].i = i
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
		t[i].i = i
		i = c
	}
	if tmp != t[i] {
		t[i] = tmp
		t[i].i = i
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

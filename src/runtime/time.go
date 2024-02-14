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
	status atomic.Uint32
}

// Code outside this file has to be careful in using a timer value.
//
// The pp, status, and nextwhen fields may only be used by code in this file.
//
// Code that creates a new timer value can set the when, period, f,
// arg, and seq fields before the first call to modtimer.
// After that, period, f, arg, and seq are immutable.
// They may be read but not modified.
//
// An active timer (one that has been passed to modtimer) may be
// passed to deltimer (time.stopTimer), after which it is no longer an
// active timer. It is an inactive timer.
// In an inactive timer the period, f, arg, and seq fields may be modified,
// but not the when field.
// It's OK to just drop an inactive timer and let the GC collect it.
//
// An active timer may be passed to modtimer. No fields may be touched.
// It remains an active timer.
//
// An inactive timer may be passed to resettimer to turn into an
// active timer with an updated when field.
// It's OK to pass a newly allocated timer value to resettimer.
//
// Timer operations are deltimer, modtimer, adjusttimers, and runtimer.
//
// We don't permit calling deltimer/modtimer simultaneously,
// but adjusttimers and runtimer can be called at the same time as any of those.
//
// Active timers live in heaps attached to P, in the timers field.
// Inactive timers live there too temporarily, until they are removed.
//
// deltimer:
//   timerWaiting         -> timerLocked -> timerModified
//   timerModified        -> timerLocked -> timerModified
//   timerRemoved         -> do nothing
//   timerLocked       -> wait until status changes
// modtimer:
//   timerWaiting    -> timerLocked -> timerModified
//   timerModified   -> timerLocked -> timerModified
//   timerRemoved    -> timerLocked -> timerWaiting
//   timerLocked  -> wait until status changes
// adjusttimers (looks in P's timer heap):
//   timerModified   -> timerLocked -> timerWaiting/timerRemoved
// runtimer (looks in P's timer heap):
//   timerRemoved   -> panic: uninitialized timer
//   timerWaiting    -> timerWaiting or
//   timerWaiting    -> timerLocked -> timerWaiting/timerRemoved
//   timerLocked  -> wait until status changes
//   timerModified   -> timerLocked -> timerWaiting/timerRemoved

// Values for the timer status field.
const (
	// Timer has no status set yet or is removed from the heap.
	// Must be zero value; see issue 21874.
	timerRemoved = iota

	// Waiting for timer to fire.
	// The timer is in some P's heap.
	timerWaiting

	// The timer is locked for exclusive use.
	// The timer will only have this status briefly.
	timerLocked

	// The timer has been modified to a different time.
	// The new when value is in the nextwhen field.
	// The timer is in some P's heap, possibly in the wrong place
	// (the right place by .when; the wrong place by .nextwhen).
	timerModified
)

// lock locks the timer, allowing reading or writing any of the timer fields.
// It returns the current m and the status prior to the lock.
// The caller must call unlock with the same m and an updated status.
func (t *timer) lock() (status uint32, mp *m) {
	acquireLockRank(lockRankTimer)
	for {
		status := t.status.Load()
		if status == timerLocked {
			osyield()
			continue
		}
		// Prevent preemption while the timer is locked.
		// This could lead to a self-deadlock. See #38070.
		mp := acquirem()
		if t.status.CompareAndSwap(status, timerLocked) {
			return status, mp
		}
		releasem(mp)
	}
}

// unlock unlocks the timer.
// If mp == nil, the caller is responsible for calling
// releasem(mp) with the mp returned by t.lock.
func (t *timer) unlock(status uint32, mp *m) {
	releaseLockRank(lockRankTimer)
	if t.status.Load() != timerLocked {
		badTimer()
	}
	if status == timerLocked {
		badTimer()
	}
	t.status.Store(status)
	if mp != nil {
		releasem(mp)
	}
}

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
	gopark(resetForSleep, unsafe.Pointer(t), waitReasonSleep, traceBlockSleep, 1)
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
	if t.status.Load() != 0 {
		throw("startTimer called with initialized timer")
	}
	resettimer(t, t.when)
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
func modTimer(t *timer, when, period int64) {
	modtimer(t, when, period, t.f, t.arg, t.seq)
}

// Go runtime.

// Ready the goroutine arg.
func goroutineReady(arg any, seq uintptr) {
	goready(arg.(*g), 0)
}

// doaddtimer adds t to the current P's heap.
// The caller must have set t.pp = pp, unlocked t,
// and then locked the timers for pp.
func doaddtimer(pp *p, t *timer) {
	// Timers rely on the network poller, so make sure the poller
	// has started.
	if netpollInited.Load() == 0 {
		netpollGenericInit()
	}

	if t.pp.ptr() != pp {
		throw("doaddtimer: P not set in timer")
	}
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
	status, mp := t.lock()
	if status == timerWaiting || (status == timerModified && t.nextwhen != 0) {
		// Timer pending: stop it.
		t.pp.ptr().deletedTimers.Add(1)
		t.nextwhen = 0
		t.unlock(timerModified, mp)
		return true
	}

	// Timer already run or deleted.
	t.unlock(status, mp)
	return false
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

	status, mp := t.lock()
	t.period = period
	t.f = f
	t.arg = arg
	t.seq = seq

	if status == timerRemoved {
		// Set up t for insertion but unlock first,
		// to avoid lock inversion with timers lock.
		// Since t is not in a heap yet, nothing will
		// find and modify it until after the doaddtimer.
		t.when = when
		pp := getg().m.p.ptr()
		t.pp.set(pp)
		// pass mp=nil to t.unlock to avoid preemption
		// between t.unlock and lock of timersLock.
		// releasem done manually below
		t.unlock(timerWaiting, nil)

		lock(&pp.timersLock)
		doaddtimer(pp, t)
		unlock(&pp.timersLock)
		releasem(mp)
		wakeNetPoller(when)
		return false
	}

	pending := status == timerWaiting || status == timerModified && t.nextwhen != 0
	if !pending {
		t.pp.ptr().deletedTimers.Add(-1)
	}

	// The timer is in some other P's heap, so we can't change
	// the when field. If we did, the other P's heap would
	// be out of order. So we put the new when value in the
	// nextwhen field, and let the other P set the when field
	// when it is prepared to resort the heap.
	t.nextwhen = when
	earlier := when < t.when
	if earlier {
		updateTimerModifiedEarliest(t.pp.ptr(), when)
	}

	t.unlock(timerModified, mp)

	// If the new status is earlier, wake up the poller.
	if earlier {
		wakeNetPoller(when)
	}

	return pending
}

// resettimer resets the time when a timer should fire.
// If used for an inactive timer, the timer will become active.
// Reports whether the timer was active and was stopped.
func resettimer(t *timer, when int64) bool {
	return modtimer(t, when, t.period, t.f, t.arg, t.seq)
}

// cleantimers cleans up the head of the timer queue. This speeds up
// programs that create and delete timers; leaving them in the heap
// slows down heap operations. Reports whether no timer problems were found.
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

		status := t.status.Load()
		if status != timerModified {
			// Fast path: head of timers does not need adjustment.
			return
		}

		status, mp := t.lock()
		if status != timerModified {
			// Head of timers does not need adjustment.
			t.unlock(status, mp)
			return
		}
		dodeltimer0(pp)
		if t.nextwhen == 0 {
			pp.deletedTimers.Add(-1)
			status = timerRemoved
			t.unlock(status, mp)
		} else {
			// Now we can change the when field.
			t.when = t.nextwhen
			t.pp.set(pp)
			status = timerWaiting
			t.unlock(status, mp)
			// Move t to the right position.
			doaddtimer(pp, t)
		}
	}
}

// adoptTimers adopts any timers from pp into the local P,
// because pp is being destroyed.
func adoptTimers(pp *p) {
	if len(pp.timers) > 0 {
		plocal := getg().m.p.ptr()
		// The world is stopped, but we acquire timersLock to
		// protect against sysmon calling timeSleepUntil.
		// This is the only case where we hold the timersLock of
		// more than one P, so there are no deadlock concerns.
		lock(&plocal.timersLock)
		lock(&pp.timersLock)
		moveTimers(plocal, pp.timers)
		pp.timers = nil
		pp.numTimers.Store(0)
		pp.deletedTimers.Store(0)
		pp.timer0When.Store(0)
		unlock(&pp.timersLock)
		unlock(&plocal.timersLock)
	}
}

// moveTimers moves a slice of timers to pp. The slice has been taken
// from a different P.
// This is currently called when the world is stopped, but the caller
// is expected to have locked the timers for pp.
func moveTimers(pp *p, timers []*timer) {
	for _, t := range timers {
		status, mp := t.lock()
		switch status {
		case timerWaiting:
			t.pp.set(pp)
			// Unlock before add, to avoid append (allocation)
			// while holding lock. This would be correct even if the world wasn't
			// stopped (but it is), and it makes staticlockranking happy.
			t.unlock(status, mp)
			doaddtimer(pp, t)
			continue
		case timerModified:
			t.pp = 0
			if t.nextwhen != 0 {
				t.when = t.nextwhen
				status = timerWaiting
				t.pp.set(pp)
				t.unlock(status, mp)
				doaddtimer(pp, t)
				continue
			} else {
				status = timerRemoved
			}
		case timerRemoved:
			badTimer()
		}
		t.unlock(status, mp)
	}
}

// adjusttimers looks through the timers in the current P's heap for
// any timers that have been modified to run earlier, and puts them in
// the correct place in the heap. While looking for those timers,
// it also moves timers that have been modified to run later,
// and removes deleted timers. The caller must have locked the timers for pp.
func adjusttimers(pp *p, now int64, force bool) {
	// If we haven't yet reached the time of the earliest timerModified
	// timer, don't do anything. This speeds up programs that adjust
	// a lot of timers back and forth if the timers rarely expire.
	// We'll postpone looking through all the adjusted timers until
	// one would actually expire.
	if !force {
		first := pp.timerModifiedEarliest.Load()
		if first == 0 || first > now {
			if verifyTimers {
				verifyTimerHeap(pp)
			}
			return
		}
	}

	// We are going to clear all timerModified timers.
	pp.timerModifiedEarliest.Store(0)

	changed := false
	for i := 0; i < len(pp.timers); i++ {
		t := pp.timers[i]
		if t.pp.ptr() != pp {
			throw("adjusttimers: bad p")
		}

		status, mp := t.lock()
		if status == timerRemoved {
			badTimer()
		}
		if status == timerModified {
			if t.nextwhen == 0 {
				n := len(pp.timers)
				pp.timers[i] = pp.timers[n-1]
				pp.timers[n-1] = nil
				pp.timers = pp.timers[:n-1]
				t.pp = 0
				status = timerRemoved
				pp.deletedTimers.Add(-1)
				i--
				changed = true
			} else {
				// Now we can change the when field.
				t.when = t.nextwhen
				changed = true
				status = timerWaiting
			}
		}
		t.unlock(status, mp)
	}

	if changed {
		initTimerHeap(pp.timers)
		updateTimer0When(pp)
	}

	if verifyTimers {
		verifyTimerHeap(pp)
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

// checkTimers runs any timers for the P that are ready.
// If now is not 0 it is the current time.
// It returns the passed time or the current time if now was passed as 0.
// and the time when the next timer should run or 0 if there is no next timer,
// and reports whether it ran any timers.
// If the time when the next timer should run is not 0,
// it is always larger than the returned time.
// We pass now in and out to avoid extra calls of nanotime.
//
//go:yeswritebarrierrec
func checkTimers(pp *p, now int64) (rnow, pollUntil int64, ran bool) {
	// If it's not yet time for the first timer, or the first adjusted
	// timer, then there is nothing to do.
	next := pp.timer0When.Load()
	nextAdj := pp.timerModifiedEarliest.Load()
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
		if pp != getg().m.p.ptr() || int(pp.deletedTimers.Load()) <= int(pp.numTimers.Load()/4) {
			return now, next, false
		}
	}

	lock(&pp.timersLock)

	if len(pp.timers) > 0 {
		// If this is the local P, and there are a lot of deleted timers,
		// clear them out. We only do this for the local P to reduce
		// lock contention on timersLock.
		force := pp == getg().m.p.ptr() && int(pp.deletedTimers.Load()) > len(pp.timers)/4
		adjusttimers(pp, now, force)
		for len(pp.timers) > 0 {
			// Note that runtimer may temporarily unlock
			// pp.timersLock.
			if tw := runtimer(pp, now); tw != 0 {
				if tw > 0 {
					pollUntil = tw
				}
				break
			}
			ran = true
		}
	}

	unlock(&pp.timersLock)

	return now, pollUntil, ran
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
Redo:
	if len(pp.timers) == 0 {
		return -1
	}
	t := pp.timers[0]
	if t.pp.ptr() != pp {
		throw("runtimer: bad p")
	}

	if t.status.Load() == timerWaiting && t.when > now {
		// Fast path: not ready to run.
		// The access of t.when is protected by the caller holding
		// pp.timersLock, even though t itself is unlocked.
		return t.when
	}

	status, mp := t.lock()
	if status == timerModified {
		dodeltimer0(pp)
		if t.nextwhen == 0 {
			status = timerRemoved
			pp.deletedTimers.Add(-1)
			t.unlock(status, mp)
		} else {
			t.when = t.nextwhen
			t.pp.set(pp)
			status = timerWaiting
			t.unlock(status, mp)
			doaddtimer(pp, t)
		}
		goto Redo
	}

	if status != timerWaiting {
		badTimer()
	}

	if t.when > now {
		// Not ready to run.
		t.unlock(status, mp)
		return t.when
	}

	unlockAndRunTimer(pp, t, now, status, mp)
	return 0
}

// unlockAndRunTimer unlocks and runs a single timer.
// The caller must have locked the timers for pp.
// This will temporarily unlock the timers while running the timer function.
//
//go:systemstack
func unlockAndRunTimer(pp *p, t *timer, now int64, status uint32, mp *m) {
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
		status = timerWaiting
		updateTimer0When(pp)
	} else {
		// Remove from heap.
		dodeltimer0(pp)
		status = timerRemoved
	}
	t.unlock(status, mp)

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
func updateTimerPMask(pp *p) {
	if pp.numTimers.Load() > 0 {
		return
	}

	// Looks like there are no timers, however another P may transiently
	// decrement numTimers when handling a timerModified timer in
	// checkTimers. We must take timersLock to serialize with these changes.
	lock(&pp.timersLock)
	if pp.numTimers.Load() == 0 {
		timerpMask.clear(pp.id)
	}
	unlock(&pp.timersLock)
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
		if old != 0 && old < nextwhen {
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

// initTimerHeap reestablishes the heap order in the slice t.
// It takes O(n) time for n=len(t), not the O(n log n) of n repeated add operations.
func initTimerHeap(t []*timer) {
	// Last possible element that needs sifting down is parent of last element;
	// last element is len(t)-1; parent of last element is (len(t)-1-1)/4.
	if len(t) <= 1 {
		return
	}
	for i := (len(t) - 1 - 1) / 4; i >= 0; i-- {
		siftdownTimer(t, i)
	}
}

// badTimer is called if the timer data structures have been corrupted,
// presumably due to racy use by the program. We panic here rather than
// panicking due to invalid slice access while holding locks.
// See issue #25686.
func badTimer() {
	throw("timer data corruption")
}

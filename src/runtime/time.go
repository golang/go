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

// A timer is a potentially repeating trigger for calling t.f(t.arg, t.seq).
// Timers are allocated by client code, often as part of other data structures.
// Each P has a heap of pointers to timers that it manages.
//
// A timer is expected to be used by only one client goroutine at a time,
// but there will be concurrent access by the P managing that timer.
// The fundamental state about the timer is managed in the atomic state field,
// including a lock bit to manage access to the other fields.
// The lock bit supports a manual cas-based spin lock that handles
// contention by yielding the OS thread. The expectation is that critical
// sections are very short and contention on the lock bit is low.
//
// Package time knows the layout of this structure.
// If this struct changes, adjust ../time/sleep.go:/runtimeTimer.
type timer struct {
	ts *timers

	// Timer wakes up at when, and then at when+period, ... (period > 0 only)
	// each time calling f(arg, now) in the timer goroutine, so f must be
	// a well-behaved function and not block.
	//
	// when must be positive on an active timer.
	// Timers in heaps are ordered by when.
	when   int64
	period int64
	f      func(any, uintptr)
	arg    any
	seq    uintptr

	// nextWhen is the next value for when,
	// set if state&timerNextWhen is true.
	// In that case, the actual update of when = nextWhen
	// must be delayed until the heap can be fixed at the same time.
	nextWhen int64

	// The state field holds state bits, defined below.
	state atomic.Uint32
}

// A timers is a per-P set of timers.
type timers struct {
	// mu protects timers; timers are per-P, but the scheduler can
	// access the timers of another P, so we have to lock.
	mu mutex

	// heap is the set of timers, ordered by t.when.
	// Must hold lock to access.
	heap []*timer

	// len is an atomic copy of len(heap).
	len atomic.Uint32

	// zombies is the number of timers in the heap
	// that are marked for removal.
	zombies atomic.Int32

	// raceCtx is the race context used while executing timer functions.
	raceCtx uintptr

	// minWhen is the minimum heap[i].when value (= heap[0].when).
	// The wakeTime method uses minWhen and minNextWhen to determine
	// the next wake time.
	// If minWhen = 0, it means there are no timers in the heap.
	minWhen atomic.Int64

	// minNextWhen is a lower bound on the minimum
	// heap[i].nextWhen over timers with the timerNextWhen bit set.
	// If minNextWhen = 0, it means there are no timerNextWhen timers in the heap.
	minNextWhen atomic.Int64
}

func (ts *timers) lock() {
	lock(&ts.mu)
}

func (ts *timers) unlock() {
	// Update atomic copy of len(ts.heap).
	// We only update at unlock so that the len is always
	// the most recent unlocked length, not an ephemeral length.
	// This matters if we lock ts, delete the only timer from the heap,
	// add it back, and unlock. We want ts.len.Load to return 1 the
	// entire time, never 0. This is important for pidleput deciding
	// whether ts is empty.
	ts.len.Store(uint32(len(ts.heap)))

	unlock(&ts.mu)
}

// Timer state field.
// Note that state 0 must be "unlocked, not in heap" and usable,
// at least for time.Timer.Stop. See go.dev/issue/21874.
const (
	// timerLocked is set when the timer is locked,
	// meaning other goroutines cannot read or write mutable fields.
	// Goroutines can still read the state word atomically to see
	// what the state was before it was locked.
	// The lock is implemented as a cas on the state field with osyield on contention;
	// the expectation is very short critical sections with little to no contention.
	timerLocked = 1 << iota

	// timerHeaped is set when the timer is stored in some P's heap.
	timerHeaped

	// timerNextWhen is set when a pending change to the timer's when
	// field has been stored in t.nextwhen. The change to t.when waits
	// until the heap in which the timer appears can also be updated.
	// Only set when timerHeaped is also set.
	timerNextWhen

	// timerZombie is set when the timer has been stopped
	// but is still present in some P's heap.
	// Only set when timerHeaped is also set.
	// It is possible for timerNextWhen and timerZombie to both
	// be set, meaning that the timer was modified and then stopped.
	timerZombie
)

// lock locks the timer, allowing reading or writing any of the timer fields.
// It returns the current m and the status prior to the lock.
// The caller must call unlock with the same m and an updated status.
func (t *timer) lock() (state uint32, mp *m) {
	acquireLockRank(lockRankTimer)
	for {
		state := t.state.Load()
		if state&timerLocked != 0 {
			osyield()
			continue
		}
		// Prevent preemption while the timer is locked.
		// This could lead to a self-deadlock. See #38070.
		mp := acquirem()
		if t.state.CompareAndSwap(state, state|timerLocked) {
			return state, mp
		}
		releasem(mp)
	}
}

// unlock unlocks the timer.
// If mp == nil, the caller is responsible for calling
// releasem(mp) with the mp returned by t.lock.
func (t *timer) unlock(state uint32, mp *m) {
	releaseLockRank(lockRankTimer)
	if t.state.Load()&timerLocked == 0 {
		badTimer()
	}
	if state&timerLocked != 0 {
		badTimer()
	}
	t.state.Store(state)
	if mp != nil {
		releasem(mp)
	}
}

// updateHeap updates t.when as directed by state, returning the new state
// and a bool indicating whether the state (and t.when) changed.
// If ts != nil, then ts must be locked, t must be ts.heap[0], and updateHeap
// takes care of moving t within the timers heap to preserve the heap invariants.
// If ts == nil, then t must not be in a heap (or is in a heap that is
// temporarily not maintaining its invariant, such as during timers.adjust).
func (t *timer) updateHeap(state uint32, ts *timers) (newState uint32, updated bool) {
	if ts != nil {
		assertLockHeld(&ts.mu)
	}
	if state&timerZombie != 0 {
		// Take timer out of heap, applying final t.when update first.
		state &^= timerHeaped | timerZombie
		if state&timerNextWhen != 0 {
			state &^= timerNextWhen
			t.when = t.nextWhen
		}
		if ts != nil {
			if t != ts.heap[0] {
				badTimer()
			}
			ts.zombies.Add(-1)
			ts.deleteMin()
		}
		return state, true
	}

	if state&timerNextWhen != 0 {
		// Apply t.when update and move within heap.
		state &^= timerNextWhen
		t.when = t.nextWhen
		// Move t to the right position.
		if ts != nil {
			if t != ts.heap[0] {
				badTimer()
			}
			ts.siftDown(0)
			ts.updateMinWhen()
		}
		return state, true
	}

	return state, false
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
	t.nextWhen = nanotime() + ns
	if t.nextWhen < 0 { // check for overflow.
		t.nextWhen = maxWhen
	}
	gopark(resetForSleep, unsafe.Pointer(t), waitReasonSleep, traceBlockSleep, 1)
}

// resetForSleep is called after the goroutine is parked for timeSleep.
// We can't call resettimer in timeSleep itself because if this is a short
// sleep and there are many goroutines then the P can wind up running the
// timer function, goroutineReady, before the goroutine has been parked.
func resetForSleep(gp *g, ut unsafe.Pointer) bool {
	t := (*timer)(ut)
	t.reset(t.nextWhen)
	return true
}

// startTimer adds t to the timer heap.
//
//go:linkname startTimer time.startTimer
func startTimer(t *timer) {
	if raceenabled {
		racerelease(unsafe.Pointer(t))
	}
	if t.state.Load() != 0 {
		throw("startTimer called with initialized timer")
	}
	t.reset(t.when)
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
func modTimer(t *timer, when, period int64) {
	t.modify(when, period, t.f, t.arg, t.seq)
}

// Go runtime.

// Ready the goroutine arg.
func goroutineReady(arg any, seq uintptr) {
	goready(arg.(*g), 0)
}

// addHeap adds t to the timers heap.
// The caller must hold ts.lock.
// The caller must also have checked that t belongs in the heap.
// Callers that are not sure can call t.maybeAdd instead,
// but note that maybeAdd has different locking requirements.
func (ts *timers) addHeap(t *timer) {
	assertLockHeld(&ts.mu)
	// Timers rely on the network poller, so make sure the poller
	// has started.
	if netpollInited.Load() == 0 {
		netpollGenericInit()
	}

	if t.ts != nil {
		throw("ts set in timer")
	}
	t.ts = ts
	ts.heap = append(ts.heap, t)
	ts.siftUp(len(ts.heap) - 1)
	if t == ts.heap[0] {
		ts.updateMinWhen()
	}
}

// stop deletes the timer t. It may be on some other P, so we can't
// actually remove it from the timers heap. We can only mark it as stopped.
// It will be removed in due course by the P whose heap it is on.
// Reports whether the timer was stopped before it was run.
func (t *timer) stop() bool {
	state, mp := t.lock()
	pending := false
	if state&timerHeaped != 0 {
		// Timer is in some heap, but is possibly already stopped
		// (indicated by a nextWhen update to 0).
		if state&timerNextWhen == 0 || t.nextWhen > 0 {
			// Timer pending: stop it.
			t.nextWhen = 0
			state |= timerNextWhen
			pending = true
		}
		// Mark timer for removal unless already marked.
		if state&timerZombie == 0 {
			state |= timerZombie
			t.ts.zombies.Add(1)
		}
	}

	t.unlock(state, mp)
	return pending
}

// deleteMin removes timer 0 from ts.
// ts must be locked.
func (ts *timers) deleteMin() {
	assertLockHeld(&ts.mu)
	t := ts.heap[0]
	if t.ts != ts {
		throw("wrong timers")
	}
	t.ts = nil
	last := len(ts.heap) - 1
	if last > 0 {
		ts.heap[0] = ts.heap[last]
	}
	ts.heap[last] = nil
	ts.heap = ts.heap[:last]
	if last > 0 {
		ts.siftDown(0)
	}
	ts.updateMinWhen()
	if last == 0 {
		// If there are no timers, then clearly there are no timerNextWhen timers.
		ts.minNextWhen.Store(0)
	}
}

// modify modifies an existing timer.
// This is called by the netpoll code or time.Ticker.Reset or time.Timer.Reset.
// Reports whether the timer was modified before it was run.
func (t *timer) modify(when, period int64, f func(any, uintptr), arg any, seq uintptr) bool {
	if when <= 0 {
		throw("timer when must be positive")
	}
	if period < 0 {
		throw("timer period must be non-negative")
	}

	state, mp := t.lock()
	t.period = period
	t.f = f
	t.arg = arg
	t.seq = seq

	if state&timerHeaped == 0 {
		// Set up t for insertion but unlock first,
		// to avoid lock inversion with timers lock.
		// Since t is not in a heap yet, nothing will
		// find and modify it until after the ts.add.
		t.when = when
		t.unlock(state, mp)
		t.maybeAdd()
		return false
	}

	pending := true // in the heap

	if state&timerZombie != 0 {
		// In the heap but marked for removal (by a Stop); therefore not pending.
		// Unmark it, since it has been Reset and will be running again.
		pending = false
		t.ts.zombies.Add(-1)
		state &^= timerZombie
	}

	// The timer is in some P's heap (perhaps another P),
	// so we can't change the when field.
	// If we did, the other P's heap would be out of order.
	// So we put the new when value in the nextWhen field
	// and set timerNextWhen, leaving the other P set the when
	// field when it is prepared to maintain the heap invariant.
	t.nextWhen = when
	state |= timerNextWhen
	earlier := when < t.when
	if earlier {
		t.ts.updateMinNextWhen(when)
	}
	t.unlock(state, mp)

	// If the new status is earlier, wake up the poller.
	if earlier {
		wakeNetPoller(when)
	}

	return pending
}

// needsAdd reports whether t needs to be added to a timers heap.
func (t *timer) needsAdd(state uint32) bool {
	return state&timerHeaped == 0 &&
		t.when > 0
}

// maybeAdd adds t to the local timers heap if it needs to be in a heap.
// The caller must not hold t's lock nor any timers heap lock.
// The caller probably just unlocked t, but that lock must be dropped
// in order to acquire a ts.lock, to avoid lock inversions.
// (timers.adjust holds ts.lock while acquiring each t's lock,
// so we cannot hold any t's lock while acquiring ts.lock).
//
// Strictly speaking it *might* be okay to hold t.lock and
// acquire ts.lock at the same time, because we know that
// t is not in any ts.heap, so nothing holding a ts.lock would
// be acquiring the t.lock at the same time, meaning there
// isn't a possible deadlock. But it is easier and safer not to be
// too clever and respect the static ordering.
// (If we don't, we have to change the static lock checking of t and ts.)
//
// Because we are not holding t while acquiring ts,
// concurrent calls to enqueueTimerChan may result in
// concurrent calls to t.maybeAdd, so we cannot assume that
// t is not in a heap on entry to t.maybeAdd.
func (t *timer) maybeAdd() {
	ts := &getg().m.p.ptr().timers
	ts.lock()
	ts.cleanHead()
	state, mp := t.lock()
	when := int64(0)
	if t.needsAdd(state) {
		state |= timerHeaped
		ts.addHeap(t)
		when = t.when
	}
	t.unlock(state, mp)
	ts.unlock()
	if when > 0 {
		wakeNetPoller(when)
	}
}

// reset resets the time when a timer should fire.
// If used for an inactive timer, the timer will become active.
// Reports whether the timer was active and was stopped.
func (t *timer) reset(when int64) bool {
	return t.modify(when, t.period, t.f, t.arg, t.seq)
}

// cleanHead cleans up the head of the timer queue. This speeds up
// programs that create and delete timers; leaving them in the heap
// slows down heap operations.
// The caller must have locked ts.
func (ts *timers) cleanHead() {
	assertLockHeld(&ts.mu)
	gp := getg()
	for {
		if len(ts.heap) == 0 {
			return
		}

		// This loop can theoretically run for a while, and because
		// it is holding timersLock it cannot be preempted.
		// If someone is trying to preempt us, just return.
		// We can clean the timers later.
		if gp.preemptStop {
			return
		}

		t := ts.heap[0]
		if t.ts != ts {
			throw("bad ts")
		}

		if t.state.Load()&(timerNextWhen|timerZombie) == 0 {
			// Fast path: head of timers does not need adjustment.
			return
		}

		state, mp := t.lock()
		state, updated := t.updateHeap(state, ts)
		t.unlock(state, mp)
		if !updated {
			// Head of timers does not need adjustment.
			return
		}
	}
}

// take moves any timers from src into ts
// and then clears the timer state from src,
// because src is being destroyed.
// The caller must not have locked either timers.
// For now this is only called when the world is stopped.
func (ts *timers) take(src *timers) {
	assertWorldStopped()
	if len(src.heap) > 0 {
		// The world is stopped, but we acquire timersLock to
		// protect against sysmon calling timeSleepUntil.
		// This is the only case where we hold more than one ts.lock,
		// so there are no deadlock concerns.
		src.lock()
		ts.lock()
		ts.move(src.heap)
		src.heap = nil
		src.zombies.Store(0)
		src.minWhen.Store(0)
		ts.unlock()
		src.unlock()
	}
}

// moveTimers moves a slice of timers to pp. The slice has been taken
// from a different P.
// This is currently called when the world is stopped, but the caller
// is expected to have locked ts.
func (ts *timers) move(timers []*timer) {
	assertLockHeld(&ts.mu)
	for _, t := range timers {
		state, mp := t.lock()
		t.ts = nil
		state, _ = t.updateHeap(state, nil)
		if state&timerHeaped != 0 {
			ts.addHeap(t)
		}
		t.unlock(state, mp)
	}
}

// adjust looks through the timers in ts.heap for
// any timers that have been modified to run earlier, and puts them in
// the correct place in the heap. While looking for those timers,
// it also moves timers that have been modified to run later,
// and removes deleted timers. The caller must have locked ts.
func (ts *timers) adjust(now int64, force bool) {
	assertLockHeld(&ts.mu)
	// If we haven't yet reached the time of the earliest modified
	// timer, don't do anything. This speeds up programs that adjust
	// a lot of timers back and forth if the timers rarely expire.
	// We'll postpone looking through all the adjusted timers until
	// one would actually expire.
	if !force {
		first := ts.minNextWhen.Load()
		if first == 0 || first > now {
			if verifyTimers {
				ts.verify()
			}
			return
		}
	}

	// minNextWhen is a lower bound on the earliest t.nextWhen
	// among the timerNextWhen timers. We want to make it more precise:
	// we are going to scan the heap and clean out all the timerNextWhen bits,
	// at which point minNextWhen can be set to 0 (indicating none at all).
	//
	// Other P's can be calling ts.wakeTime concurrently, and we'd like to
	// keep ts.wakeTime returning an accurate value throughout this entire process.
	//
	// Setting minNextWhen = 0 *before* the scan could make wakeTime
	// return an incorrect value: if minNextWhen < nextWhen, then clearing
	// it to 0 will make wakeTime return nextWhen (too late) until the scan finishes.
	// To avoid that, we want to set minNextWhen to 0 *after* the scan.
	//
	// Setting minNextWhen = 0 *after* the scan could result in missing
	// concurrent timer modifications in other goroutines; those will lock
	// the specific timer, set the timerNextWhen bit, and set t.nextWhen.
	// To avoid that, we want to set minNextWhen to 0 *before* the scan.
	//
	// The way out of this dilemma is to preserve wakeTime a different way.
	// wakeTime is min(minWhen, minNextWhen), and minWhen is protected by
	// ts.lock, which we hold, so we can modify it however we like in service
	// of keeping wakeTime accurate.
	//
	// So we can:
	//
	//	1. Set minWhen = min(minWhen, minNextWhen)
	//	2. Set minNextWhen = 0
	//	   (Other goroutines may modify timers and update minNextWhen now.)
	//	3. Scan timers
	//	4. Set minWhen = heap[0].when
	//
	// That order preserves a correct value of wakeTime throughout the entire
	// operation:
	// Step 1 “locks in” an accurate wakeTime even with minNextWhen cleared.
	// Step 2 makes sure concurrent nextWhen updates are not lost during the scan.
	// Step 3 clears all the old nextWhen values, justifying minNextWhen = 0.
	// Step 4 corrects minWhen to a precise value.
	//
	// The wakeTime method implementation reads minNextWhen *before* minWhen,
	// so that if the minNextWhen observes a 0, that means the minWhen
	// will include the information about what was zeroed.

	ts.minWhen.Store(ts.wakeTime())
	ts.minNextWhen.Store(0)

	changed := false
	for i := 0; i < len(ts.heap); i++ {
		t := ts.heap[i]
		if t.ts != ts {
			throw("bad ts")
		}

		state, mp := t.lock()
		if state&timerHeaped == 0 {
			badTimer()
		}
		if state&timerZombie != 0 {
			ts.zombies.Add(-1) // updateHeap will return updated=true and we will delete t
		}
		state, updated := t.updateHeap(state, nil)
		if updated {
			changed = true
			if state&timerHeaped == 0 {
				n := len(ts.heap)
				ts.heap[i] = ts.heap[n-1]
				ts.heap[n-1] = nil
				ts.heap = ts.heap[:n-1]
				t.ts = nil
				i--
			}
		}
		t.unlock(state, mp)
	}

	if changed {
		ts.initHeap()
	}
	ts.updateMinWhen()

	if verifyTimers {
		ts.verify()
	}
}

// wakeTime looks at ts's timers and returns the time when we
// should wake up the netpoller. It returns 0 if there are no timers.
// This function is invoked when dropping a P, so it must run without
// any write barriers.
//
//go:nowritebarrierrec
func (ts *timers) wakeTime() int64 {
	// Note that the order of these two loads matters:
	// adjust updates minWhen to make it safe to clear minNextWhen.
	// We read minWhen after reading minNextWhen so that
	// if we see a cleared minNextWhen, we are guaranteed to see
	// the updated minWhen.
	nextWhen := ts.minNextWhen.Load()
	when := ts.minWhen.Load()
	if when == 0 || (nextWhen != 0 && nextWhen < when) {
		when = nextWhen
	}
	return when
}

// check runs any timers in ts that are ready.
// If now is not 0 it is the current time.
// It returns the passed time or the current time if now was passed as 0.
// and the time when the next timer should run or 0 if there is no next timer,
// and reports whether it ran any timers.
// If the time when the next timer should run is not 0,
// it is always larger than the returned time.
// We pass now in and out to avoid extra calls of nanotime.
//
//go:yeswritebarrierrec
func (ts *timers) check(now int64) (rnow, pollUntil int64, ran bool) {
	// If it's not yet time for the first timer, or the first adjusted
	// timer, then there is nothing to do.
	next := ts.wakeTime()
	if next == 0 {
		// No timers to run or adjust.
		return now, 0, false
	}

	if now == 0 {
		now = nanotime()
	}

	// If this is the local P, and there are a lot of deleted timers,
	// clear them out. We only do this for the local P to reduce
	// lock contention on timersLock.
	zombies := ts.zombies.Load()
	if zombies < 0 {
		badTimer()
	}
	force := ts == &getg().m.p.ptr().timers && int(zombies) > int(ts.len.Load())/4

	if now < next && !force {
		// Next timer is not ready to run, and we don't need to clear deleted timers.
		return now, next, false
	}

	ts.lock()
	if len(ts.heap) > 0 {
		ts.adjust(now, force)
		for len(ts.heap) > 0 {
			// Note that runtimer may temporarily unlock ts.
			if tw := ts.run(now); tw != 0 {
				if tw > 0 {
					pollUntil = tw
				}
				break
			}
			ran = true
		}
	}
	ts.unlock()

	return now, pollUntil, ran
}

// run examines the first timer in ts. If it is ready based on now,
// it runs the timer and removes or updates it.
// Returns 0 if it ran a timer, -1 if there are no more timers, or the time
// when the first timer should run.
// The caller must have locked ts.
// If a timer is run, this will temporarily unlock ts.
//
//go:systemstack
func (ts *timers) run(now int64) int64 {
	assertLockHeld(&ts.mu)
Redo:
	if len(ts.heap) == 0 {
		return -1
	}
	t := ts.heap[0]
	if t.ts != ts {
		throw("bad ts")
	}

	if t.state.Load()&(timerNextWhen|timerZombie) == 0 && t.when > now {
		// Fast path: not ready to run.
		// The access of t.when is protected by the caller holding
		// ts.lock, even though t itself is unlocked.
		return t.when
	}

	state, mp := t.lock()
	state, updated := t.updateHeap(state, ts)
	if updated {
		t.unlock(state, mp)
		goto Redo
	}

	if state&timerHeaped == 0 {
		badTimer()
	}

	if t.when > now {
		// Not ready to run.
		t.unlock(state, mp)
		return t.when
	}

	ts.unlockAndRun(t, now, state, mp)
	assertLockHeld(&ts.mu) // t is unlocked now, but not ts
	return 0
}

// unlockAndRun unlocks and runs the timer t.
// If t is in a timer set (t.ts != nil), the caller must have locked the timer set,
// and this call will temporarily unlock the timer set while running the timer function.
//
//go:systemstack
func (ts *timers) unlockAndRun(t *timer, now int64, state uint32, mp *m) {
	assertLockHeld(&ts.mu)
	if raceenabled {
		tsLocal := &getg().m.p.ptr().timers
		if tsLocal.raceCtx == 0 {
			tsLocal.raceCtx = racegostart(abi.FuncPCABIInternal((*timers).run) + sys.PCQuantum)
		}
		raceacquirectx(tsLocal.raceCtx, unsafe.Pointer(t))
	}

	if state&(timerNextWhen|timerZombie) != 0 {
		badTimer()
	}

	f := t.f
	arg := t.arg
	seq := t.seq
	var next int64
	delay := now - t.when
	if t.period > 0 {
		// Leave in heap but adjust next time to fire.
		next = t.when + t.period*(1+delay/t.period)
		if next < 0 { // check for overflow.
			next = maxWhen
		}
	} else {
		next = 0
	}
	if state&timerHeaped != 0 {
		t.nextWhen = next
		state |= timerNextWhen
		if next == 0 {
			state |= timerZombie
			t.ts.zombies.Add(1)
		}
	} else {
		t.when = next
	}
	state, _ = t.updateHeap(state, ts)
	t.unlock(state, mp)

	if raceenabled {
		// Temporarily use the current P's racectx for g0.
		gp := getg()
		if gp.racectx != 0 {
			throw("unexpected racectx")
		}
		gp.racectx = gp.m.p.ptr().timers.raceCtx
	}

	ts.unlock()
	f(arg, seq)
	ts.lock()

	if raceenabled {
		gp := getg()
		gp.racectx = 0
	}
}

// verifyTimerHeap verifies that the timers is in a valid state.
// This is only for debugging, and is only called if verifyTimers is true.
// The caller must have locked ts.
func (ts *timers) verify() {
	assertLockHeld(&ts.mu)
	for i, t := range ts.heap {
		if i == 0 {
			// First timer has no parent.
			continue
		}

		// The heap is 4-ary. See siftupTimer and siftdownTimer.
		p := (i - 1) / 4
		if t.when < ts.heap[p].when {
			print("bad timer heap at ", i, ": ", p, ": ", ts.heap[p].when, ", ", i, ": ", t.when, "\n")
			throw("bad timer heap")
		}
	}
	if n := int(ts.len.Load()); len(ts.heap) != n {
		println("timer heap len", len(ts.heap), "!= atomic len", n)
		throw("bad timer heap len")
	}
}

// updateMinWhen sets ts.minWhen to ts.heap[0].when.
// The caller must have locked ts.
func (ts *timers) updateMinWhen() {
	assertLockHeld(&ts.mu)
	if len(ts.heap) == 0 {
		ts.minWhen.Store(0)
	} else {
		ts.minWhen.Store(ts.heap[0].when)
	}
}

// updateMinNextWhen updates ts.minNextWhen to be <= when.
// ts need not be (and usually is not) locked.
func (ts *timers) updateMinNextWhen(when int64) {
	for {
		old := ts.minNextWhen.Load()
		if old != 0 && old < when {
			return
		}
		if ts.minNextWhen.CompareAndSwap(old, when) {
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

		if w := pp.timers.wakeTime(); w != 0 {
			next = min(next, w)
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

// siftUp puts the timer at position i in the right place
// in the heap by moving it up toward the top of the heap.
func (ts *timers) siftUp(i int) {
	t := ts.heap
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
}

// siftDown puts the timer at position i in the right place
// in the heap by moving it down toward the bottom of the heap.
func (ts *timers) siftDown(i int) {
	t := ts.heap
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

// initHeap reestablishes the heap order in the slice ts.heap.
// It takes O(n) time for n=len(ts.heap), not the O(n log n) of n repeated add operations.
func (ts *timers) initHeap() {
	// Last possible element that needs sifting down is parent of last element;
	// last element is len(t)-1; parent of last element is (len(t)-1-1)/4.
	if len(ts.heap) <= 1 {
		return
	}
	for i := (len(ts.heap) - 1 - 1) / 4; i >= 0; i-- {
		ts.siftDown(i)
	}
}

// badTimer is called if the timer data structures have been corrupted,
// presumably due to racy use by the program. We panic here rather than
// panicking due to invalid slice access while holding locks.
// See issue #25686.
func badTimer() {
	throw("timer data corruption")
}

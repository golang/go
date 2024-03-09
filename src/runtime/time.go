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
// Timer accesses are protected by the lock t.mu, with a snapshot of
// t's state bits published in t.astate to enable certain fast paths to make
// decisions about a timer without acquiring the lock.
type timer struct {
	// mu protects reads and writes to all fields, with exceptions noted below.
	mu mutex

	astate atomic.Uint8 // atomic copy of state bits at last unlock; can be read without lock
	state  uint8        // state bits

	// Timer wakes up at when, and then at when+period, ... (period > 0 only)
	// each time calling f(arg, now) in the timer goroutine, so f must be
	// a well-behaved function and not block.
	when   int64
	period int64
	f      func(any, uintptr)
	arg    any
	seq    uintptr

	// If non-nil, the timers containing t.
	ts *timers

	// whenHeap is a (perhaps outdated) copy of t.when for use
	// ordering t within t.ts.heap.
	// When t is in a heap but t.whenHeap is outdated,
	// the timerModified state bit is set.
	// The actual update t.whenHeap = t.when must be
	// delayed until the heap can be reordered at the same time
	// (meaning t's lock must be held for whenHeap,
	// and t.ts's lock must be held for the heap reordering).
	// Since writes to whenHeap are protected by two locks (t.mu and t.ts.mu),
	// it is permitted to read whenHeap when holding either one.
	whenHeap int64
}

// init initializes a newly allocated timer t.
// Any code that allocates a timer must call t.init before using it.
// The arg and f can be set during init, or they can be nil in init
// and set by a future call to t.modify.
func (t *timer) init(f func(any, uintptr), arg any) {
	lockInit(&t.mu, lockRankTimer)
	t.f = f
	t.arg = arg
}

// A timers is a per-P set of timers.
type timers struct {
	// mu protects timers; timers are per-P, but the scheduler can
	// access the timers of another P, so we have to lock.
	mu mutex

	// heap is the set of timers, ordered by t.whenHeap.
	// Must hold lock to access.
	heap []*timer

	// len is an atomic copy of len(heap).
	len atomic.Uint32

	// zombies is the number of timers in the heap
	// that are marked for removal.
	zombies atomic.Int32

	// raceCtx is the race context used while executing timer functions.
	raceCtx uintptr

	// minWhenHeap is the minimum heap[i].whenHeap value (= heap[0].whenHeap).
	// The wakeTime method uses minWhenHeap and minWhenModified
	// to determine the next wake time.
	// If minWhenHeap = 0, it means there are no timers in the heap.
	minWhenHeap atomic.Int64

	// minWhenModified is a lower bound on the minimum
	// heap[i].when over timers with the timerModified bit set.
	// If minWhenModified = 0, it means there are no timerModified timers in the heap.
	minWhenModified atomic.Int64
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
const (
	// timerHeaped is set when the timer is stored in some P's heap.
	timerHeaped uint8 = 1 << iota

	// timerModified is set when t.when has been modified but
	// t.whenHeap still needs to be updated as well.
	// The change to t.whenHeap waits until the heap in which
	// the timer appears can be locked and rearranged.
	// timerModified is only set when timerHeaped is also set.
	timerModified

	// timerZombie is set when the timer has been stopped
	// but is still present in some P's heap.
	// Only set when timerHeaped is also set.
	// It is possible for timerModified and timerZombie to both
	// be set, meaning that the timer was modified and then stopped.
	timerZombie
)

// lock locks the timer, allowing reading or writing any of the timer fields.
func (t *timer) lock() {
	lock(&t.mu)
}

// unlock updates t.astate and unlocks the timer.
func (t *timer) unlock() {
	// Let heap fast paths know whether t.whenHeap is accurate.
	t.astate.Store(t.state)
	unlock(&t.mu)
}

// updateHeap updates t.whenHeap as directed by t.state, updating t.state
// and returning a bool indicating whether the state (and t.whenHeap) changed.
// The caller must hold t's lock, or the world can be stopped instead.
// If ts != nil, then ts must be locked, t must be ts.heap[0], and updateHeap
// takes care of moving t within the timers heap to preserve the heap invariants.
// If ts == nil, then t must not be in a heap (or is in a heap that is
// temporarily not maintaining its invariant, such as during timers.adjust).
func (t *timer) updateHeap(ts *timers) (updated bool) {
	assertWorldStoppedOrLockHeld(&t.mu)
	if ts != nil {
		if t.ts != ts || t != ts.heap[0] {
			badTimer()
		}
		assertLockHeld(&ts.mu)
	}
	if t.state&timerZombie != 0 {
		// Take timer out of heap, applying final t.whenHeap update first.
		t.state &^= timerHeaped | timerZombie
		if t.state&timerModified != 0 {
			t.state &^= timerModified
			t.whenHeap = t.when
		}
		if ts != nil {
			ts.zombies.Add(-1)
			ts.deleteMin()
		}
		return true
	}

	if t.state&timerModified != 0 {
		// Apply t.whenHeap update and move within heap.
		t.state &^= timerModified
		t.whenHeap = t.when
		// Move t to the right position.
		if ts != nil {
			ts.siftDown(0)
			ts.updateMinWhenHeap()
		}
		return true
	}

	return false
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
		t.init(goroutineReady, gp)
		gp.timer = t
	}
	when := nanotime() + ns
	if when < 0 { // check for overflow.
		when = maxWhen
	}
	gp.sleepWhen = when
	gopark(resetForSleep, nil, waitReasonSleep, traceBlockSleep, 1)
}

// resetForSleep is called after the goroutine is parked for timeSleep.
// We can't call timer.reset in timeSleep itself because if this is a short
// sleep and there are many goroutines then the P can wind up running the
// timer function, goroutineReady, before the goroutine has been parked.
func resetForSleep(gp *g, _ unsafe.Pointer) bool {
	gp.timer.reset(gp.sleepWhen, 0)
	return true
}

// A timeTimer is a runtime-allocated time.Timer or time.Ticker
// with the additional runtime state following it.
// The runtime state is inaccessible to package time.
type timeTimer struct {
	c    unsafe.Pointer // <-chan time.Time
	init bool
	timer
}

// newTimer allocates and returns a new time.Timer or time.Ticker (same layout)
// with the given parameters.
//
//go:linkname newTimer time.newTimer
func newTimer(when, period int64, f func(any, uintptr), arg any) *timeTimer {
	t := new(timeTimer)
	t.timer.init(nil, nil)
	if raceenabled {
		racerelease(unsafe.Pointer(&t.timer))
	}
	t.modify(when, period, f, arg, 0)
	t.init = true
	return t
}

// stopTimer stops a timer.
// It reports whether t was stopped before being run.
//
//go:linkname stopTimer time.stopTimer
func stopTimer(t *timeTimer) bool {
	return t.stop()
}

// resetTimer resets an inactive timer, adding it to the timer heap.
//
// Reports whether the timer was modified before it was run.
//
//go:linkname resetTimer time.resetTimer
func resetTimer(t *timeTimer, when, period int64) bool {
	if raceenabled {
		racerelease(unsafe.Pointer(&t.timer))
	}
	return t.reset(when, period)
}

// Go runtime.

// Ready the goroutine arg.
func goroutineReady(arg any, seq uintptr) {
	goready(arg.(*g), 0)
}

// addHeap adds t to the timers heap.
// The caller must hold ts.lock or the world must be stopped.
// The caller must also have checked that t belongs in the heap.
// Callers that are not sure can call t.maybeAdd instead,
// but note that maybeAdd has different locking requirements.
func (ts *timers) addHeap(t *timer) {
	assertWorldStoppedOrLockHeld(&ts.mu)
	// Timers rely on the network poller, so make sure the poller
	// has started.
	if netpollInited.Load() == 0 {
		netpollGenericInit()
	}

	if t.ts != nil {
		throw("ts set in timer")
	}
	t.ts = ts
	t.whenHeap = t.when
	ts.heap = append(ts.heap, t)
	ts.siftUp(len(ts.heap) - 1)
	if t == ts.heap[0] {
		ts.updateMinWhenHeap()
	}
}

// stop stops the timer t. It may be on some other P, so we can't
// actually remove it from the timers heap. We can only mark it as stopped.
// It will be removed in due course by the P whose heap it is on.
// Reports whether the timer was stopped before it was run.
func (t *timer) stop() bool {
	t.lock()
	if t.state&timerHeaped != 0 {
		t.state |= timerModified
		if t.state&timerZombie == 0 {
			t.state |= timerZombie
			t.ts.zombies.Add(1)
		}
	}
	pending := t.when > 0
	t.when = 0
	t.unlock()
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
	ts.updateMinWhenHeap()
	if last == 0 {
		// If there are no timers, then clearly there are no timerModified timers.
		ts.minWhenModified.Store(0)
	}
}

// modify modifies an existing timer.
// This is called by the netpoll code or time.Ticker.Reset or time.Timer.Reset.
// Reports whether the timer was modified before it was run.
// If f == nil, then t.f, t.arg, and t.seq are not modified.
func (t *timer) modify(when, period int64, f func(any, uintptr), arg any, seq uintptr) bool {
	if when <= 0 {
		throw("timer when must be positive")
	}
	if period < 0 {
		throw("timer period must be non-negative")
	}

	t.lock()
	t.period = period
	if f != nil {
		t.f = f
		t.arg = arg
		t.seq = seq
	}

	wake := false
	pending := t.when > 0
	t.when = when
	if t.state&timerHeaped != 0 {
		t.state |= timerModified
		if t.state&timerZombie != 0 {
			// In the heap but marked for removal (by a Stop).
			// Unmark it, since it has been Reset and will be running again.
			t.ts.zombies.Add(-1)
			t.state &^= timerZombie
		}
		// Cannot modify t.whenHeap until t.ts is locked.
		// See comment in type timer above and in timers.adjust below.
		if when < t.whenHeap {
			wake = true
			t.ts.updateMinWhenModified(when)
		}
	}

	add := t.needsAdd()
	t.unlock()
	if add {
		t.maybeAdd()
	}
	if wake {
		wakeNetPoller(when)
	}

	return pending
}

// needsAdd reports whether t needs to be added to a timers heap.
// t must be locked.
func (t *timer) needsAdd() bool {
	assertLockHeld(&t.mu)
	return t.state&timerHeaped == 0 && t.when > 0
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
// Concurrent calls to time.Timer.Reset
// may result in concurrent calls to t.maybeAdd,
// so we cannot assume that t is not in a heap on entry to t.maybeAdd.
func (t *timer) maybeAdd() {
	ts := &getg().m.p.ptr().timers
	ts.lock()
	ts.cleanHead()
	t.lock()
	when := int64(0)
	if t.needsAdd() {
		t.state |= timerHeaped
		when = t.when
		ts.addHeap(t)
	}
	t.unlock()
	ts.unlock()
	if when > 0 {
		wakeNetPoller(when)
	}
}

// reset resets the time when a timer should fire.
// If used for an inactive timer, the timer will become active.
// Reports whether the timer was active and was stopped.
func (t *timer) reset(when, period int64) bool {
	return t.modify(when, period, nil, nil, 0)
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

		if t.astate.Load()&(timerModified|timerZombie) == 0 {
			// Fast path: head of timers does not need adjustment.
			return
		}

		t.lock()
		updated := t.updateHeap(ts)
		t.unlock()
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
		// The world is stopped, so we ignore the locking of ts and src here.
		// That would introduce a sched < timers lock ordering,
		// which we'd rather avoid in the static ranking.
		ts.move(src.heap)
		src.heap = nil
		src.zombies.Store(0)
		src.minWhenHeap.Store(0)
		src.minWhenModified.Store(0)
		src.len.Store(0)
		ts.len.Store(uint32(len(ts.heap)))
	}
}

// moveTimers moves a slice of timers to pp. The slice has been taken
// from a different P.
// The world must be stopped so that ts is safe to modify.
func (ts *timers) move(timers []*timer) {
	assertWorldStopped()
	for _, t := range timers {
		t.ts = nil
		t.updateHeap(nil)
		if t.state&timerHeaped != 0 {
			ts.addHeap(t)
		}
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
		first := ts.minWhenModified.Load()
		if first == 0 || first > now {
			if verifyTimers {
				ts.verify()
			}
			return
		}
	}

	// minWhenModified is a lower bound on the earliest t.when
	// among the timerModified timers. We want to make it more precise:
	// we are going to scan the heap and clean out all the timerModified bits,
	// at which point minWhenModified can be set to 0 (indicating none at all).
	//
	// Other P's can be calling ts.wakeTime concurrently, and we'd like to
	// keep ts.wakeTime returning an accurate value throughout this entire process.
	//
	// Setting minWhenModified = 0 *before* the scan could make wakeTime
	// return an incorrect value: if minWhenModified < minWhenHeap, then clearing
	// it to 0 will make wakeTime return minWhenHeap (too late) until the scan finishes.
	// To avoid that, we want to set minWhenModified to 0 *after* the scan.
	//
	// Setting minWhenModified = 0 *after* the scan could result in missing
	// concurrent timer modifications in other goroutines; those will lock
	// the specific timer, set the timerModified bit, and set t.when.
	// To avoid that, we want to set minWhenModified to 0 *before* the scan.
	//
	// The way out of this dilemma is to preserve wakeTime a different way.
	// wakeTime is min(minWhenHeap, minWhenModified), and minWhenHeap
	// is protected by ts.lock, which we hold, so we can modify it however we like
	// in service of keeping wakeTime accurate.
	//
	// So we can:
	//
	//	1. Set minWhenHeap = min(minWhenHeap, minWhenModified)
	//	2. Set minWhenModified = 0
	//	   (Other goroutines may modify timers and update minWhenModified now.)
	//	3. Scan timers
	//	4. Set minWhenHeap = heap[0].whenHeap
	//
	// That order preserves a correct value of wakeTime throughout the entire
	// operation:
	// Step 1 “locks in” an accurate wakeTime even with minWhenModified cleared.
	// Step 2 makes sure concurrent t.when updates are not lost during the scan.
	// Step 3 processes all modified timer values, justifying minWhenModified = 0.
	// Step 4 corrects minWhenHeap to a precise value.
	//
	// The wakeTime method implementation reads minWhenModified *before* minWhenHeap,
	// so that if the minWhenModified is observed to be 0, that means the minWhenHeap that
	// follows will include the information that was zeroed out of it.
	ts.minWhenHeap.Store(ts.wakeTime())
	ts.minWhenModified.Store(0)

	changed := false
	for i := 0; i < len(ts.heap); i++ {
		t := ts.heap[i]
		if t.ts != ts {
			throw("bad ts")
		}

		t.lock()
		if t.state&timerHeaped == 0 {
			badTimer()
		}
		if t.state&timerZombie != 0 {
			ts.zombies.Add(-1) // updateHeap will return updated=true and we will delete t
		}
		if t.updateHeap(nil) {
			changed = true
			if t.state&timerHeaped == 0 {
				n := len(ts.heap)
				ts.heap[i] = ts.heap[n-1]
				ts.heap[n-1] = nil
				ts.heap = ts.heap[:n-1]
				t.ts = nil
				i--
			}
		}
		t.unlock()
	}

	if changed {
		ts.initHeap()
	}
	ts.updateMinWhenHeap()

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
	nextWhen := ts.minWhenModified.Load()
	when := ts.minWhenHeap.Load()
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

	if t.astate.Load()&(timerModified|timerZombie) == 0 && t.whenHeap > now {
		// Fast path: not ready to run.
		// The access of t.whenHeap is protected by the caller holding
		// ts.lock, even though t itself is unlocked.
		return t.whenHeap
	}

	t.lock()
	if t.updateHeap(ts) {
		t.unlock()
		goto Redo
	}

	if t.state&timerHeaped == 0 || t.state&timerModified != 0 {
		badTimer()
	}

	if t.when > now {
		// Not ready to run.
		t.unlock()
		return t.when
	}

	t.unlockAndRun(now)
	assertLockHeld(&ts.mu) // t is unlocked now, but not ts
	return 0
}

// unlockAndRun unlocks and runs the timer t (which must be locked).
// If t is in a timer set (t.ts != nil), the caller must also have locked the timer set,
// and this call will temporarily unlock the timer set while running the timer function.
// unlockAndRun returns with t unlocked and t.ts (re-)locked.
//
//go:systemstack
func (t *timer) unlockAndRun(now int64) {
	assertLockHeld(&t.mu)
	if t.ts != nil {
		assertLockHeld(&t.ts.mu)
	}
	if raceenabled {
		// Note that we are running on a system stack,
		// so there is no chance of getg().m being reassigned
		// out from under us while this function executes.
		tsLocal := &getg().m.p.ptr().timers
		if tsLocal.raceCtx == 0 {
			tsLocal.raceCtx = racegostart(abi.FuncPCABIInternal((*timers).run) + sys.PCQuantum)
		}
		raceacquirectx(tsLocal.raceCtx, unsafe.Pointer(t))
	}

	if t.state&(timerModified|timerZombie) != 0 {
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
	if t.state&timerHeaped != 0 {
		t.when = next
		t.state |= timerModified
		if next == 0 {
			t.state |= timerZombie
			t.ts.zombies.Add(1)
		}
	} else {
		t.when = next
	}
	ts := t.ts
	t.updateHeap(ts)
	t.unlock()

	if raceenabled {
		// Temporarily use the current P's racectx for g0.
		gp := getg()
		if gp.racectx != 0 {
			throw("unexpected racectx")
		}
		gp.racectx = gp.m.p.ptr().timers.raceCtx
	}

	if ts != nil {
		ts.unlock()
	}
	f(arg, seq)
	if ts != nil {
		ts.lock()
	}

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
		if t.whenHeap < ts.heap[p].whenHeap {
			print("bad timer heap at ", i, ": ", p, ": ", ts.heap[p].whenHeap, ", ", i, ": ", t.whenHeap, "\n")
			throw("bad timer heap")
		}
	}
	if n := int(ts.len.Load()); len(ts.heap) != n {
		println("timer heap len", len(ts.heap), "!= atomic len", n)
		throw("bad timer heap len")
	}
}

// updateMinWhenHeap sets ts.minWhenHeap to ts.heap[0].whenHeap.
// The caller must have locked ts or the world must be stopped.
func (ts *timers) updateMinWhenHeap() {
	assertWorldStoppedOrLockHeld(&ts.mu)
	if len(ts.heap) == 0 {
		ts.minWhenHeap.Store(0)
	} else {
		ts.minWhenHeap.Store(ts.heap[0].whenHeap)
	}
}

// updateMinWhenModified updates ts.minWhenModified to be <= when.
// ts need not be (and usually is not) locked.
func (ts *timers) updateMinWhenModified(when int64) {
	for {
		old := ts.minWhenModified.Load()
		if old != 0 && old < when {
			return
		}
		if ts.minWhenModified.CompareAndSwap(old, when) {
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
	when := t[i].whenHeap
	if when <= 0 {
		badTimer()
	}
	tmp := t[i]
	for i > 0 {
		p := (i - 1) / 4 // parent
		if when >= t[p].whenHeap {
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
	when := t[i].whenHeap
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
		w := t[c].whenHeap
		if c+1 < n && t[c+1].whenHeap < w {
			w = t[c+1].whenHeap
			c++
		}
		if c3 < n {
			w3 := t[c3].whenHeap
			if c3+1 < n && t[c3+1].whenHeap < w3 {
				w3 = t[c3+1].whenHeap
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

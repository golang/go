// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Time-related runtime and pieces of package time.

package runtime

import (
	"internal/cpu"
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
	throw("new timeSleep not yet implemented")
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

// stopTimer removes t from the timer heap if it is there.
// It returns true if t was removed, false if t wasn't even there.
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

func addtimer(t *timer) {
	if oldTimers {
		addtimerOld(t)
		return
	}
	throw("new addtimer not yet implemented")
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

// Delete timer t from the heap.
// Do not need to update the timerproc: if it wakes up early, no big deal.
func deltimer(t *timer) bool {
	if oldTimers {
		return deltimerOld(t)
	}
	throw("no deltimer not yet implemented")
	return false
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

func modtimer(t *timer, when, period int64, f func(interface{}, uintptr), arg interface{}, seq uintptr) {
	if oldTimers {
		modtimerOld(t, when, period, f, arg, seq)
		return
	}
	throw("new modtimer not yet implemented")
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
	throw("new resettimer not yet implemented")
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

// moveTimers moves a slice of timers to pp. The slice has been taken
// from a different P.
// This is currently called when the world is stopped, but it could
// work as long as the timers for pp are locked.
func moveTimers(pp *p, timers []*timer) {
	throw("movetimers: not yet implemented")
}

// adjusttimers looks through the timers in the current P's heap for
// any timers that have been modified to run earlier, and puts them in
// the correct place in the heap.
// The caller must have locked the timers for pp.
func adjusttimers(pp *p) {
	if len(pp.timers) == 0 {
		return
	}
	throw("adjusttimers: not yet implemented")
}

// runtimer examines the first timer in timers. If it is ready based on now,
// it runs the timer and removes or updates it.
// Returns 0 if it ran a timer, -1 if there are no more timers, or the time
// when the first timer should run.
// The caller must have locked the timers for pp.
func runtimer(pp *p, now int64) int64 {
	throw("runtimer: not yet implemented")
	return -1
}

func timejump() *g {
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

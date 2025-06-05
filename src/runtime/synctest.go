// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
	"internal/runtime/sys"
	"unsafe"
)

// A synctestBubble is a set of goroutines started by synctest.Run.
type synctestBubble struct {
	mu      mutex
	timers  timers
	id      uint64 // unique id
	now     int64  // current fake time
	root    *g     // caller of synctest.Run
	waiter  *g     // caller of synctest.Wait
	main    *g     // goroutine started by synctest.Run
	waiting bool   // true if a goroutine is calling synctest.Wait
	done    bool   // true if main has exited

	// The bubble is active (not blocked) so long as running > 0 || active > 0.
	//
	// running is the number of goroutines which are not "durably blocked":
	// Goroutines which are either running, runnable, or non-durably blocked
	// (for example, blocked in a syscall).
	//
	// active is used to keep the bubble from becoming blocked,
	// even if all goroutines in the bubble are blocked.
	// For example, park_m can choose to immediately unpark a goroutine after parking it.
	// It increments the active count to keep the bubble active until it has determined
	// that the park operation has completed.
	total   int // total goroutines
	running int // non-blocked goroutines
	active  int // other sources of activity
}

// changegstatus is called when the non-lock status of a g changes.
// It is never called with a Gscanstatus.
func (bubble *synctestBubble) changegstatus(gp *g, oldval, newval uint32) {
	// Determine whether this change in status affects the idleness of the bubble.
	// If this isn't a goroutine starting, stopping, durably blocking,
	// or waking up after durably blocking, then return immediately without
	// locking bubble.mu.
	//
	// For example, stack growth (newstack) will changegstatus
	// from _Grunning to _Gcopystack. This is uninteresting to synctest,
	// but if stack growth occurs while bubble.mu is held, we must not recursively lock.
	totalDelta := 0
	wasRunning := true
	switch oldval {
	case _Gdead:
		wasRunning = false
		totalDelta++
	case _Gwaiting:
		if gp.waitreason.isIdleInSynctest() {
			wasRunning = false
		}
	}
	isRunning := true
	switch newval {
	case _Gdead:
		isRunning = false
		totalDelta--
		if gp == bubble.main {
			bubble.done = true
		}
	case _Gwaiting:
		if gp.waitreason.isIdleInSynctest() {
			isRunning = false
		}
	}
	// It's possible for wasRunning == isRunning while totalDelta != 0;
	// for example, if a new goroutine is created in a non-running state.
	if wasRunning == isRunning && totalDelta == 0 {
		return
	}

	lock(&bubble.mu)
	bubble.total += totalDelta
	if wasRunning != isRunning {
		if isRunning {
			bubble.running++
		} else {
			bubble.running--
			if raceenabled && newval != _Gdead {
				// Record that this goroutine parking happens before
				// any subsequent Wait.
				racereleasemergeg(gp, bubble.raceaddr())
			}
		}
	}
	if bubble.total < 0 {
		fatal("total < 0")
	}
	if bubble.running < 0 {
		fatal("running < 0")
	}
	wake := bubble.maybeWakeLocked()
	unlock(&bubble.mu)
	if wake != nil {
		goready(wake, 0)
	}
}

// incActive increments the active-count for the bubble.
// A bubble does not become durably blocked while the active-count is non-zero.
func (bubble *synctestBubble) incActive() {
	lock(&bubble.mu)
	bubble.active++
	unlock(&bubble.mu)
}

// decActive decrements the active-count for the bubble.
func (bubble *synctestBubble) decActive() {
	lock(&bubble.mu)
	bubble.active--
	if bubble.active < 0 {
		throw("active < 0")
	}
	wake := bubble.maybeWakeLocked()
	unlock(&bubble.mu)
	if wake != nil {
		goready(wake, 0)
	}
}

// maybeWakeLocked returns a g to wake if the bubble is durably blocked.
func (bubble *synctestBubble) maybeWakeLocked() *g {
	if bubble.running > 0 || bubble.active > 0 {
		return nil
	}
	// Increment the bubble active count, since we've determined to wake something.
	// The woken goroutine will decrement the count.
	// We can't just call goready and let it increment bubble.running,
	// since we can't call goready with bubble.mu held.
	//
	// Incrementing the active count here is only necessary if something has gone wrong,
	// and a goroutine that we considered durably blocked wakes up unexpectedly.
	// Two wakes happening at the same time leads to very confusing failure modes,
	// so we take steps to avoid it happening.
	bubble.active++
	next := bubble.timers.wakeTime()
	if next > 0 && next <= bubble.now {
		// A timer is scheduled to fire. Wake the root goroutine to handle it.
		return bubble.root
	}
	if gp := bubble.waiter; gp != nil {
		// A goroutine is blocked in Wait. Wake it.
		return gp
	}
	// All goroutines in the bubble are durably blocked, and nothing has called Wait.
	// Wake the root goroutine.
	return bubble.root
}

func (bubble *synctestBubble) raceaddr() unsafe.Pointer {
	// Address used to record happens-before relationships created by the bubble.
	//
	// Wait creates a happens-before relationship between itself and
	// the blocking operations which caused other goroutines in the bubble to park.
	return unsafe.Pointer(bubble)
}

var bubbleGen atomic.Uint64 // bubble ID counter

//go:linkname synctestRun internal/synctest.Run
func synctestRun(f func()) {
	if debug.asynctimerchan.Load() != 0 {
		panic("synctest.Run not supported with asynctimerchan!=0")
	}

	gp := getg()
	if gp.bubble != nil {
		panic("synctest.Run called from within a synctest bubble")
	}
	bubble := &synctestBubble{
		id:      bubbleGen.Add(1),
		total:   1,
		running: 1,
		root:    gp,
	}
	const synctestBaseTime = 946684800000000000 // midnight UTC 2000-01-01
	bubble.now = synctestBaseTime
	lockInit(&bubble.mu, lockRankSynctest)
	lockInit(&bubble.timers.mu, lockRankTimers)

	gp.bubble = bubble
	defer func() {
		gp.bubble = nil
	}()

	// This is newproc, but also records the new g in bubble.main.
	pc := sys.GetCallerPC()
	systemstack(func() {
		fv := *(**funcval)(unsafe.Pointer(&f))
		bubble.main = newproc1(fv, gp, pc, false, waitReasonZero)
		pp := getg().m.p.ptr()
		runqput(pp, bubble.main, true)
		wakep()
	})

	lock(&bubble.mu)
	bubble.active++
	for {
		unlock(&bubble.mu)
		systemstack(func() {
			// Clear gp.m.curg while running timers,
			// so timer goroutines inherit their child race context from g0.
			curg := gp.m.curg
			gp.m.curg = nil
			gp.bubble.timers.check(bubble.now, bubble)
			gp.m.curg = curg
		})
		gopark(synctestidle_c, nil, waitReasonSynctestRun, traceBlockSynctest, 0)
		lock(&bubble.mu)
		if bubble.active < 0 {
			throw("active < 0")
		}
		next := bubble.timers.wakeTime()
		if next == 0 {
			break
		}
		if next < bubble.now {
			throw("time went backwards")
		}
		if bubble.done {
			// Time stops once the bubble's main goroutine has exited.
			break
		}
		bubble.now = next
	}

	total := bubble.total
	unlock(&bubble.mu)
	if raceenabled {
		// Establish a happens-before relationship between bubbled goroutines exiting
		// and Run returning.
		raceacquireg(gp, gp.bubble.raceaddr())
	}
	if total != 1 {
		panic(synctestDeadlockError{bubble})
	}
	if gp.timer != nil && gp.timer.isFake {
		// Verify that we haven't marked this goroutine's sleep timer as fake.
		// This could happen if something in Run were to call timeSleep.
		throw("synctest root goroutine has a fake timer")
	}
}

type synctestDeadlockError struct {
	bubble *synctestBubble
}

func (synctestDeadlockError) Error() string {
	return "deadlock: all goroutines in bubble are blocked"
}

func synctestidle_c(gp *g, _ unsafe.Pointer) bool {
	lock(&gp.bubble.mu)
	canIdle := true
	if gp.bubble.running == 0 && gp.bubble.active == 1 {
		// All goroutines in the bubble have blocked or exited.
		canIdle = false
	} else {
		gp.bubble.active--
	}
	unlock(&gp.bubble.mu)
	return canIdle
}

//go:linkname synctestWait internal/synctest.Wait
func synctestWait() {
	gp := getg()
	if gp.bubble == nil {
		panic("goroutine is not in a bubble")
	}
	lock(&gp.bubble.mu)
	// We use a bubble.waiting bool to detect simultaneous calls to Wait rather than
	// checking to see if bubble.waiter is non-nil. This avoids a race between unlocking
	// bubble.mu and setting bubble.waiter while parking.
	if gp.bubble.waiting {
		unlock(&gp.bubble.mu)
		panic("wait already in progress")
	}
	gp.bubble.waiting = true
	unlock(&gp.bubble.mu)
	gopark(synctestwait_c, nil, waitReasonSynctestWait, traceBlockSynctest, 0)

	lock(&gp.bubble.mu)
	gp.bubble.active--
	if gp.bubble.active < 0 {
		throw("active < 0")
	}
	gp.bubble.waiter = nil
	gp.bubble.waiting = false
	unlock(&gp.bubble.mu)

	// Establish a happens-before relationship on the activity of the now-blocked
	// goroutines in the bubble.
	if raceenabled {
		raceacquireg(gp, gp.bubble.raceaddr())
	}
}

func synctestwait_c(gp *g, _ unsafe.Pointer) bool {
	lock(&gp.bubble.mu)
	if gp.bubble.running == 0 && gp.bubble.active == 0 {
		// This shouldn't be possible, since gopark increments active during unlockf.
		throw("running == 0 && active == 0")
	}
	gp.bubble.waiter = gp
	unlock(&gp.bubble.mu)
	return true
}

//go:linkname synctest_isInBubble internal/synctest.IsInBubble
func synctest_isInBubble() bool {
	return getg().bubble != nil
}

//go:linkname synctest_acquire internal/synctest.acquire
func synctest_acquire() any {
	if bubble := getg().bubble; bubble != nil {
		bubble.incActive()
		return bubble
	}
	return nil
}

//go:linkname synctest_release internal/synctest.release
func synctest_release(bubble any) {
	bubble.(*synctestBubble).decActive()
}

//go:linkname synctest_inBubble internal/synctest.inBubble
func synctest_inBubble(bubble any, f func()) {
	gp := getg()
	if gp.bubble != nil {
		panic("goroutine is already bubbled")
	}
	gp.bubble = bubble.(*synctestBubble)
	defer func() {
		gp.bubble = nil
	}()
	f()
}

// specialBubble is a special used to associate objects with bubbles.
type specialBubble struct {
	_        sys.NotInHeap
	special  special
	bubbleid uint64
}

// getOrSetBubbleSpecial checks the special record for p's bubble membership.
//
// If add is true and p is not associated with any bubble,
// it adds a special record for p associating it with bubbleid.
//
// It returns ok==true if p is associated with bubbleid
// (including if a new association was added),
// and ok==false if not.
func getOrSetBubbleSpecial(p unsafe.Pointer, bubbleid uint64, add bool) (ok bool) {
	span := spanOfHeap(uintptr(p))
	if span == nil {
		throw("getOrSetBubbleSpecial on invalid pointer")
	}

	// Ensure that the span is swept.
	// Sweeping accesses the specials list w/o locks, so we have
	// to synchronize with it. And it's just much safer.
	mp := acquirem()
	span.ensureSwept()

	offset := uintptr(p) - span.base()

	lock(&span.speciallock)

	// Find splice point, check for existing record.
	iter, exists := span.specialFindSplicePoint(offset, _KindSpecialBubble)
	if exists {
		// p is already associated with a bubble.
		// Return true iff it's the same bubble.
		s := (*specialBubble)((unsafe.Pointer)(*iter))
		ok = s.bubbleid == bubbleid
	} else if add {
		// p is not associated with a bubble,
		// and we've been asked to add an association.
		s := (*specialBubble)(mheap_.specialBubbleAlloc.alloc())
		s.bubbleid = bubbleid
		s.special.kind = _KindSpecialBubble
		s.special.offset = offset
		s.special.next = *iter
		*iter = (*special)(unsafe.Pointer(s))
		spanHasSpecials(span)
		ok = true
	} else {
		// p is not associated with a bubble.
		ok = false
	}

	unlock(&span.speciallock)
	releasem(mp)

	return ok
}

// synctest_associate associates p with the current bubble.
// It returns false if p is already associated with a different bubble.
//
//go:linkname synctest_associate internal/synctest.associate
func synctest_associate(p unsafe.Pointer) (ok bool) {
	return getOrSetBubbleSpecial(p, getg().bubble.id, true)
}

// synctest_disassociate disassociates p from its bubble.
//
//go:linkname synctest_disassociate internal/synctest.disassociate
func synctest_disassociate(p unsafe.Pointer) {
	removespecial(p, _KindSpecialBubble)
}

// synctest_isAssociated reports whether p is associated with the current bubble.
//
//go:linkname synctest_isAssociated internal/synctest.isAssociated
func synctest_isAssociated(p unsafe.Pointer) bool {
	return getOrSetBubbleSpecial(p, getg().bubble.id, false)
}

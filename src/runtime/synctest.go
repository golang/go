// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

// A synctestGroup is a group of goroutines started by synctest.Run.
type synctestGroup struct {
	mu      mutex
	timers  timers
	now     int64 // current fake time
	root    *g    // caller of synctest.Run
	waiter  *g    // caller of synctest.Wait
	waiting bool  // true if a goroutine is calling synctest.Wait

	// The group is active (not blocked) so long as running > 0 || active > 0.
	//
	// running is the number of goroutines which are not "durably blocked":
	// Goroutines which are either running, runnable, or non-durably blocked
	// (for example, blocked in a syscall).
	//
	// active is used to keep the group from becoming blocked,
	// even if all goroutines in the group are blocked.
	// For example, park_m can choose to immediately unpark a goroutine after parking it.
	// It increments the active count to keep the group active until it has determined
	// that the park operation has completed.
	total   int // total goroutines
	running int // non-blocked goroutines
	active  int // other sources of activity
}

// changegstatus is called when the non-lock status of a g changes.
// It is never called with a Gscanstatus.
func (sg *synctestGroup) changegstatus(gp *g, oldval, newval uint32) {
	// Determine whether this change in status affects the idleness of the group.
	// If this isn't a goroutine starting, stopping, durably blocking,
	// or waking up after durably blocking, then return immediately without
	// locking sg.mu.
	//
	// For example, stack growth (newstack) will changegstatus
	// from _Grunning to _Gcopystack. This is uninteresting to synctest,
	// but if stack growth occurs while sg.mu is held, we must not recursively lock.
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

	lock(&sg.mu)
	sg.total += totalDelta
	if wasRunning != isRunning {
		if isRunning {
			sg.running++
		} else {
			sg.running--
			if raceenabled && newval != _Gdead {
				racereleasemergeg(gp, sg.raceaddr())
			}
		}
	}
	if sg.total < 0 {
		fatal("total < 0")
	}
	if sg.running < 0 {
		fatal("running < 0")
	}
	wake := sg.maybeWakeLocked()
	unlock(&sg.mu)
	if wake != nil {
		goready(wake, 0)
	}
}

// incActive increments the active-count for the group.
// A group does not become durably blocked while the active-count is non-zero.
func (sg *synctestGroup) incActive() {
	lock(&sg.mu)
	sg.active++
	unlock(&sg.mu)
}

// decActive decrements the active-count for the group.
func (sg *synctestGroup) decActive() {
	lock(&sg.mu)
	sg.active--
	if sg.active < 0 {
		throw("active < 0")
	}
	wake := sg.maybeWakeLocked()
	unlock(&sg.mu)
	if wake != nil {
		goready(wake, 0)
	}
}

// maybeWakeLocked returns a g to wake if the group is durably blocked.
func (sg *synctestGroup) maybeWakeLocked() *g {
	if sg.running > 0 || sg.active > 0 {
		return nil
	}
	// Increment the group active count, since we've determined to wake something.
	// The woken goroutine will decrement the count.
	// We can't just call goready and let it increment sg.running,
	// since we can't call goready with sg.mu held.
	//
	// Incrementing the active count here is only necessary if something has gone wrong,
	// and a goroutine that we considered durably blocked wakes up unexpectedly.
	// Two wakes happening at the same time leads to very confusing failure modes,
	// so we take steps to avoid it happening.
	sg.active++
	if gp := sg.waiter; gp != nil {
		// A goroutine is blocked in Wait. Wake it.
		return gp
	}
	// All goroutines in the group are durably blocked, and nothing has called Wait.
	// Wake the root goroutine.
	return sg.root
}

func (sg *synctestGroup) raceaddr() unsafe.Pointer {
	// Address used to record happens-before relationships created by the group.
	//
	// Wait creates a happens-before relationship between itself and
	// the blocking operations which caused other goroutines in the group to park.
	return unsafe.Pointer(sg)
}

//go:linkname synctestRun internal/synctest.Run
func synctestRun(f func()) {
	if debug.asynctimerchan.Load() != 0 {
		panic("synctest.Run not supported with asynctimerchan!=0")
	}

	gp := getg()
	if gp.syncGroup != nil {
		panic("synctest.Run called from within a synctest bubble")
	}
	gp.syncGroup = &synctestGroup{
		total:   1,
		running: 1,
		root:    gp,
	}
	const synctestBaseTime = 946684800000000000 // midnight UTC 2000-01-01
	gp.syncGroup.now = synctestBaseTime
	gp.syncGroup.timers.syncGroup = gp.syncGroup
	lockInit(&gp.syncGroup.mu, lockRankSynctest)
	lockInit(&gp.syncGroup.timers.mu, lockRankTimers)
	defer func() {
		gp.syncGroup = nil
	}()

	fv := *(**funcval)(unsafe.Pointer(&f))
	newproc(fv)

	sg := gp.syncGroup
	lock(&sg.mu)
	sg.active++
	for {
		if raceenabled {
			// Establish a happens-before relationship between a timer being created,
			// and the timer running.
			raceacquireg(gp, gp.syncGroup.raceaddr())
		}
		unlock(&sg.mu)
		systemstack(func() {
			gp.syncGroup.timers.check(gp.syncGroup.now)
		})
		gopark(synctestidle_c, nil, waitReasonSynctestRun, traceBlockSynctest, 0)
		lock(&sg.mu)
		if sg.active < 0 {
			throw("active < 0")
		}
		next := sg.timers.wakeTime()
		if next == 0 {
			break
		}
		if next < sg.now {
			throw("time went backwards")
		}
		sg.now = next
	}

	total := sg.total
	unlock(&sg.mu)
	if raceenabled {
		// Establish a happens-before relationship between bubbled goroutines exiting
		// and Run returning.
		raceacquireg(gp, gp.syncGroup.raceaddr())
	}
	if total != 1 {
		panic("deadlock: all goroutines in bubble are blocked")
	}
	if gp.timer != nil && gp.timer.isFake {
		// Verify that we haven't marked this goroutine's sleep timer as fake.
		// This could happen if something in Run were to call timeSleep.
		throw("synctest root goroutine has a fake timer")
	}
}

func synctestidle_c(gp *g, _ unsafe.Pointer) bool {
	lock(&gp.syncGroup.mu)
	canIdle := true
	if gp.syncGroup.running == 0 && gp.syncGroup.active == 1 {
		// All goroutines in the group have blocked or exited.
		canIdle = false
	} else {
		gp.syncGroup.active--
	}
	unlock(&gp.syncGroup.mu)
	return canIdle
}

//go:linkname synctestWait internal/synctest.Wait
func synctestWait() {
	gp := getg()
	if gp.syncGroup == nil {
		panic("goroutine is not in a bubble")
	}
	lock(&gp.syncGroup.mu)
	// We use a syncGroup.waiting bool to detect simultaneous calls to Wait rather than
	// checking to see if syncGroup.waiter is non-nil. This avoids a race between unlocking
	// syncGroup.mu and setting syncGroup.waiter while parking.
	if gp.syncGroup.waiting {
		unlock(&gp.syncGroup.mu)
		panic("wait already in progress")
	}
	gp.syncGroup.waiting = true
	unlock(&gp.syncGroup.mu)
	gopark(synctestwait_c, nil, waitReasonSynctestWait, traceBlockSynctest, 0)

	lock(&gp.syncGroup.mu)
	gp.syncGroup.active--
	if gp.syncGroup.active < 0 {
		throw("active < 0")
	}
	gp.syncGroup.waiter = nil
	gp.syncGroup.waiting = false
	unlock(&gp.syncGroup.mu)

	// Establish a happens-before relationship on the activity of the now-blocked
	// goroutines in the group.
	if raceenabled {
		raceacquireg(gp, gp.syncGroup.raceaddr())
	}
}

func synctestwait_c(gp *g, _ unsafe.Pointer) bool {
	lock(&gp.syncGroup.mu)
	if gp.syncGroup.running == 0 && gp.syncGroup.active == 0 {
		// This shouldn't be possible, since gopark increments active during unlockf.
		throw("running == 0 && active == 0")
	}
	gp.syncGroup.waiter = gp
	unlock(&gp.syncGroup.mu)
	return true
}

//go:linkname synctest_acquire internal/synctest.acquire
func synctest_acquire() any {
	if sg := getg().syncGroup; sg != nil {
		sg.incActive()
		return sg
	}
	return nil
}

//go:linkname synctest_release internal/synctest.release
func synctest_release(sg any) {
	sg.(*synctestGroup).decActive()
}

//go:linkname synctest_inBubble internal/synctest.inBubble
func synctest_inBubble(sg any, f func()) {
	gp := getg()
	if gp.syncGroup != nil {
		panic("goroutine is already bubbled")
	}
	gp.syncGroup = sg.(*synctestGroup)
	defer func() {
		gp.syncGroup = nil
	}()
	f()
}

// Local fallback improvement: appended a small comment.

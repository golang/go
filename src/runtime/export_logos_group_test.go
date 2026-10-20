// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/runtime/atomic"

// LogosSlotsAcquire takes n group slots through the entry point
// logos/rt.NewGroup uses.
func LogosSlotsAcquire(n int) []uint32 {
	slots := make([]uint32, n)
	for i := range slots {
		slots[i] = logosGroupAcquire()
	}
	return slots
}

// LogosSlotsRelease returns slots through the entry point
// logos/rt.Group.Wait uses.
func LogosSlotsRelease(slots []uint32) {
	for _, slot := range slots {
		logosGroupRelease(slot)
	}
}

// LogosStaleReleaseKeepsCancel registers the calling goroutine as a
// group waiter, retires its descriptor through the reuse purge, lets a
// new waiter's cancel bit land on the reused descriptor, and then runs
// the stale waiter's late release. It reports whether the new bit
// survived.
func LogosStaleReleaseKeepsCancel() bool {
	pd := pollcache.alloc()
	defer pollcache.free(pd)
	slot := logosGroupAcquire()
	defer logosGroupRelease(slot)
	gp := getg()
	gp.logosGroup = slot
	registered := logosWaitRegister(pd, 'r')
	gp.logosGroup = 0
	if !registered {
		return false
	}
	logosPollPurge(pd)
	atomic.Or32(&pd.logosCancel, logosCancelRead)
	logosWaitRelease(pd, 'r')
	kept := logosCancelled(pd, 'r')
	logosPollPurge(pd)
	return kept
}

// LogosPurgeClearsBit sets both cancel bits on a descriptor that has no
// registered waiter, purges it, and reports whether the bits are gone:
// a reused descriptor starts clean even on the lock-free idle path.
func LogosPurgeClearsBit() bool {
	pd := pollcache.alloc()
	defer pollcache.free(pd)
	atomic.Or32(&pd.logosCancel, logosCancelRead|logosCancelWrite)
	logosPollPurge(pd)
	return atomic.Load(&pd.logosCancel) == 0
}

// LogosRegistryIdle reports whether the published waiter count says the
// registry is empty.
func LogosRegistryIdle() bool {
	return LogosWaiterCount() == 0
}

// LogosWaiterCount returns the published number of registered group
// waiters.
func LogosWaiterCount() int {
	return int(atomic.Load(&logosWaiterCount))
}

// LogosCancelFiltersByGroup registers a waiter of one group and cancels
// first a different group, then its own. It reports the cancel bit seen
// after each cancel.
func LogosCancelFiltersByGroup() (otherBit, ownBit bool) {
	pd := pollcache.alloc()
	defer pollcache.free(pd)
	own, other := logosGroupAcquire(), logosGroupAcquire()
	defer logosGroupRelease(own)
	defer logosGroupRelease(other)
	gp := getg()
	gp.logosGroup = own
	logosWaitRegister(pd, 'r')
	gp.logosGroup = 0
	logosCancelGroup(other)
	otherBit = logosCancelled(pd, 'r')
	logosCancelGroup(own)
	ownBit = logosCancelled(pd, 'r')
	logosPollPurge(pd)
	return otherBit, ownBit
}

// LogosReusedSlotStartsClean cancels a slot, returns it, lends the same
// slot out again, and registers a waiter tagged with it. It reports
// whether the slot was reused and the fresh registration stayed
// unarmed: neither the release nor the retake may carry the old
// cancellation into the next group.
func LogosReusedSlotStartsClean() bool {
	pd := pollcache.alloc()
	defer pollcache.free(pd)
	slot := logosGroupAcquire()
	logosCancelGroup(slot)
	logosGroupRelease(slot)
	retaken := logosGroupAcquire()
	defer logosGroupRelease(retaken)
	if retaken != slot {
		return false
	}
	gp := getg()
	gp.logosGroup = retaken
	registered := logosWaitRegister(pd, 'r')
	gp.logosGroup = 0
	if !registered {
		return false
	}
	armed := logosCancelled(pd, 'r')
	logosWaitRelease(pd, 'r')
	logosPollPurge(pd)
	return !armed
}

// LogosReusedSlotCancel holds a waiter of one live group while the slot
// of a second group is released and lent out again, then cancels the
// reused slot. It reports the live group's slot, the reused slot, and
// whether the cancel set the live waiter's bit.
func LogosReusedSlotCancel() (live, reused uint32, liveCancelled bool) {
	pd := pollcache.alloc()
	defer pollcache.free(pd)
	live, other := logosGroupAcquire(), logosGroupAcquire()
	gp := getg()
	gp.logosGroup = live
	logosWaitRegister(pd, 'r')
	gp.logosGroup = 0
	logosGroupRelease(other)
	reused = logosGroupAcquire()
	logosCancelGroup(reused)
	liveCancelled = logosCancelled(pd, 'r')
	logosPollPurge(pd)
	logosGroupRelease(reused)
	logosGroupRelease(live)
	return live, reused, liveCancelled
}

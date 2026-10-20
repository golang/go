// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
	_ "unsafe" // linkname push below
)

// This file implements the M8-S4 wake of group children blocked in
// netpoll-backed I/O (SPEC Sec 2.6, T-206). A goroutine tagged with a
// group slot (logosSetGroup) registers itself as a waiter while it sits in
// poll_runtime_pollWait. logosCancelGroup wakes exactly the registered
// waiters of one group: it sets the cancel bit of the waiter's mode and
// unblocks the parked goroutine, and netpollcheckerr then reports a
// timeout. It also records the failure on the group slot, so a
// goroutine that registers afterwards arms its wait instead of parking.
// The waiter clears its bit when it leaves the wait, so the
// descriptor stays usable.
//
// Lock order: logosWaitersLock is taken alone (register, release, slot
// take and free, see logos_group.go) or before a descriptor lock
// (cancel). No path takes a descriptor lock before logosWaitersLock, so
// the order is acyclic. Descriptors are type-stable (pollCache), so a
// registry entry never names freed memory;
// an entry exists only while its goroutine waits, so it never names a
// reused descriptor.

// logosCancelRead and logosCancelWrite are the pollDesc.logosCancel bits
// for the read and write waiters.
const (
	logosCancelRead  uint32 = 1
	logosCancelWrite uint32 = 2
)

// logosWaiter is one registered group waiter: the goroutine parked (or
// about to park) on pd's mode, and the group it belongs to.
type logosWaiter struct {
	pd    *pollDesc
	gp    *g
	mode  int32
	group uint32
}

var (
	logosWaitersLock mutex
	logosWaiters     []logosWaiter
	logosWaiterCount uint32
)

// init ranks the registry lock before any group goroutine can run. The
// lock ranks before pollDesc: logosCancelGroup takes a descriptor lock
// while holding it, and registering appends (allocates) under it, which
// the chain from pollDesc to the scheduler and MALLOC permits. No other
// runtime lock is held when a group waiter registers, releases, or
// cancels, or when a group slot is taken or freed.
func init() {
	lockInit(&logosWaitersLock, lockRankLogosWaiters)
}

// logosWaitersSync publishes the registry length for the lock-free idle
// check of logosPollPurge. Callers hold logosWaitersLock.
func logosWaitersSync() {
	atomic.Store(&logosWaiterCount, uint32(len(logosWaiters)))
}

// logosCancelBit returns the pollDesc.logosCancel bit of mode.
func logosCancelBit(mode int32) uint32 {
	if mode == 'w' {
		return logosCancelWrite
	}
	return logosCancelRead
}

// logosCancelled reports whether the group cancel bit of mode is set on
// pd. netpollcheckerr peeks it; only the waiter clears it.
func logosCancelled(pd *pollDesc, mode int32) bool {
	return atomic.Load(&pd.logosCancel)&logosCancelBit(mode) != 0
}

// logosWaitRegister registers the calling goroutine as a waiter of pd in
// mode when it belongs to a group. When the group has already failed,
// it arms the wait first, so the check inside netpollblock reports the
// cancel without parking. It reports whether it registered, so the
// caller pairs it with logosWaitRelease.
func logosWaitRegister(pd *pollDesc, mode int32) bool {
	gp := getg()
	if gp.logosGroup == 0 {
		return false
	}
	lock(&logosWaitersLock)
	if logosSlotCancelGet(gp.logosGroup) {
		atomic.Or32(&pd.logosCancel, logosCancelBit(mode))
	}
	logosWaiters = append(logosWaiters, logosWaiter{pd: pd, gp: gp, mode: mode, group: gp.logosGroup})
	logosWaitersSync()
	unlock(&logosWaitersLock)
	return true
}

// logosWaitRelease removes the calling goroutine's registration under
// the registry lock, then clears its cancel bit: no cancel can set the
// bit after the entry is gone, so the clear races nothing. The clear is
// scoped to the caller's own mode and only when it owned an entry, so a
// stray release never erases another waiter's cancellation.
func logosWaitRelease(pd *pollDesc, mode int32) {
	gp := getg()
	owned := false
	lock(&logosWaitersLock)
	for i := range logosWaiters {
		if logosWaiters[i].gp == gp {
			last := len(logosWaiters) - 1
			logosWaiters[i] = logosWaiters[last]
			logosWaiters[last] = logosWaiter{}
			logosWaiters = logosWaiters[:last]
			logosWaitersSync()
			owned = true
			break
		}
	}
	if owned {
		atomic.And32(&pd.logosCancel, ^logosCancelBit(mode))
	}
	unlock(&logosWaitersLock)
}

// logosPollPurge drops every registry entry naming pd and clears its
// cancel bits. pollOpen calls it when a descriptor leaves the free
// cache, so a reused descriptor starts with no stale group waiters;
// pollClose calls it before the descriptor retires, so a registered
// waiter of a closing descriptor cannot outlive it and name the next
// occupant. The registry lock covers the list and the bit clear, the
// same window every other bit mutation uses. While no group waiter is
// registered anywhere, only the bit clear remains, so ordinary programs
// never take the registry lock when a descriptor opens or closes: an
// entry for pd cannot appear meanwhile, since nobody waits on a
// descriptor that is opening or has no remaining references.
func logosPollPurge(pd *pollDesc) {
	if atomic.Load(&logosWaiterCount) == 0 {
		atomic.Store(&pd.logosCancel, 0)
		return
	}
	lock(&logosWaitersLock)
	last := len(logosWaiters)
	for i := 0; i < last; {
		if logosWaiters[i].pd == pd {
			last--
			logosWaiters[i] = logosWaiters[last]
			logosWaiters[last] = logosWaiter{}
			continue
		}
		i++
	}
	logosWaiters = logosWaiters[:last]
	logosWaitersSync()
	atomic.And32(&pd.logosCancel, 0)
	unlock(&logosWaitersLock)
}

// logosCancelGroup records the failure of group slot, then wakes every
// registered waiter of it. For each one it sets the cancel bit of its
// mode under the descriptor lock and unblocks the goroutine parked
// there, if any, and readies that goroutine after the registry lock is
// released. The record outlives the scan, so a waiter that registers
// afterwards still arms its wait.
//
//go:linkname logosCancelGroup
func logosCancelGroup(slot uint32) {
	var toRun gList
	var delta int32
	lock(&logosWaitersLock)
	logosSlotCancelSet(slot)
	for i := range logosWaiters {
		w := &logosWaiters[i]
		if w.group != slot {
			continue
		}
		pd := w.pd
		lock(&pd.lock)
		atomic.Or32(&pd.logosCancel, logosCancelBit(w.mode))
		if gp := netpollunblock(pd, w.mode, false, &delta); gp != nil {
			toRun.push(gp)
		}
		unlock(&pd.lock)
	}
	unlock(&logosWaitersLock)
	for !toRun.empty() {
		netpollgoready(toRun.pop(), 0)
	}
	netpollAdjustWaiters(delta)
}

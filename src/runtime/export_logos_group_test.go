// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/runtime/atomic"

// logosTestGroup is a group ID no real logos/rt.Group mints: real IDs
// count up from 1.
const logosTestGroup uint64 = 1 << 62

// LogosStaleReleaseKeepsCancel registers the calling goroutine as a
// group waiter, retires its descriptor through the reuse purge, lets a
// new waiter's cancel bit land on the reused descriptor, and then runs
// the stale waiter's late release. It reports whether the new bit
// survived.
func LogosStaleReleaseKeepsCancel() bool {
	pd := pollcache.alloc()
	defer pollcache.free(pd)
	gp := getg()
	gp.logosGroup = logosTestGroup
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
	return atomic.Load(&logosWaiterCount) == 0
}

// LogosCancelFiltersByGroup registers a waiter of one group and cancels
// first a different group, then its own. It reports the cancel bit seen
// after each cancel.
func LogosCancelFiltersByGroup() (otherBit, ownBit bool) {
	pd := pollcache.alloc()
	defer pollcache.free(pd)
	gp := getg()
	gp.logosGroup = logosTestGroup
	logosWaitRegister(pd, 'r')
	gp.logosGroup = 0
	logosCancelGroup(logosTestGroup + 1)
	otherBit = logosCancelled(pd, 'r')
	logosCancelGroup(logosTestGroup)
	ownBit = logosCancelled(pd, 'r')
	logosPollPurge(pd)
	return otherBit, ownBit
}

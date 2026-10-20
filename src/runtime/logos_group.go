// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/sys"
	_ "unsafe"
)

// This file implements the per-goroutine group tag backing
// logos/rt.Group. Each goroutine carries the slot of the loGo/s group it
// runs for in its g field (0 outside a group). Group.Go tags its child
// at start through logosSetGroup and clears the tag on exit, so a
// recycled g never keeps a stale slot: every tag is set before use, and
// the deferred clear runs on every exit path (return, recovered panic,
// and Goexit runs deferred calls).
//
// The tag is a slot index, not a counter, which keeps it 32 bits wide.
// logos/rt.NewGroup takes the lowest free slot >= 1 from
// logosGroupAcquire, and Group.Wait returns it through logosGroupRelease
// once every child has joined. A child clears its tag before it reports
// done, so after the join no goroutine carries the slot and a slot names
// at most one live group at a time: netpoll wake, which filters waiters
// by slot, never reaches a goroutine of another live group. A group that
// is never joined keeps its slot for good, which wastes one bit but
// never aliases.
//
// The slot bitmap is guarded by logosWaitersLock, the lock of the
// registry that the wake filter reads.
//
// logosSetGroup, logosGroupAcquire, and logosGroupRelease are referenced
// from logos/rt via //go:linkname pull and carry a push-style
// //go:linkname mark here, so the linker's linkname check accepts the
// pull.

// logosSlotLimit is the largest slot a goroutine tag can hold.
const logosSlotLimit = 1<<32 - 1

var (
	// logosSlotBits is the slot bitmap: bit i of word w is set while slot
	// w*64+i+1 is held. Guarded by logosWaitersLock.
	logosSlotBits []uint64
	// logosSlotCancel parallels logosSlotBits: bit i of word w is set
	// once the group of slot w*64+i+1 has failed, until the slot is lent
	// out again, so a waiter that registers after the cancel still arms
	// its wait. Guarded by logosWaitersLock.
	logosSlotCancel []uint64
	// logosSlotHint is the lowest word of logosSlotBits that can hold a
	// free bit, so a take skips the full prefix. Guarded by
	// logosWaitersLock.
	logosSlotHint int
)

//go:linkname logosSetGroup
func logosSetGroup(slot uint32) {
	getg().logosGroup = slot
}

// logosGroupAcquire lends the lowest free group slot >= 1.
//
//go:linkname logosGroupAcquire
func logosGroupAcquire() uint32 {
	lock(&logosWaitersLock)
	slot := logosSlotTake()
	unlock(&logosWaitersLock)
	return slot
}

// logosGroupRelease returns a slot lent by logosGroupAcquire. The caller
// must have joined every goroutine tagged with it.
//
//go:linkname logosGroupRelease
func logosGroupRelease(slot uint32) {
	lock(&logosWaitersLock)
	logosSlotFree(slot)
	unlock(&logosWaitersLock)
}

// logosSlotTake marks the lowest free slot held and returns it. The
// caller holds logosWaitersLock. More than logosSlotLimit live groups
// cannot exist in memory, so exhaustion is fatal.
func logosSlotTake() uint32 {
	for w := logosSlotHint; ; w++ {
		if w == len(logosSlotBits) {
			logosSlotBits = append(logosSlotBits, 0)
			logosSlotCancel = append(logosSlotCancel, 0)
		}
		free := ^logosSlotBits[w]
		if free == 0 {
			continue
		}
		bit := sys.TrailingZeros64(free)
		slot := uint64(w)*64 + uint64(bit) + 1
		if slot > logosSlotLimit {
			throw("logos: group slots exhausted")
		}
		logosSlotBits[w] |= 1 << bit
		logosSlotCancel[w] &^= 1 << bit
		logosSlotHint = w
		return uint32(slot)
	}
}

// logosSlotFree marks slot free. The caller holds logosWaitersLock.
// Freeing a slot that is not held would let two live groups share it, so
// it is fatal.
func logosSlotFree(slot uint32) {
	i := uint64(slot) - 1
	w, mask := int(i/64), uint64(1)<<(i%64)
	if slot == 0 || w >= len(logosSlotBits) || logosSlotBits[w]&mask == 0 {
		throw("logos: release of a group slot that is not held")
	}
	logosSlotBits[w] &^= mask
	logosSlotCancel[w] &^= mask
	logosSlotHint = min(logosSlotHint, w)
}

// logosSlotCancelSet records the failure of slot, so a waiter that
// registers afterwards arms its wait instead of parking. The caller
// holds logosWaitersLock and slot is held.
func logosSlotCancelSet(slot uint32) {
	if slot == 0 {
		return
	}
	i := uint64(slot) - 1
	w, mask := int(i/64), uint64(1)<<(i%64)
	if w < len(logosSlotCancel) {
		logosSlotCancel[w] |= mask
	}
}

// logosSlotCancelGet reports whether the group of slot has failed. The
// caller holds logosWaitersLock.
func logosSlotCancelGet(slot uint32) bool {
	if slot == 0 {
		return false
	}
	i := uint64(slot) - 1
	w, mask := int(i/64), uint64(1)<<(i%64)
	return w < len(logosSlotCancel) && logosSlotCancel[w]&mask != 0
}

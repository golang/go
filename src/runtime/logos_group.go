// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/sys"
	_ "unsafe"
)

var logosWaitersLock mutex

func init() {
	lockInit(&logosWaitersLock, lockRankLogosWaiters)
}

const logosSlotLimit = 1<<32 - 1

var (
	logosSlotBits   []uint64
	logosSlotCancel []uint64
	logosSlotHint   int
)

//go:linkname logosSetGroup
func logosSetGroup(slot uint32) {
	getg().logosGroup = slot
}

//go:linkname logosGroupAcquire
func logosGroupAcquire() uint32 {
	lock(&logosWaitersLock)
	slot := logosSlotTake()
	unlock(&logosWaitersLock)
	return slot
}

//go:linkname logosGroupRelease
func logosGroupRelease(slot uint32) {
	lock(&logosWaitersLock)
	logosSlotFree(slot)
	unlock(&logosWaitersLock)
}

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

func logosSlotCancelGet(slot uint32) bool {
	if slot == 0 {
		return false
	}
	i := uint64(slot) - 1
	w, mask := int(i/64), uint64(1)<<(i%64)
	return w < len(logosSlotCancel) && logosSlotCancel[w]&mask != 0
}

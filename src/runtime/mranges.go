// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Address range data structure.
//
// This file contains an implementation of a data structure which
// manages ordered address ranges.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

// addrRange represents a region of address space.
type addrRange struct {
	// base and limit together represent the region of address space
	// [base, limit). That is, base is inclusive, limit is exclusive.
	base, limit uintptr
}

// size returns the size of the range represented in bytes.
func (a addrRange) size() uintptr {
	if a.limit <= a.base {
		return 0
	}
	return a.limit - a.base
}

// contains returns whether or not the range contains a given address.
func (a addrRange) contains(addr uintptr) bool {
	return addr >= a.base && addr < a.limit
}

// subtract takes the addrRange toPrune and cuts out any overlap with
// from, then returns the new range. subtract assumes that a and b
// either don't overlap at all, only overlap on one side, or are equal.
// If b is strictly contained in a, thus forcing a split, it will throw.
func (a addrRange) subtract(b addrRange) addrRange {
	if a.base >= b.base && a.limit <= b.limit {
		return addrRange{}
	} else if a.base < b.base && a.limit > b.limit {
		throw("bad prune")
	} else if a.limit > b.limit && a.base < b.limit {
		a.base = b.limit
	} else if a.base < b.base && a.limit > b.base {
		a.limit = b.base
	}
	return a
}

// addrRanges is a data structure holding a collection of ranges of
// address space.
//
// The ranges are coalesced eagerly to reduce the
// number ranges it holds.
//
// The slice backing store for this field is persistentalloc'd
// and thus there is no way to free it.
//
// addrRanges is not thread-safe.
type addrRanges struct {
	// ranges is a slice of ranges sorted by base.
	ranges []addrRange

	// sysStat is the stat to track allocations by this type
	sysStat *uint64
}

func (a *addrRanges) init(sysStat *uint64) {
	ranges := (*notInHeapSlice)(unsafe.Pointer(&a.ranges))
	ranges.len = 0
	ranges.cap = 16
	ranges.array = (*notInHeap)(persistentalloc(unsafe.Sizeof(addrRange{})*uintptr(ranges.cap), sys.PtrSize, sysStat))
	a.sysStat = sysStat
}

// findSucc returns the first index in a such that base is
// less than the base of the addrRange at that index.
func (a *addrRanges) findSucc(base uintptr) int {
	// TODO(mknyszek): Consider a binary search for large arrays.
	// While iterating over these ranges is potentially expensive,
	// the expected number of ranges is small, ideally just 1,
	// since Go heaps are usually mostly contiguous.
	for i := range a.ranges {
		if base < a.ranges[i].base {
			return i
		}
	}
	return len(a.ranges)
}

// contains returns true if a covers the address addr.
func (a *addrRanges) contains(addr uintptr) bool {
	i := a.findSucc(addr)
	if i == 0 {
		return false
	}
	return a.ranges[i-1].contains(addr)
}

// add inserts a new address range to a.
//
// r must not overlap with any address range in a.
func (a *addrRanges) add(r addrRange) {
	// The copies in this function are potentially expensive, but this data
	// structure is meant to represent the Go heap. At worst, copying this
	// would take ~160µs assuming a conservative copying rate of 25 GiB/s (the
	// copy will almost never trigger a page fault) for a 1 TiB heap with 4 MiB
	// arenas which is completely discontiguous. ~160µs is still a lot, but in
	// practice most platforms have 64 MiB arenas (which cuts this by a factor
	// of 16) and Go heaps are usually mostly contiguous, so the chance that
	// an addrRanges even grows to that size is extremely low.

	// Because we assume r is not currently represented in a,
	// findSucc gives us our insertion index.
	i := a.findSucc(r.base)
	coalescesDown := i > 0 && a.ranges[i-1].limit == r.base
	coalescesUp := i < len(a.ranges) && r.limit == a.ranges[i].base
	if coalescesUp && coalescesDown {
		// We have neighbors and they both border us.
		// Merge a.ranges[i-1], r, and a.ranges[i] together into a.ranges[i-1].
		a.ranges[i-1].limit = a.ranges[i].limit

		// Delete a.ranges[i].
		copy(a.ranges[i:], a.ranges[i+1:])
		a.ranges = a.ranges[:len(a.ranges)-1]
	} else if coalescesDown {
		// We have a neighbor at a lower address only and it borders us.
		// Merge the new space into a.ranges[i-1].
		a.ranges[i-1].limit = r.limit
	} else if coalescesUp {
		// We have a neighbor at a higher address only and it borders us.
		// Merge the new space into a.ranges[i].
		a.ranges[i].base = r.base
	} else {
		// We may or may not have neighbors which don't border us.
		// Add the new range.
		if len(a.ranges)+1 > cap(a.ranges) {
			// Grow the array. Note that this leaks the old array, but since
			// we're doubling we have at most 2x waste. For a 1 TiB heap and
			// 4 MiB arenas which are all discontiguous (both very conservative
			// assumptions), this would waste at most 4 MiB of memory.
			oldRanges := a.ranges
			ranges := (*notInHeapSlice)(unsafe.Pointer(&a.ranges))
			ranges.len = len(oldRanges) + 1
			ranges.cap = cap(oldRanges) * 2
			ranges.array = (*notInHeap)(persistentalloc(unsafe.Sizeof(addrRange{})*uintptr(ranges.cap), sys.PtrSize, a.sysStat))

			// Copy in the old array, but make space for the new range.
			copy(a.ranges[:i], oldRanges[:i])
			copy(a.ranges[i+1:], oldRanges[i:])
		} else {
			a.ranges = a.ranges[:len(a.ranges)+1]
			copy(a.ranges[i+1:], a.ranges[i:])
		}
		a.ranges[i] = r
	}
}

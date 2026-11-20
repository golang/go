// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps

import (
	"internal/abi"
	"unsafe"
)

type CtrlGroup = ctrlGroup

// Bit 7 denotes slot 7 in a packed mask and slot 0 in a byte-wise mask.
func BitsetIsPacked() bool { return bitsetFirst(1<<7) == 7 }

const CtrlEmpty = ctrlEmpty
const CtrlDeleted = ctrlDeleted

var H2 = h2

// Function values keep these calls out of the compiler's intrinsic path.
var ctrlMatchH2Portable = ctrlGroupMatchH2
var ctrlMatchEmptyPortable = ctrlGroupMatchEmpty
var ctrlMatchEmptyOrDeletedPortable = ctrlGroupMatchEmptyOrDeleted
var ctrlMatchFullPortable = ctrlGroupMatchFull

func CtrlGroupMatch(g CtrlGroup, h ctrl, portable bool) [4]uint64 {
	if portable {
		return [4]uint64{
			uint64(ctrlMatchH2Portable(g, h)),
			uint64(ctrlMatchEmptyPortable(g)),
			uint64(ctrlMatchEmptyOrDeletedPortable(g)),
			uint64(ctrlMatchFullPortable(g)),
		}
	}
	return [4]uint64{
		uint64(g.matchH2(h)),
		uint64(g.matchEmpty()),
		uint64(g.matchEmptyOrDeleted()),
		uint64(g.matchFull()),
	}
}

func CtrlGroupAnyFull(g CtrlGroup) bool { return g.anyFull() }

func CtrlGroupSetEmpty(g CtrlGroup) CtrlGroup {
	g.setEmpty()
	return g
}

func CtrlGroupSet(g CtrlGroup, i uintptr, c uint8) (CtrlGroup, uint8) {
	g.set(i, ctrl(c))
	return g, uint8(g.get(i))
}

const DebugLog = debugLog

var AlignUpPow2 = alignUpPow2

const MaxTableCapacity = maxTableCapacity
const MaxAvgGroupLoad = maxAvgGroupLoad

// This isn't equivalent to runtime.maxAlloc. It is fine for basic testing but
// we can't properly test hint alloc overflows with this.
const maxAllocTest = 1 << 30

// Do not convert to var-of-funcval, these are used in benchmarks and must be
// inlined.
func MemHashAES(p unsafe.Pointer, h, s uintptr) uintptr      { return memHashAES(p, h, s) }
func MemHashFallback(p unsafe.Pointer, h, s uintptr) uintptr { return memHashFallback(p, h, s) }

func newTestMapType[K comparable, V any]() *abi.MapType {
	var m map[K]V
	mTyp := abi.TypeOf(m)
	mt := (*abi.MapType)(unsafe.Pointer(mTyp))
	return mt
}

func NewTestMap[K comparable, V any](hint uintptr) (*Map, *abi.MapType) {
	mt := newTestMapType[K, V]()
	return newMap(mt, hint, nil, maxAllocTest), mt
}

// GrowMaxAlloc is Grow with an explicit allocation limit.
func (m *Map) GrowMaxAlloc(typ *abi.MapType, n, maxAlloc uintptr) {
	m.grow(typ, n, maxAlloc)
}

func (m *Map) TableCount() int {
	if m.dirLen <= 0 {
		return 0
	}
	return m.dirLen
}

// Total group count, summed across all tables.
func (m *Map) GroupCount() uint64 {
	if m.dirLen <= 0 {
		if m.dirPtr == nil {
			return 0
		}
		return 1
	}

	var n uint64
	var lastTab *table
	for i := range m.dirLen {
		t := m.directoryAt(uintptr(i))
		if t == lastTab {
			continue
		}
		lastTab = t
		n += t.groups.lengthMask + 1
	}
	return n
}

// Return a key from a group containing no empty slots.
//
// Returns nil if there are no full groups.
// Returns nil if a group is full but contains entirely deleted slots.
// Returns nil if the map is small.
func (m *Map) KeyFromFullGroup(typ *abi.MapType) unsafe.Pointer {
	if m.dirLen <= 0 {
		return nil
	}

	var lastTab *table
	for i := range m.dirLen {
		t := m.directoryAt(uintptr(i))
		if t == lastTab {
			continue
		}
		lastTab = t

		for i := uint64(0); i <= t.groups.lengthMask; i++ {
			g := t.groups.group(typ, i)
			match := g.ctrls().matchEmpty()
			if match != 0 {
				continue
			}

			// All full or deleted slots.
			for j := uintptr(0); j < abi.MapGroupSlots; j++ {
				if g.ctrls().get(j) == ctrlDeleted {
					continue
				}
				slotKey := g.key(typ, j)
				if typ.IndirectKey() {
					slotKey = *((*unsafe.Pointer)(slotKey))
				}
				return slotKey
			}
		}
	}

	return nil
}

// Returns nil if the map is small.
func (m *Map) TableFor(typ *abi.MapType, key unsafe.Pointer) *table {
	if m.dirLen <= 0 {
		return nil
	}

	hash := typ.Hasher(key, m.seed)
	idx := m.directoryIndex(hash)
	return m.directoryAt(idx)
}

func (t *table) GrowthLeft() uint64 {
	return uint64(t.growthLeft)
}

// Returns the start address of the groups array.
func (t *table) GroupsStart() unsafe.Pointer {
	return t.groups.data
}

// Returns the length of the groups array.
func (t *table) GroupsLength() uintptr {
	return uintptr(t.groups.lengthMask + 1)
}

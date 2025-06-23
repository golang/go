// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps

import (
	"internal/abi"
	"unsafe"
)

type CtrlGroup = ctrlGroup

const DebugLog = debugLog

var AlignUpPow2 = alignUpPow2

const MaxTableCapacity = maxTableCapacity
const MaxAvgGroupLoad = maxAvgGroupLoad

// This isn't equivalent to runtime.maxAlloc. It is fine for basic testing but
// we can't properly test hint alloc overflows with this.
const maxAllocTest = 1 << 30

func NewTestMap[K comparable, V any](hint uintptr) (*Map, *abi.SwissMapType) {
	mt := newTestMapType[K, V]()
	return NewMap(mt, hint, nil, maxAllocTest), mt
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
func (m *Map) KeyFromFullGroup(typ *abi.SwissMapType) unsafe.Pointer {
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
			for j := uintptr(0); j < abi.SwissMapGroupSlots; j++ {
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
func (m *Map) TableFor(typ *abi.SwissMapType, key unsafe.Pointer) *table {
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

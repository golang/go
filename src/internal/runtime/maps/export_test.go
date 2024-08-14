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

func NewTestMap[K comparable, V any](length uint64) (*Map, *abi.SwissMapType) {
	mt := newTestMapType[K, V]()
	return NewMap(mt, length), mt
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
func (m *Map) KeyFromFullGroup() unsafe.Pointer {
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
			g := t.groups.group(i)
			match := g.ctrls().matchEmpty()
			if match != 0 {
				continue
			}

			// All full or deleted slots.
			for j := uint32(0); j < abi.SwissMapGroupSlots; j++ {
				if g.ctrls().get(j) == ctrlDeleted {
					continue
				}
				return g.key(j)
			}
		}
	}

	return nil
}

// Returns nil if the map is small.
func (m *Map) TableFor(key unsafe.Pointer) *table {
	if m.dirLen <= 0 {
		return nil
	}

	hash := m.typ.Hasher(key, m.seed)
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

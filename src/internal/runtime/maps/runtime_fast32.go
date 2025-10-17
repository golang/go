// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps

import (
	"internal/abi"
	"internal/race"
	"internal/runtime/sys"
	"unsafe"
)

//go:linkname runtime_mapaccess1_fast32 runtime.mapaccess1_fast32
func runtime_mapaccess1_fast32(typ *abi.MapType, m *Map, key uint32) unsafe.Pointer {
	if race.Enabled && m != nil {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapaccess1_fast32)
		race.ReadPC(unsafe.Pointer(m), callerpc, pc)
	}

	if m == nil || m.Used() == 0 {
		return unsafe.Pointer(&zeroVal[0])
	}

	if m.writing != 0 {
		fatal("concurrent map read and map write")
		return nil
	}

	if m.dirLen == 0 {
		g := groupReference{
			data: m.dirPtr,
		}
		full := g.ctrls().matchFull()
		slotKey := g.key(typ, 0)
		slotSize := typ.SlotSize
		for full != 0 {
			if key == *(*uint32)(slotKey) && full.lowestSet() {
				slotElem := unsafe.Pointer(uintptr(slotKey) + typ.ElemOff)
				return slotElem
			}
			slotKey = unsafe.Pointer(uintptr(slotKey) + slotSize)
			full = full.shiftOutLowest()
		}
		return unsafe.Pointer(&zeroVal[0])
	}

	k := key
	hash := typ.Hasher(abi.NoEscape(unsafe.Pointer(&k)), m.seed)

	// Select table.
	idx := m.directoryIndex(hash)
	t := m.directoryAt(idx)

	// Probe table.
	seq := makeProbeSeq(h1(hash), t.groups.lengthMask)
	h2Hash := h2(hash)
	for ; ; seq = seq.next() {
		g := t.groups.group(typ, seq.offset)

		match := g.ctrls().matchH2(h2Hash)

		for match != 0 {
			i := match.first()

			slotKey := g.key(typ, i)
			if key == *(*uint32)(slotKey) {
				slotElem := unsafe.Pointer(uintptr(slotKey) + typ.ElemOff)
				return slotElem
			}
			match = match.removeFirst()
		}

		match = g.ctrls().matchEmpty()
		if match != 0 {
			// Finding an empty slot means we've reached the end of
			// the probe sequence.
			return unsafe.Pointer(&zeroVal[0])
		}
	}
}

//go:linkname runtime_mapaccess2_fast32 runtime.mapaccess2_fast32
func runtime_mapaccess2_fast32(typ *abi.MapType, m *Map, key uint32) (unsafe.Pointer, bool) {
	if race.Enabled && m != nil {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapaccess2_fast32)
		race.ReadPC(unsafe.Pointer(m), callerpc, pc)
	}

	if m == nil || m.Used() == 0 {
		return unsafe.Pointer(&zeroVal[0]), false
	}

	if m.writing != 0 {
		fatal("concurrent map read and map write")
		return nil, false
	}

	if m.dirLen == 0 {
		g := groupReference{
			data: m.dirPtr,
		}
		full := g.ctrls().matchFull()
		slotKey := g.key(typ, 0)
		slotSize := typ.SlotSize
		for full != 0 {
			if key == *(*uint32)(slotKey) && full.lowestSet() {
				slotElem := unsafe.Pointer(uintptr(slotKey) + typ.ElemOff)
				return slotElem, true
			}
			slotKey = unsafe.Pointer(uintptr(slotKey) + slotSize)
			full = full.shiftOutLowest()
		}
		return unsafe.Pointer(&zeroVal[0]), false
	}

	k := key
	hash := typ.Hasher(abi.NoEscape(unsafe.Pointer(&k)), m.seed)

	// Select table.
	idx := m.directoryIndex(hash)
	t := m.directoryAt(idx)

	// Probe table.
	seq := makeProbeSeq(h1(hash), t.groups.lengthMask)
	h2Hash := h2(hash)
	for ; ; seq = seq.next() {
		g := t.groups.group(typ, seq.offset)

		match := g.ctrls().matchH2(h2Hash)

		for match != 0 {
			i := match.first()

			slotKey := g.key(typ, i)
			if key == *(*uint32)(slotKey) {
				slotElem := unsafe.Pointer(uintptr(slotKey) + typ.ElemOff)
				return slotElem, true
			}
			match = match.removeFirst()
		}

		match = g.ctrls().matchEmpty()
		if match != 0 {
			// Finding an empty slot means we've reached the end of
			// the probe sequence.
			return unsafe.Pointer(&zeroVal[0]), false
		}
	}
}

func (m *Map) putSlotSmallFast32(typ *abi.MapType, hash uintptr, key uint32) unsafe.Pointer {
	g := groupReference{
		data: m.dirPtr,
	}

	match := g.ctrls().matchH2(h2(hash))

	// Look for an existing slot containing this key.
	for match != 0 {
		i := match.first()

		slotKey := g.key(typ, i)
		if key == *(*uint32)(slotKey) {
			slotElem := g.elem(typ, i)
			return slotElem
		}
		match = match.removeFirst()
	}

	// There can't be deleted slots, small maps can't have them
	// (see deleteSmall). Use matchEmptyOrDeleted as it is a bit
	// more efficient than matchEmpty.
	match = g.ctrls().matchEmptyOrDeleted()
	if match == 0 {
		fatal("small map with no empty slot (concurrent map writes?)")
	}

	i := match.first()

	slotKey := g.key(typ, i)
	*(*uint32)(slotKey) = key

	slotElem := g.elem(typ, i)

	g.ctrls().set(i, ctrl(h2(hash)))
	m.used++

	return slotElem
}

//go:linkname runtime_mapassign_fast32 runtime.mapassign_fast32
func runtime_mapassign_fast32(typ *abi.MapType, m *Map, key uint32) unsafe.Pointer {
	if m == nil {
		panic(errNilAssign)
	}
	if race.Enabled {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapassign_fast32)
		race.WritePC(unsafe.Pointer(m), callerpc, pc)
	}
	if m.writing != 0 {
		fatal("concurrent map writes")
	}

	k := key
	hash := typ.Hasher(abi.NoEscape(unsafe.Pointer(&k)), m.seed)

	// Set writing after calling Hasher, since Hasher may panic, in which
	// case we have not actually done a write.
	m.writing ^= 1 // toggle, see comment on writing

	if m.dirPtr == nil {
		m.growToSmall(typ)
	}

	if m.dirLen == 0 {
		if m.used < abi.MapGroupSlots {
			elem := m.putSlotSmallFast32(typ, hash, key)

			if m.writing == 0 {
				fatal("concurrent map writes")
			}
			m.writing ^= 1

			return elem
		}

		// Can't fit another entry, grow to full size map.
		m.growToTable(typ)
	}

	var slotElem unsafe.Pointer
outer:
	for {
		// Select table.
		idx := m.directoryIndex(hash)
		t := m.directoryAt(idx)

		seq := makeProbeSeq(h1(hash), t.groups.lengthMask)

		// As we look for a match, keep track of the first deleted slot
		// we find, which we'll use to insert the new entry if
		// necessary.
		var firstDeletedGroup groupReference
		var firstDeletedSlot uintptr

		h2Hash := h2(hash)
		for ; ; seq = seq.next() {
			g := t.groups.group(typ, seq.offset)
			match := g.ctrls().matchH2(h2Hash)

			// Look for an existing slot containing this key.
			for match != 0 {
				i := match.first()

				slotKey := g.key(typ, i)
				if key == *(*uint32)(slotKey) {
					slotElem = g.elem(typ, i)

					t.checkInvariants(typ, m)
					break outer
				}
				match = match.removeFirst()
			}

			// No existing slot for this key in this group. Is this the end
			// of the probe sequence?
			match = g.ctrls().matchEmptyOrDeleted()
			if match == 0 {
				continue // nothing but filled slots. Keep probing.
			}
			i := match.first()
			if g.ctrls().get(i) == ctrlDeleted {
				// There are some deleted slots. Remember
				// the first one, and keep probing.
				if firstDeletedGroup.data == nil {
					firstDeletedGroup = g
					firstDeletedSlot = i
				}
				continue
			}
			// We've found an empty slot, which means we've reached the end of
			// the probe sequence.

			// If we found a deleted slot along the way, we can
			// replace it without consuming growthLeft.
			if firstDeletedGroup.data != nil {
				g = firstDeletedGroup
				i = firstDeletedSlot
				t.growthLeft++ // will be decremented below to become a no-op.
			}

			// If we have no space left, first try to remove some tombstones.
			if t.growthLeft == 0 {
				t.pruneTombstones(typ, m)
			}

			// If there is room left to grow, just insert the new entry.
			if t.growthLeft > 0 {
				slotKey := g.key(typ, i)
				*(*uint32)(slotKey) = key

				slotElem = g.elem(typ, i)

				g.ctrls().set(i, ctrl(h2Hash))
				t.growthLeft--
				t.used++
				m.used++

				t.checkInvariants(typ, m)
				break outer
			}

			t.rehash(typ, m)
			continue outer
		}
	}

	if m.writing == 0 {
		fatal("concurrent map writes")
	}
	m.writing ^= 1

	return slotElem
}

// Key is a 32-bit pointer (only called on 32-bit GOARCH). This source is identical to fast64ptr.
//
// TODO(prattmic): With some compiler refactoring we could avoid duplication of this function.
//
//go:linkname runtime_mapassign_fast32ptr runtime.mapassign_fast32ptr
func runtime_mapassign_fast32ptr(typ *abi.MapType, m *Map, key unsafe.Pointer) unsafe.Pointer {
	if m == nil {
		panic(errNilAssign)
	}
	if race.Enabled {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapassign_fast32ptr)
		race.WritePC(unsafe.Pointer(m), callerpc, pc)
	}
	if m.writing != 0 {
		fatal("concurrent map writes")
	}

	k := key
	hash := typ.Hasher(abi.NoEscape(unsafe.Pointer(&k)), m.seed)

	// Set writing after calling Hasher, since Hasher may panic, in which
	// case we have not actually done a write.
	m.writing ^= 1 // toggle, see comment on writing

	if m.dirPtr == nil {
		m.growToSmall(typ)
	}

	if m.dirLen == 0 {
		if m.used < abi.MapGroupSlots {
			elem := m.putSlotSmallFastPtr(typ, hash, key)

			if m.writing == 0 {
				fatal("concurrent map writes")
			}
			m.writing ^= 1

			return elem
		}

		// Can't fit another entry, grow to full size map.
		m.growToTable(typ)
	}

	var slotElem unsafe.Pointer
outer:
	for {
		// Select table.
		idx := m.directoryIndex(hash)
		t := m.directoryAt(idx)

		seq := makeProbeSeq(h1(hash), t.groups.lengthMask)

		// As we look for a match, keep track of the first deleted slot we
		// find, which we'll use to insert the new entry if necessary.
		var firstDeletedGroup groupReference
		var firstDeletedSlot uintptr

		h2Hash := h2(hash)
		for ; ; seq = seq.next() {
			g := t.groups.group(typ, seq.offset)
			match := g.ctrls().matchH2(h2Hash)

			// Look for an existing slot containing this key.
			for match != 0 {
				i := match.first()

				slotKey := g.key(typ, i)
				if key == *(*unsafe.Pointer)(slotKey) {
					slotElem = g.elem(typ, i)

					t.checkInvariants(typ, m)
					break outer
				}
				match = match.removeFirst()
			}

			// No existing slot for this key in this group. Is this the end
			// of the probe sequence?
			match = g.ctrls().matchEmptyOrDeleted()
			if match == 0 {
				continue // nothing but filled slots. Keep probing.
			}
			i := match.first()
			if g.ctrls().get(i) == ctrlDeleted {
				// There are some deleted slots. Remember
				// the first one, and keep probing.
				if firstDeletedGroup.data == nil {
					firstDeletedGroup = g
					firstDeletedSlot = i
				}
				continue
			}
			// We've found an empty slot, which means we've reached the end of
			// the probe sequence.

			// If we found a deleted slot along the way, we can
			// replace it without consuming growthLeft.
			if firstDeletedGroup.data != nil {
				g = firstDeletedGroup
				i = firstDeletedSlot
				t.growthLeft++ // will be decremented below to become a no-op.
			}

			// If there is room left to grow, just insert the new entry.
			if t.growthLeft > 0 {
				slotKey := g.key(typ, i)
				*(*unsafe.Pointer)(slotKey) = key

				slotElem = g.elem(typ, i)

				g.ctrls().set(i, ctrl(h2Hash))
				t.growthLeft--
				t.used++
				m.used++

				t.checkInvariants(typ, m)
				break outer
			}

			t.rehash(typ, m)
			continue outer
		}
	}

	if m.writing == 0 {
		fatal("concurrent map writes")
	}
	m.writing ^= 1

	return slotElem
}

//go:linkname runtime_mapdelete_fast32 runtime.mapdelete_fast32
func runtime_mapdelete_fast32(typ *abi.MapType, m *Map, key uint32) {
	if race.Enabled {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapdelete_fast32)
		race.WritePC(unsafe.Pointer(m), callerpc, pc)
	}

	if m == nil || m.Used() == 0 {
		return
	}

	m.Delete(typ, abi.NoEscape(unsafe.Pointer(&key)))
}

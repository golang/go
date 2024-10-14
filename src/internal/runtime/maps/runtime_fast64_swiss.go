// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.swissmap

package maps

import (
	"internal/abi"
	"internal/race"
	"internal/runtime/sys"
	"unsafe"
)

//go:linkname runtime_mapaccess1_fast64 runtime.mapaccess1_fast64
func runtime_mapaccess1_fast64(typ *abi.SwissMapType, m *Map, key uint64) unsafe.Pointer {
	if race.Enabled && m != nil {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapaccess1)
		race.ReadPC(unsafe.Pointer(m), callerpc, pc)
	}

	if m == nil || m.Used() == 0 {
		return unsafe.Pointer(&zeroVal[0])
	}

	if m.writing != 0 {
		fatal("concurrent map read and map write")
	}

	if m.dirLen == 0 {
		g := groupReference{
			data: m.dirPtr,
		}

		slotSize := typ.SlotSize
		for i, slotKey := uintptr(0), g.key(typ, 0); i < abi.SwissMapGroupSlots; i, slotKey = i+1, unsafe.Pointer(uintptr(slotKey)+slotSize) {
			if key == *(*uint64)(slotKey) && (g.ctrls().get(i)&(1<<7)) == 0 {
				slotElem := unsafe.Pointer(uintptr(slotKey) + typ.ElemOff)
				return slotElem
			}
		}
		return unsafe.Pointer(&zeroVal[0])
	}

	hash := typ.Hasher(abi.NoEscape(unsafe.Pointer(&key)), m.seed)

	// Select table.
	idx := m.directoryIndex(hash)
	t := m.directoryAt(idx)

	// Probe table.
	seq := makeProbeSeq(h1(hash), t.groups.lengthMask)
	for ; ; seq = seq.next() {
		g := t.groups.group(typ, seq.offset)

		match := g.ctrls().matchH2(h2(hash))

		for match != 0 {
			i := match.first()

			slotKey := g.key(typ, i)
			if key == *(*uint64)(slotKey) {
				slotElem := g.elem(typ, i)
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

//go:linkname runtime_mapaccess2_fast64 runtime.mapaccess2_fast64
func runtime_mapaccess2_fast64(typ *abi.SwissMapType, m *Map, key uint64) (unsafe.Pointer, bool) {
	if race.Enabled && m != nil {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapaccess1)
		race.ReadPC(unsafe.Pointer(m), callerpc, pc)
	}

	if m == nil || m.Used() == 0 {
		return unsafe.Pointer(&zeroVal[0]), false
	}

	if m.writing != 0 {
		fatal("concurrent map read and map write")
	}

	if m.dirLen == 0 {
		g := groupReference{
			data: m.dirPtr,
		}
		slotSize := typ.SlotSize
		for i, slotKey := uintptr(0), g.key(typ, 0); i < abi.SwissMapGroupSlots; i, slotKey = i+1, unsafe.Pointer(uintptr(slotKey)+slotSize) {
			if key == *(*uint64)(slotKey) && (g.ctrls().get(i)&(1<<7)) == 0 {
				slotElem := unsafe.Pointer(uintptr(slotKey) + typ.ElemOff)
				return slotElem, true
			}
		}
		return unsafe.Pointer(&zeroVal[0]), false
	}

	hash := typ.Hasher(abi.NoEscape(unsafe.Pointer(&key)), m.seed)

	// Select table.
	idx := m.directoryIndex(hash)
	t := m.directoryAt(idx)

	// Probe table.
	seq := makeProbeSeq(h1(hash), t.groups.lengthMask)
	for ; ; seq = seq.next() {
		g := t.groups.group(typ, seq.offset)

		match := g.ctrls().matchH2(h2(hash))

		for match != 0 {
			i := match.first()

			slotKey := g.key(typ, i)
			if key == *(*uint64)(slotKey) {
				slotElem := g.elem(typ, i)
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

func (m *Map) putSlotSmallFast64(typ *abi.SwissMapType, hash uintptr, key uint64) unsafe.Pointer {
	g := groupReference{
		data: m.dirPtr,
	}

	match := g.ctrls().matchH2(h2(hash))

	// Look for an existing slot containing this key.
	for match != 0 {
		i := match.first()

		slotKey := g.key(typ, i)
		if key == *(*uint64)(slotKey) {
			slotElem := g.elem(typ, i)
			return slotElem
		}
		match = match.removeFirst()
	}

	// No need to look for deleted slots, small maps can't have them (see
	// deleteSmall).
	match = g.ctrls().matchEmpty()
	if match == 0 {
		fatal("small map with no empty slot (concurrent map writes?)")
	}

	i := match.first()

	slotKey := g.key(typ, i)
	*(*uint64)(slotKey) = key

	slotElem := g.elem(typ, i)

	g.ctrls().set(i, ctrl(h2(hash)))
	m.used++

	return slotElem
}

//go:linkname runtime_mapassign_fast64 runtime.mapassign_fast64
func runtime_mapassign_fast64(typ *abi.SwissMapType, m *Map, key uint64) unsafe.Pointer {
	if m == nil {
		panic(errNilAssign)
	}
	if race.Enabled {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapassign)
		race.WritePC(unsafe.Pointer(m), callerpc, pc)
	}
	if m.writing != 0 {
		fatal("concurrent map writes")
	}

	hash := typ.Hasher(abi.NoEscape(unsafe.Pointer(&key)), m.seed)

	// Set writing after calling Hasher, since Hasher may panic, in which
	// case we have not actually done a write.
	m.writing ^= 1 // toggle, see comment on writing

	if m.dirPtr == nil {
		m.growToSmall(typ)
	}

	if m.dirLen == 0 {
		if m.used < abi.SwissMapGroupSlots {
			elem := m.putSlotSmallFast64(typ, hash, key)

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

		for ; ; seq = seq.next() {
			g := t.groups.group(typ, seq.offset)
			match := g.ctrls().matchH2(h2(hash))

			// Look for an existing slot containing this key.
			for match != 0 {
				i := match.first()

				slotKey := g.key(typ, i)
				if key == *(*uint64)(slotKey) {
					slotElem = g.elem(typ, i)

					t.checkInvariants(typ, m)
					break outer
				}
				match = match.removeFirst()
			}

			// No existing slot for this key in this group. Is this the end
			// of the probe sequence?
			match = g.ctrls().matchEmpty()
			if match != 0 {
				// Finding an empty slot means we've reached the end of
				// the probe sequence.

				var i uintptr

				// If we found a deleted slot along the way, we
				// can replace it without consuming growthLeft.
				if firstDeletedGroup.data != nil {
					g = firstDeletedGroup
					i = firstDeletedSlot
					t.growthLeft++ // will be decremented below to become a no-op.
				} else {
					// Otherwise, use the empty slot.
					i = match.first()
				}

				// If there is room left to grow, just insert the new entry.
				if t.growthLeft > 0 {
					slotKey := g.key(typ, i)
					*(*uint64)(slotKey) = key

					slotElem = g.elem(typ, i)

					g.ctrls().set(i, ctrl(h2(hash)))
					t.growthLeft--
					t.used++
					m.used++

					t.checkInvariants(typ, m)
					break outer
				}

				t.rehash(typ, m)
				continue outer
			}

			// No empty slots in this group. Check for a deleted
			// slot, which we'll use if we don't find a match later
			// in the probe sequence.
			//
			// We only need to remember a single deleted slot.
			if firstDeletedGroup.data == nil {
				// Since we already checked for empty slots
				// above, matches here must be deleted slots.
				match = g.ctrls().matchEmptyOrDeleted()
				if match != 0 {
					firstDeletedGroup = g
					firstDeletedSlot = match.first()
				}
			}
		}
	}

	if m.writing == 0 {
		fatal("concurrent map writes")
	}
	m.writing ^= 1

	return slotElem
}

func (m *Map) putSlotSmallFastPtr(typ *abi.SwissMapType, hash uintptr, key unsafe.Pointer) unsafe.Pointer {
	g := groupReference{
		data: m.dirPtr,
	}

	match := g.ctrls().matchH2(h2(hash))

	// Look for an existing slot containing this key.
	for match != 0 {
		i := match.first()

		slotKey := g.key(typ, i)
		if key == *(*unsafe.Pointer)(slotKey) {
			slotElem := g.elem(typ, i)
			return slotElem
		}
		match = match.removeFirst()
	}

	// No need to look for deleted slots, small maps can't have them (see
	// deleteSmall).
	match = g.ctrls().matchEmpty()
	if match == 0 {
		fatal("small map with no empty slot (concurrent map writes?)")
	}

	i := match.first()

	slotKey := g.key(typ, i)
	*(*unsafe.Pointer)(slotKey) = key

	slotElem := g.elem(typ, i)

	g.ctrls().set(i, ctrl(h2(hash)))
	m.used++

	return slotElem
}

// Key is a 64-bit pointer (only called on 64-bit GOARCH).
//
//go:linkname runtime_mapassign_fast64ptr runtime.mapassign_fast64ptr
func runtime_mapassign_fast64ptr(typ *abi.SwissMapType, m *Map, key unsafe.Pointer) unsafe.Pointer {
	if m == nil {
		panic(errNilAssign)
	}
	if race.Enabled {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapassign)
		race.WritePC(unsafe.Pointer(m), callerpc, pc)
	}
	if m.writing != 0 {
		fatal("concurrent map writes")
	}

	hash := typ.Hasher(abi.NoEscape(unsafe.Pointer(&key)), m.seed)

	// Set writing after calling Hasher, since Hasher may panic, in which
	// case we have not actually done a write.
	m.writing ^= 1 // toggle, see comment on writing

	if m.dirPtr == nil {
		m.growToSmall(typ)
	}

	if m.dirLen == 0 {
		if m.used < abi.SwissMapGroupSlots {
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

		// As we look for a match, keep track of the first deleted slot
		// we find, which we'll use to insert the new entry if
		// necessary.
		var firstDeletedGroup groupReference
		var firstDeletedSlot uintptr

		for ; ; seq = seq.next() {
			g := t.groups.group(typ, seq.offset)
			match := g.ctrls().matchH2(h2(hash))

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
			match = g.ctrls().matchEmpty()
			if match != 0 {
				// Finding an empty slot means we've reached the end of
				// the probe sequence.

				var i uintptr

				// If we found a deleted slot along the way, we
				// can replace it without consuming growthLeft.
				if firstDeletedGroup.data != nil {
					g = firstDeletedGroup
					i = firstDeletedSlot
					t.growthLeft++ // will be decremented below to become a no-op.
				} else {
					// Otherwise, use the empty slot.
					i = match.first()
				}

				// If there is room left to grow, just insert the new entry.
				if t.growthLeft > 0 {
					slotKey := g.key(typ, i)
					*(*unsafe.Pointer)(slotKey) = key

					slotElem = g.elem(typ, i)

					g.ctrls().set(i, ctrl(h2(hash)))
					t.growthLeft--
					t.used++
					m.used++

					t.checkInvariants(typ, m)
					break outer
				}

				t.rehash(typ, m)
				continue outer
			}

			// No empty slots in this group. Check for a deleted
			// slot, which we'll use if we don't find a match later
			// in the probe sequence.
			//
			// We only need to remember a single deleted slot.
			if firstDeletedGroup.data == nil {
				// Since we already checked for empty slots
				// above, matches here must be deleted slots.
				match = g.ctrls().matchEmptyOrDeleted()
				if match != 0 {
					firstDeletedGroup = g
					firstDeletedSlot = match.first()
				}
			}
		}
	}

	if m.writing == 0 {
		fatal("concurrent map writes")
	}
	m.writing ^= 1

	return slotElem
}

//go:linkname runtime_mapdelete_fast64 runtime.mapdelete_fast64
func runtime_mapdelete_fast64(typ *abi.SwissMapType, m *Map, key uint64) {
	if race.Enabled {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapassign)
		race.WritePC(unsafe.Pointer(m), callerpc, pc)
	}

	if m == nil || m.Used() == 0 {
		return
	}

	m.Delete(typ, abi.NoEscape(unsafe.Pointer(&key)))
}

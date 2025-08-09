// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps

import (
	"internal/abi"
	"internal/goarch"
	"internal/race"
	"internal/runtime/sys"
	"unsafe"
)

func (m *Map) getWithoutKeySmallFastStr(typ *abi.MapType, key string) unsafe.Pointer {
	g := groupReference{
		data: m.dirPtr,
	}

	ctrls := *g.ctrls()
	slotKey := g.key(typ, 0)
	slotSize := typ.SlotSize

	// The 64 threshold was chosen based on performance of BenchmarkMapStringKeysEight,
	// where there are 8 keys to check, all of which don't quick-match the lookup key.
	// In that case, we can save hashing the lookup key. That savings is worth this extra code
	// for strings that are long enough that hashing is expensive.
	if len(key) > 64 {
		// String hashing and equality might be expensive. Do a quick check first.
		j := abi.MapGroupSlots
		for i := range abi.MapGroupSlots {
			if ctrls&(1<<7) == 0 && longStringQuickEqualityTest(key, *(*string)(slotKey)) {
				if j < abi.MapGroupSlots {
					// 2 strings both passed the quick equality test.
					// Break out of this loop and do it the slow way.
					goto dohash
				}
				j = i
			}
			slotKey = unsafe.Pointer(uintptr(slotKey) + slotSize)
			ctrls >>= 8
		}
		if j == abi.MapGroupSlots {
			// No slot passed the quick test.
			return nil
		}
		// There's exactly one slot that passed the quick test. Do the single expensive comparison.
		slotKey = g.key(typ, uintptr(j))
		if key == *(*string)(slotKey) {
			return unsafe.Pointer(uintptr(slotKey) + 2*goarch.PtrSize)
		}
		return nil
	}

dohash:
	// This path will cost 1 hash and 1+ε comparisons.
	hash := typ.Hasher(abi.NoEscape(unsafe.Pointer(&key)), m.seed)
	h2 := uint8(h2(hash))
	ctrls = *g.ctrls()
	slotKey = g.key(typ, 0)

	for range abi.MapGroupSlots {
		if uint8(ctrls) == h2 && key == *(*string)(slotKey) {
			return unsafe.Pointer(uintptr(slotKey) + 2*goarch.PtrSize)
		}
		slotKey = unsafe.Pointer(uintptr(slotKey) + slotSize)
		ctrls >>= 8
	}
	return nil
}

// Returns true if a and b might be equal.
// Returns false if a and b are definitely not equal.
// Requires len(a)>=8.
func longStringQuickEqualityTest(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	x, y := stringPtr(a), stringPtr(b)
	// Check first 8 bytes.
	if *(*[8]byte)(x) != *(*[8]byte)(y) {
		return false
	}
	// Check last 8 bytes.
	x = unsafe.Pointer(uintptr(x) + uintptr(len(a)) - 8)
	y = unsafe.Pointer(uintptr(y) + uintptr(len(a)) - 8)
	if *(*[8]byte)(x) != *(*[8]byte)(y) {
		return false
	}
	return true
}
func stringPtr(s string) unsafe.Pointer {
	type stringStruct struct {
		ptr unsafe.Pointer
		len int
	}
	return (*stringStruct)(unsafe.Pointer(&s)).ptr
}

//go:linkname runtime_mapaccess1_faststr runtime.mapaccess1_faststr
func runtime_mapaccess1_faststr(typ *abi.MapType, m *Map, key string) unsafe.Pointer {
	if race.Enabled && m != nil {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapaccess1_faststr)
		race.ReadPC(unsafe.Pointer(m), callerpc, pc)
	}

	if m == nil || m.Used() == 0 {
		return unsafe.Pointer(&zeroVal[0])
	}

	if m.writing != 0 {
		fatal("concurrent map read and map write")
		return nil
	}

	if m.dirLen <= 0 {
		elem := m.getWithoutKeySmallFastStr(typ, key)
		if elem == nil {
			return unsafe.Pointer(&zeroVal[0])
		}
		return elem
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
			if key == *(*string)(slotKey) {
				slotElem := unsafe.Pointer(uintptr(slotKey) + 2*goarch.PtrSize)
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

//go:linkname runtime_mapaccess2_faststr runtime.mapaccess2_faststr
func runtime_mapaccess2_faststr(typ *abi.MapType, m *Map, key string) (unsafe.Pointer, bool) {
	if race.Enabled && m != nil {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapaccess2_faststr)
		race.ReadPC(unsafe.Pointer(m), callerpc, pc)
	}

	if m == nil || m.Used() == 0 {
		return unsafe.Pointer(&zeroVal[0]), false
	}

	if m.writing != 0 {
		fatal("concurrent map read and map write")
		return nil, false
	}

	if m.dirLen <= 0 {
		elem := m.getWithoutKeySmallFastStr(typ, key)
		if elem == nil {
			return unsafe.Pointer(&zeroVal[0]), false
		}
		return elem, true
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
			if key == *(*string)(slotKey) {
				slotElem := unsafe.Pointer(uintptr(slotKey) + 2*goarch.PtrSize)
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

func (m *Map) putSlotSmallFastStr(typ *abi.MapType, hash uintptr, key string) unsafe.Pointer {
	g := groupReference{
		data: m.dirPtr,
	}

	match := g.ctrls().matchH2(h2(hash))

	// Look for an existing slot containing this key.
	for match != 0 {
		i := match.first()

		slotKey := g.key(typ, i)
		if key == *(*string)(slotKey) {
			// Key needs update, as the backing storage may differ.
			*(*string)(slotKey) = key
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
	*(*string)(slotKey) = key

	slotElem := g.elem(typ, i)

	g.ctrls().set(i, ctrl(h2(hash)))
	m.used++

	return slotElem
}

//go:linkname runtime_mapassign_faststr runtime.mapassign_faststr
func runtime_mapassign_faststr(typ *abi.MapType, m *Map, key string) unsafe.Pointer {
	if m == nil {
		panic(errNilAssign)
	}
	if race.Enabled {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapassign_faststr)
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
			elem := m.putSlotSmallFastStr(typ, hash, key)

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
				if key == *(*string)(slotKey) {
					// Key needs update, as the backing
					// storage may differ.
					*(*string)(slotKey) = key
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
				*(*string)(slotKey) = key

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

//go:linkname runtime_mapdelete_faststr runtime.mapdelete_faststr
func runtime_mapdelete_faststr(typ *abi.MapType, m *Map, key string) {
	if race.Enabled {
		callerpc := sys.GetCallerPC()
		pc := abi.FuncPCABIInternal(runtime_mapdelete_faststr)
		race.WritePC(unsafe.Pointer(m), callerpc, pc)
	}

	if m == nil || m.Used() == 0 {
		return
	}

	m.Delete(typ, abi.NoEscape(unsafe.Pointer(&key)))
}

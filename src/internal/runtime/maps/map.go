// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package maps implements Go's builtin map type.
package maps

import (
	"internal/abi"
	"internal/goarch"
	"internal/runtime/sys"
	"unsafe"
)

// This package contains the implementation of Go's builtin map type.
//
// The map design is based on Abseil's "Swiss Table" map design
// (https://abseil.io/about/design/swisstables), with additional modifications
// to cover Go's additional requirements, discussed below.
//
// Terminology:
// - Slot: A storage location of a single key/element pair.
// - Group: A group of abi.SwissMapGroupSlots (8) slots, plus a control word.
// - Control word: An 8-byte word which denotes whether each slot is empty,
//   deleted, or used. If a slot is used, its control byte also contains the
//   lower 7 bits of the hash (H2).
// - H1: Upper 57 bits of a hash.
// - H2: Lower 7 bits of a hash.
// - Table: A complete "Swiss Table" hash table. A table consists of one or
//   more groups for storage plus metadata to handle operation and determining
//   when to grow.
// - Map: The top-level Map type consists of zero or more tables for storage.
//   The upper bits of the hash select which table a key belongs to.
// - Directory: Array of the tables used by the map.
//
// At its core, the table design is similar to a traditional open-addressed
// hash table. Storage consists of an array of groups, which effectively means
// an array of key/elem slots with some control words interspersed. Lookup uses
// the hash to determine an initial group to check. If, due to collisions, this
// group contains no match, the probe sequence selects the next group to check
// (see below for more detail about the probe sequence).
//
// The key difference occurs within a group. In a standard open-addressed
// linear probed hash table, we would check each slot one at a time to find a
// match. A swiss table utilizes the extra control word to check all 8 slots in
// parallel.
//
// Each byte in the control word corresponds to one of the slots in the group.
// In each byte, 1 bit is used to indicate whether the slot is in use, or if it
// is empty/deleted. The other 7 bits contain the lower 7 bits of the hash for
// the key in that slot. See [ctrl] for the exact encoding.
//
// During lookup, we can use some clever bitwise manipulation to compare all 8
// 7-bit hashes against the input hash in parallel (see [ctrlGroup.matchH2]).
// That is, we effectively perform 8 steps of probing in a single operation.
// With SIMD instructions, this could be extended to 16 slots with a 16-byte
// control word.
//
// Since we only use 7 bits of the 64 bit hash, there is a 1 in 128 (~0.7%)
// probability of false positive on each slot, but that's fine: we always need
// double check each match with a standard key comparison regardless.
//
// Probing
//
// Probing is done using the upper 57 bits (H1) of the hash as an index into
// the groups array. Probing walks through the groups using quadratic probing
// until it finds a group with a match or a group with an empty slot. See
// [probeSeq] for specifics about the probe sequence. Note the probe
// invariants: the number of groups must be a power of two, and the end of a
// probe sequence must be a group with an empty slot (the table can never be
// 100% full).
//
// Deletion
//
// Probing stops when it finds a group with an empty slot. This affects
// deletion: when deleting from a completely full group, we must not mark the
// slot as empty, as there could be more slots used later in a probe sequence
// and this deletion would cause probing to stop too early. Instead, we mark
// such slots as "deleted" with a tombstone. If the group still has an empty
// slot, we don't need a tombstone and directly mark the slot empty. Currently,
// tombstone are only cleared during grow, as an in-place cleanup complicates
// iteration.
//
// Growth
//
// The probe sequence depends on the number of groups. Thus, when growing the
// group count all slots must be reordered to match the new probe sequence. In
// other words, an entire table must be grown at once.
//
// In order to support incremental growth, the map splits its contents across
// multiple tables. Each table is still a full hash table, but an individual
// table may only service a subset of the hash space. Growth occurs on
// individual tables, so while an entire table must grow at once, each of these
// grows is only a small portion of a map. The maximum size of a single grow is
// limited by limiting the maximum size of a table before it is split into
// multiple tables.
//
// A map starts with a single table. Up to [maxTableCapacity], growth simply
// replaces this table with a replacement with double capacity. Beyond this
// limit, growth splits the table into two.
//
// The map uses "extendible hashing" to select which table to use. In
// extendible hashing, we use the upper bits of the hash as an index into an
// array of tables (called the "directory"). The number of bits uses increases
// as the number of tables increases. For example, when there is only 1 table,
// we use 0 bits (no selection necessary). When there are 2 tables, we use 1
// bit to select either the 0th or 1st table. [Map.globalDepth] is the number
// of bits currently used for table selection, and by extension (1 <<
// globalDepth), the size of the directory.
//
// Note that each table has its own load factor and grows independently. If the
// 1st bucket grows, it will split. We'll need 2 bits to select tables, though
// we'll have 3 tables total rather than 4. We support this by allowing
// multiple indicies to point to the same table. This example:
//
//	directory (globalDepth=2)
//	+----+
//	| 00 | --\
//	+----+    +--> table (localDepth=1)
//	| 01 | --/
//	+----+
//	| 10 | ------> table (localDepth=2)
//	+----+
//	| 11 | ------> table (localDepth=2)
//	+----+
//
// Tables track the depth they were created at (localDepth). It is necessary to
// grow the directory when splitting a table where globalDepth == localDepth.
//
// Iteration
//
// Iteration is the most complex part of the map due to Go's generous iteration
// semantics. A summary of semantics from the spec:
// 1. Adding and/or deleting entries during iteration MUST NOT cause iteration
//    to return the same entry more than once.
// 2. Entries added during iteration MAY be returned by iteration.
// 3. Entries modified during iteration MUST return their latest value.
// 4. Entries deleted during iteration MUST NOT be returned by iteration.
// 5. Iteration order is unspecified. In the implementation, it is explicitly
//    randomized.
//
// If the map never grows, these semantics are straightforward: just iterate
// over every table in the directory and every group and slot in each table.
// These semantics all land as expected.
//
// If the map grows during iteration, things complicate significantly. First
// and foremost, we need to track which entries we already returned to satisfy
// (1). There are three types of grow:
// a. A table replaced by a single larger table.
// b. A table split into two replacement tables.
// c. Growing the directory (occurs as part of (b) if necessary).
//
// For all of these cases, the replacement table(s) will have a different probe
// sequence, so simply tracking the current group and slot indices is not
// sufficient.
//
// For (a) and (b), note that grows of tables other than the one we are
// currently iterating over are irrelevant.
//
// We handle (a) and (b) by having the iterator keep a reference to the table
// it is currently iterating over, even after the table is replaced. We keep
// iterating over the original table to maintain the iteration order and avoid
// violating (1). Any new entries added only to the replacement table(s) will
// be skipped (allowed by (2)). To avoid violating (3) or (4), while we use the
// original table to select the keys, we must look them up again in the new
// table(s) to determine if they have been modified or deleted. There is yet
// another layer of complexity if the key does not compare equal itself. See
// [Iter.Next] for the gory details.
//
// Note that for (b) once we finish iterating over the old table we'll need to
// skip the next entry in the directory, as that contains the second split of
// the old table. We can use the old table's localDepth to determine the next
// logical index to use.
//
// For (b), we must adjust the current directory index when the directory
// grows. This is more straightforward, as the directory orders remains the
// same after grow, so we just double the index if the directory size doubles.

// Extracts the H1 portion of a hash: the 57 upper bits.
// TODO(prattmic): what about 32-bit systems?
func h1(h uintptr) uintptr {
	return h >> 7
}

// Extracts the H2 portion of a hash: the 7 bits not used for h1.
//
// These are used as an occupied control byte.
func h2(h uintptr) uintptr {
	return h & 0x7f
}

type Map struct {
	// The number of filled slots (i.e. the number of elements in all
	// tables).
	used uint64

	// Type of this map.
	//
	// TODO(prattmic): Old maps pass this into every call instead of
	// keeping a reference in the map header. This is probably more
	// efficient and arguably more robust (crafty users can't reach into to
	// the map to change its type), but I leave it here for now for
	// simplicity.
	typ *abi.SwissMapType

	// seed is the hash seed, computed as a unique random number per map.
	// TODO(prattmic): Populate this on table initialization.
	seed uintptr

	// The directory of tables. The length of this slice is
	// `1 << globalDepth`. Multiple entries may point to the same table.
	// See top-level comment for more details.
	directory []*table

	// The number of bits to use in table directory lookups.
	globalDepth uint8

	// clearSeq is a sequence counter of calls to Clear. It is used to
	// detect map clears during iteration.
	clearSeq uint64
}

func NewMap(mt *abi.SwissMapType, capacity uint64) *Map {
	if capacity < abi.SwissMapGroupSlots {
		// TODO: temporary to simplify initial implementation.
		capacity = abi.SwissMapGroupSlots
	}
	dirSize := (capacity + maxTableCapacity - 1) / maxTableCapacity
	dirSize, overflow := alignUpPow2(dirSize)
	if overflow {
		panic("rounded-up capacity overflows uint64")
	}
	globalDepth := uint8(sys.TrailingZeros64(dirSize))

	m := &Map{
		typ: mt,

		//TODO
		//seed: uintptr(rand()),

		directory: make([]*table, dirSize),

		globalDepth: globalDepth,
	}

	for i := range m.directory {
		// TODO: Think more about initial table capacity.
		m.directory[i] = newTable(mt, capacity/dirSize, i, globalDepth)
	}

	return m
}

func (m *Map) Type() *abi.SwissMapType {
	return m.typ
}

func (m *Map) directoryIndex(hash uintptr) uintptr {
	// TODO(prattmic): Store the shift as globalShift, as we need that more
	// often than globalDepth.
	if goarch.PtrSize == 4 {
		return hash >> (32 - m.globalDepth)
	}
	return hash >> (64 - m.globalDepth)
}

func (m *Map) replaceTable(nt *table) {
	// The number of entries that reference the same table doubles for each
	// time the globalDepth grows without the table splitting.
	entries := 1 << (m.globalDepth - nt.localDepth)
	for i := 0; i < entries; i++ {
		m.directory[nt.index+i] = nt
	}
}

func (m *Map) installTableSplit(old, left, right *table) {
	if old.localDepth == m.globalDepth {
		// No room for another level in the directory. Grow the
		// directory.
		newDir := make([]*table, len(m.directory)*2)
		for i, t := range m.directory {
			newDir[2*i] = t
			newDir[2*i+1] = t
			// t may already exist in multiple indicies. We should
			// only update t.index once. Since the index must
			// increase, seeing the original index means this must
			// be the first time we've encountered this table.
			if t.index == i {
				t.index = 2 * i
			}
		}
		m.globalDepth++
		m.directory = newDir
	}

	// N.B. left and right may still consume multiple indicies if the
	// directory has grown multiple times since old was last split.
	left.index = old.index
	m.replaceTable(left)

	entries := 1 << (m.globalDepth - left.localDepth)
	right.index = left.index + entries
	m.replaceTable(right)
}

func (m *Map) Used() uint64 {
	return m.used
}

// Get performs a lookup of the key that key points to. It returns a pointer to
// the element, or false if the key doesn't exist.
func (m *Map) Get(key unsafe.Pointer) (unsafe.Pointer, bool) {
	_, elem, ok := m.getWithKey(key)
	return elem, ok
}

func (m *Map) getWithKey(key unsafe.Pointer) (unsafe.Pointer, unsafe.Pointer, bool) {
	hash := m.typ.Hasher(key, m.seed)

	idx := m.directoryIndex(hash)
	return m.directory[idx].getWithKey(hash, key)
}

func (m *Map) Put(key, elem unsafe.Pointer) {
	slotElem := m.PutSlot(key)
	typedmemmove(m.typ.Elem, slotElem, elem)
}

// PutSlot returns a pointer to the element slot where an inserted element
// should be written.
//
// PutSlot never returns nil.
func (m *Map) PutSlot(key unsafe.Pointer) unsafe.Pointer {
	hash := m.typ.Hasher(key, m.seed)

	for {
		idx := m.directoryIndex(hash)
		elem, ok := m.directory[idx].PutSlot(m, hash, key)
		if !ok {
			continue
		}
		return elem
	}
}

func (m *Map) Delete(key unsafe.Pointer) {
	hash := m.typ.Hasher(key, m.seed)

	idx := m.directoryIndex(hash)
	m.directory[idx].Delete(m, key)
}

// Clear deletes all entries from the map resulting in an empty map.
func (m *Map) Clear() {
	var lastTab *table
	for _, t := range m.directory {
		if t == lastTab {
			continue
		}
		t.Clear()
		lastTab = t
	}
	m.used = 0
	m.clearSeq++
	// TODO: shrink directory?
}

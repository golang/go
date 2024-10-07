// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package maps implements Go's builtin map type.
package maps

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

// Maximum size of a table before it is split at the directory level.
//
// TODO: Completely made up value. This should be tuned for performance vs grow
// latency.
// TODO: This should likely be based on byte size, as copying costs will
// dominate grow latency for large objects.
const maxTableCapacity = 1024

// Ensure the max capacity fits in uint16, used for capacity and growthLeft
// below.
var _ = uint16(maxTableCapacity)

// table is a Swiss table hash table structure.
//
// Each table is a complete hash table implementation.
//
// Map uses one or more tables to store entries. Extendible hashing (hash
// prefix) is used to select the table to use for a specific key. Using
// multiple tables enables incremental growth by growing only one table at a
// time.
type table struct {
	// The number of filled slots (i.e. the number of elements in the table).
	used uint16

	// The total number of slots (always 2^N). Equal to
	// `(groups.lengthMask+1)*abi.SwissMapGroupSlots`.
	capacity uint16

	// The number of slots we can still fill without needing to rehash.
	//
	// We rehash when used + tombstones > loadFactor*capacity, including
	// tombstones so the table doesn't overfill with tombstones. This field
	// counts down remaining empty slots before the next rehash.
	growthLeft uint16

	// The number of bits used by directory lookups above this table. Note
	// that this may be less then globalDepth, if the directory has grown
	// but this table has not yet been split.
	localDepth uint8

	// TODO(prattmic): Old maps pass this into every call instead of
	// keeping a reference in the map header. This is probably more
	// efficient and arguably more robust (crafty users can't reach into to
	// the map to change its type), but I leave it here for now for
	// simplicity.
	typ *abi.SwissMapType

	// seed is the hash seed, computed as a unique random number per table.
	// TODO(prattmic): Populate this on table initialization.
	seed uintptr

	// Index of this table in the Map directory. This is the index of the
	// _first_ location in the directory. The table may occur in multiple
	// sequential indicies.
	index int

	// groups is an array of slot groups. Each group holds abi.SwissMapGroupSlots
	// key/elem slots and their control bytes. A table has a fixed size
	// groups array. The table is replaced (in rehash) when more space is
	// required.
	//
	// TODO(prattmic): keys and elements are interleaved to maximize
	// locality, but it comes at the expense of wasted space for some types
	// (consider uint8 key, uint64 element). Consider placing all keys
	// together in these cases to save space.
	//
	// TODO(prattmic): Support indirect keys/values? This means storing
	// keys/values as pointers rather than inline in the slot. This avoid
	// bloating the table size if either type is very large.
	groups groupsReference
}

func newTable(mt *abi.SwissMapType, capacity uint64, index int, localDepth uint8) *table {
	if capacity < abi.SwissMapGroupSlots {
		// TODO: temporary until we have a real map type.
		capacity = abi.SwissMapGroupSlots
	}

	t := &table{
		typ: mt,

		index:      index,
		localDepth: localDepth,
	}

	if capacity > maxTableCapacity {
		panic("initial table capacity too large")
	}

	// N.B. group count must be a power of two for probeSeq to visit every
	// group.
	capacity, overflow := alignUpPow2(capacity)
	if overflow {
		panic("rounded-up capacity overflows uint64")
	}

	t.reset(uint16(capacity))

	return t
}

// reset resets the table with new, empty groups with the specified new total
// capacity.
func (t *table) reset(capacity uint16) {
	groupCount := uint64(capacity) / abi.SwissMapGroupSlots
	t.groups = newGroups(t.typ, groupCount)
	t.capacity = capacity
	t.resetGrowthLeft()

	for i := uint64(0); i <= t.groups.lengthMask; i++ {
		g := t.groups.group(i)
		g.ctrls().setEmpty()
	}
}

// Preconditions: table must be empty.
func (t *table) resetGrowthLeft() {
	var growthLeft uint16
	if t.capacity == 0 {
		// No real reason to support zero capacity table, since an
		// empty Map simply won't have a table.
		panic("table must have positive capacity")
	} else if t.capacity <= abi.SwissMapGroupSlots {
		// If the map fits in a single group then we're able to fill all of
		// the slots except 1 (an empty slot is needed to terminate find
		// operations).
		//
		// TODO(go.dev/issue/54766): With a special case in probing for
		// single-group tables, we could fill all slots.
		growthLeft = t.capacity - 1
	} else {
		if t.capacity*maxAvgGroupLoad < t.capacity {
			// TODO(prattmic): Do something cleaner.
			panic("overflow")
		}
		growthLeft = (t.capacity * maxAvgGroupLoad) / abi.SwissMapGroupSlots
	}
	t.growthLeft = growthLeft
}

func (t *table) Used() uint64 {
	return uint64(t.used)
}

// Get performs a lookup of the key that key points to. It returns a pointer to
// the element, or false if the key doesn't exist.
func (t *table) Get(key unsafe.Pointer) (unsafe.Pointer, bool) {
	// TODO(prattmic): We could avoid hashing in a variety of special
	// cases.
	//
	// - One group maps with simple keys could iterate over all keys and
	//   compare them directly.
	// - One entry maps could just directly compare the single entry
	//   without hashing.
	// - String keys could do quick checks of a few bytes before hashing.
	hash := t.typ.Hasher(key, t.seed)
	_, elem, ok := t.getWithKey(hash, key)
	return elem, ok
}

// getWithKey performs a lookup of key, returning a pointer to the version of
// the key in the map in addition to the element.
//
// This is relevant when multiple different key values compare equal (e.g.,
// +0.0 and -0.0). When a grow occurs during iteration, iteration perform a
// lookup of keys from the old group in the new group in order to correctly
// expose updated elements. For NeedsKeyUpdate keys, iteration also must return
// the new key value, not the old key value.
// hash must be the hash of the key.
func (t *table) getWithKey(hash uintptr, key unsafe.Pointer) (unsafe.Pointer, unsafe.Pointer, bool) {
	// To find the location of a key in the table, we compute hash(key). From
	// h1(hash(key)) and the capacity, we construct a probeSeq that visits
	// every group of slots in some interesting order. See [probeSeq].
	//
	// We walk through these indices. At each index, we select the entire
	// group starting with that index and extract potential candidates:
	// occupied slots with a control byte equal to h2(hash(key)). The key
	// at candidate slot i is compared with key; if key == g.slot(i).key
	// we are done and return the slot; if there is an empty slot in the
	// group, we stop and return an error; otherwise we continue to the
	// next probe index. Tombstones (ctrlDeleted) effectively behave like
	// full slots that never match the value we're looking for.
	//
	// The h2 bits ensure when we compare a key we are likely to have
	// actually found the object. That is, the chance is low that keys
	// compare false. Thus, when we search for an object, we are unlikely
	// to call Equal many times. This likelihood can be analyzed as follows
	// (assuming that h2 is a random enough hash function).
	//
	// Let's assume that there are k "wrong" objects that must be examined
	// in a probe sequence. For example, when doing a find on an object
	// that is in the table, k is the number of objects between the start
	// of the probe sequence and the final found object (not including the
	// final found object). The expected number of objects with an h2 match
	// is then k/128. Measurements and analysis indicate that even at high
	// load factors, k is less than 32, meaning that the number of false
	// positive comparisons we must perform is less than 1/8 per find.
	seq := makeProbeSeq(h1(hash), t.groups.lengthMask)
	for ; ; seq = seq.next() {
		g := t.groups.group(seq.offset)

		match := g.ctrls().matchH2(h2(hash))

		for match != 0 {
			i := match.first()

			slotKey := g.key(i)
			if t.typ.Key.Equal(key, slotKey) {
				return slotKey, g.elem(i), true
			}
			match = match.removeFirst()
		}

		match = g.ctrls().matchEmpty()
		if match != 0 {
			// Finding an empty slot means we've reached the end of
			// the probe sequence.
			return nil, nil, false
		}
	}
}

// PutSlot returns a pointer to the element slot where an inserted element
// should be written, and ok if it returned a valid slot.
//
// PutSlot returns ok false if the table was split and the Map needs to find
// the new table.
//
// hash must be the hash of key.
func (t *table) PutSlot(m *Map, hash uintptr, key unsafe.Pointer) (unsafe.Pointer, bool) {
	seq := makeProbeSeq(h1(hash), t.groups.lengthMask)

	for ; ; seq = seq.next() {
		g := t.groups.group(seq.offset)
		match := g.ctrls().matchH2(h2(hash))

		// Look for an existing slot containing this key.
		for match != 0 {
			i := match.first()

			slotKey := g.key(i)
			if t.typ.Key.Equal(key, slotKey) {
				if t.typ.NeedKeyUpdate() {
					typedmemmove(t.typ.Key, slotKey, key)
				}

				slotElem := g.elem(i)

				t.checkInvariants()
				return slotElem, true
			}
			match = match.removeFirst()
		}

		match = g.ctrls().matchEmpty()
		if match != 0 {
			// Finding an empty slot means we've reached the end of
			// the probe sequence.

			// If there is room left to grow, just insert the new entry.
			if t.growthLeft > 0 {
				i := match.first()

				slotKey := g.key(i)
				typedmemmove(t.typ.Key, slotKey, key)
				slotElem := g.elem(i)

				g.ctrls().set(i, ctrl(h2(hash)))
				t.growthLeft--
				t.used++
				m.used++

				t.checkInvariants()
				return slotElem, true
			}

			// TODO(prattmic): While searching the probe sequence,
			// we may have passed deleted slots which we could use
			// for this entry.
			//
			// At the moment, we leave this behind for
			// rehash to free up.
			//
			// cockroachlabs/swiss restarts search of the probe
			// sequence for a deleted slot.
			//
			// TODO(go.dev/issue/54766): We want this optimization
			// back. We could search for the first deleted slot
			// during the main search, but only use it if we don't
			// find an existing entry.

			t.rehash(m)
			return nil, false
		}
	}
}

// uncheckedPutSlot inserts an entry known not to be in the table, returning an
// entry to the element slot where the element should be written. Used by
// PutSlot after it has failed to find an existing entry to overwrite duration
// insertion.
//
// Updates growthLeft if necessary, but does not update used.
//
// Requires that the entry does not exist in the table, and that the table has
// room for another element without rehashing.
//
// Never returns nil.
func (t *table) uncheckedPutSlot(hash uintptr, key unsafe.Pointer) unsafe.Pointer {
	if t.growthLeft == 0 {
		panic("invariant failed: growthLeft is unexpectedly 0")
	}

	// Given key and its hash hash(key), to insert it, we construct a
	// probeSeq, and use it to find the first group with an unoccupied (empty
	// or deleted) slot. We place the key/value into the first such slot in
	// the group and mark it as full with key's H2.
	seq := makeProbeSeq(h1(hash), t.groups.lengthMask)
	for ; ; seq = seq.next() {
		g := t.groups.group(seq.offset)

		match := g.ctrls().matchEmpty()
		if match != 0 {
			i := match.first()

			slotKey := g.key(i)
			typedmemmove(t.typ.Key, slotKey, key)
			slotElem := g.elem(i)

			if g.ctrls().get(i) == ctrlEmpty {
				t.growthLeft--
			}
			g.ctrls().set(i, ctrl(h2(hash)))
			return slotElem
		}
	}
}

func (t *table) Delete(m *Map, key unsafe.Pointer) {
	hash := t.typ.Hasher(key, t.seed)

	seq := makeProbeSeq(h1(hash), t.groups.lengthMask)
	for ; ; seq = seq.next() {
		g := t.groups.group(seq.offset)
		match := g.ctrls().matchH2(h2(hash))

		for match != 0 {
			i := match.first()
			slotKey := g.key(i)
			if t.typ.Key.Equal(key, slotKey) {
				t.used--
				m.used--

				typedmemclr(t.typ.Key, slotKey)
				typedmemclr(t.typ.Elem, g.elem(i))

				// Only a full group can appear in the middle
				// of a probe sequence (a group with at least
				// one empty slot terminates probing). Once a
				// group becomes full, it stays full until
				// rehashing/resizing. So if the group isn't
				// full now, we can simply remove the element.
				// Otherwise, we create a tombstone to mark the
				// slot as deleted.
				if g.ctrls().matchEmpty() != 0 {
					g.ctrls().set(i, ctrlEmpty)
					t.growthLeft++
				} else {
					g.ctrls().set(i, ctrlDeleted)
				}

				t.checkInvariants()
				return
			}
			match = match.removeFirst()
		}

		match = g.ctrls().matchEmpty()
		if match != 0 {
			// Finding an empty slot means we've reached the end of
			// the probe sequence.
			return
		}
	}
}

// tombstones returns the number of deleted (tombstone) entries in the table. A
// tombstone is a slot that has been deleted but is still considered occupied
// so as not to violate the probing invariant.
func (t *table) tombstones() uint16 {
	return (t.capacity*maxAvgGroupLoad)/abi.SwissMapGroupSlots - t.used - t.growthLeft
}

// Clear deletes all entries from the map resulting in an empty map.
func (t *table) Clear() {
	for i := uint64(0); i <= t.groups.lengthMask; i++ {
		g := t.groups.group(i)
		typedmemclr(t.typ.Group, g.data)
		g.ctrls().setEmpty()
	}

	t.used = 0
	t.resetGrowthLeft()

	// Reset the hash seed to make it more difficult for attackers to
	// repeatedly trigger hash collisions. See issue
	// https://github.com/golang/go/issues/25237.
	// TODO
	//t.seed = uintptr(rand())
}

type Iter struct {
	key  unsafe.Pointer // Must be in first position.  Write nil to indicate iteration end (see cmd/compile/internal/walk/range.go).
	elem unsafe.Pointer // Must be in second position (see cmd/compile/internal/walk/range.go).
	typ  *abi.SwissMapType
	m    *Map

	// Randomize iteration order by starting iteration at a random slot
	// offset. The offset into the directory uses a separate offset, as it
	// must adjust when the directory grows.
	entryOffset uint64
	dirOffset   uint64

	// Snapshot of Map.clearSeq at iteration initialization time. Used to
	// detect clear during iteration.
	clearSeq uint64

	// Value of Map.globalDepth during the last call to Next. Used to
	// detect directory grow during iteration.
	globalDepth uint8

	// dirIdx is the current directory index, prior to adjustment by
	// dirOffset.
	dirIdx int

	// tab is the table at dirIdx during the previous call to Next.
	tab *table

	// entryIdx is the current entry index, prior to adjustment by entryOffset.
	// The lower 3 bits of the index are the slot index, and the upper bits
	// are the group index.
	entryIdx uint64
}

// Init initializes Iter for iteration.
func (it *Iter) Init(typ *abi.SwissMapType, m *Map) {
	it.typ = typ
	if m == nil || m.used == 0 {
		return
	}

	it.typ = m.typ
	it.m = m
	it.entryOffset = rand()
	it.dirOffset = rand()
	it.globalDepth = m.globalDepth
	it.clearSeq = m.clearSeq
}

func (it *Iter) Initialized() bool {
	return it.typ != nil
}

// Map returns the map this iterator is iterating over.
func (it *Iter) Map() *Map {
	return it.m
}

// Key returns a pointer to the current key. nil indicates end of iteration.
//
// Must not be called prior to Next.
func (it *Iter) Key() unsafe.Pointer {
	return it.key
}

// Key returns a pointer to the current element. nil indicates end of
// iteration.
//
// Must not be called prior to Next.
func (it *Iter) Elem() unsafe.Pointer {
	return it.elem
}

// Next proceeds to the next element in iteration, which can be accessed via
// the Key and Elem methods.
//
// The table can be mutated during iteration, though there is no guarantee that
// the mutations will be visible to the iteration.
//
// Init must be called prior to Next.
func (it *Iter) Next() {
	if it.m == nil {
		// Map was empty at Iter.Init.
		it.key = nil
		it.elem = nil
		return
	}

	if it.globalDepth != it.m.globalDepth {
		// Directory has grown since the last call to Next. Adjust our
		// directory index.
		//
		// Consider:
		//
		// Before:
		// - 0: *t1
		// - 1: *t2  <- dirIdx
		//
		// After:
		// - 0: *t1a (split)
		// - 1: *t1b (split)
		// - 2: *t2  <- dirIdx
		// - 3: *t2
		//
		// That is, we want to double the current index when the
		// directory size doubles (or quadruple when the directory size
		// quadruples, etc).
		//
		// The actual (randomized) dirIdx is computed below as:
		//
		// dirIdx := (it.dirIdx + it.dirOffset) % it.m.dirLen
		//
		// Multiplication is associative across modulo operations,
		// A * (B % C) = (A * B) % (A * C),
		// provided that A is positive.
		//
		// Thus we can achieve this by adjusting it.dirIdx,
		// it.dirOffset, and it.m.dirLen individually.
		orders := it.m.globalDepth - it.globalDepth
		it.dirIdx <<= orders
		it.dirOffset <<= orders
		// it.m.dirLen was already adjusted when the directory grew.

		it.globalDepth = it.m.globalDepth
	}

	// Continue iteration until we find a full slot.
	for it.dirIdx < len(it.m.directory) {
		// TODO(prattmic): We currently look up the latest table on
		// every call, even if it.tab is set because the inner loop
		// checks if it.tab has grown by checking it.tab != newTab.
		//
		// We could avoid most of these lookups if we left a flag
		// behind on the old table to denote that it is stale.
		dirIdx := int((uint64(it.dirIdx) + it.dirOffset) % uint64(len(it.m.directory)))
		newTab := it.m.directory[dirIdx]
		if it.tab == nil {
			if newTab.index != dirIdx {
				// Normally we skip past all duplicates of the
				// same entry in the table (see updates to
				// it.dirIdx at the end of the loop below), so
				// this case wouldn't occur.
				//
				// But on the very first call, we have a
				// completely randomized dirIdx that may refer
				// to a middle of a run of tables in the
				// directory. Do a one-time adjustment of the
				// offset to ensure we start at first index for
				// newTable.
				diff := dirIdx - newTab.index
				it.dirOffset -= uint64(diff)
				dirIdx = newTab.index
			}
			it.tab = newTab
		}

		var g groupReference

		// N.B. Use it.tab, not newTab. It is important to use the old
		// table for key selection if the table has grown. See comment
		// on grown below.
		for ; it.entryIdx <= it.tab.groups.entryMask; it.entryIdx++ {
			entryIdx := (it.entryIdx + it.entryOffset) & it.tab.groups.entryMask
			slotIdx := uint32(entryIdx & (abi.SwissMapGroupSlots - 1))

			if slotIdx == 0 || g.data == nil {
				// Only compute the group (a) when we switch
				// groups (slotIdx rolls over) and (b) on the
				// first iteration in this table (slotIdx may
				// not be zero due to entryOffset).
				groupIdx := entryIdx >> abi.SwissMapGroupSlotsBits
				g = it.tab.groups.group(groupIdx)
			}

			// TODO(prattmic): Skip over groups that are composed of only empty
			// or deleted slots using matchEmptyOrDeleted() and counting the
			// number of bits set.

			if (g.ctrls().get(slotIdx) & ctrlEmpty) == ctrlEmpty {
				// Empty or deleted.
				continue
			}

			key := g.key(slotIdx)

			// If the table has changed since the last
			// call, then it has grown or split. In this
			// case, further mutations (changes to
			// key->elem or deletions) will not be visible
			// in our snapshot table. Instead we must
			// consult the new table by doing a full
			// lookup.
			//
			// We still use our old table to decide which
			// keys to lookup in order to avoid returning
			// the same key twice.
			grown := it.tab != newTab
			var elem unsafe.Pointer
			if grown {
				var ok bool
				newKey, newElem, ok := it.m.getWithKey(key)
				if !ok {
					// Key has likely been deleted, and
					// should be skipped.
					//
					// One exception is keys that don't
					// compare equal to themselves (e.g.,
					// NaN). These keys cannot be looked
					// up, so getWithKey will fail even if
					// the key exists.
					//
					// However, we are in luck because such
					// keys cannot be updated and they
					// cannot be deleted except with clear.
					// Thus if no clear has occurted, the
					// key/elem must still exist exactly as
					// in the old groups, so we can return
					// them from there.
					//
					// TODO(prattmic): Consider checking
					// clearSeq early. If a clear occurred,
					// Next could always return
					// immediately, as iteration doesn't
					// need to return anything added after
					// clear.
					if it.clearSeq == it.m.clearSeq && !it.m.typ.Key.Equal(key, key) {
						elem = g.elem(slotIdx)
					} else {
						continue
					}
				} else {
					key = newKey
					elem = newElem
				}
			} else {
				elem = g.elem(slotIdx)
			}

			it.entryIdx++
			it.key = key
			it.elem = elem
			return
		}

		// Skip other entries in the directory that refer to the same
		// logical table. There are two cases of this:
		//
		// Consider this directory:
		//
		// - 0: *t1
		// - 1: *t1
		// - 2: *t2a
		// - 3: *t2b
		//
		// At some point, the directory grew to accomodate a split of
		// t2. t1 did not split, so entries 0 and 1 both point to t1.
		// t2 did split, so the two halves were installed in entries 2
		// and 3.
		//
		// If dirIdx is 0 and it.tab is t1, then we should skip past
		// entry 1 to avoid repeating t1.
		//
		// If dirIdx is 2 and it.tab is t2 (pre-split), then we should
		// skip past entry 3 because our pre-split t2 already covers
		// all keys from t2a and t2b (except for new insertions, which
		// iteration need not return).
		//
		// We can achieve both of these by using to difference between
		// the directory and table depth to compute how many entries
		// the table covers.
		entries := 1 << (it.m.globalDepth - it.tab.localDepth)
		it.dirIdx += entries
		it.tab = nil
		it.entryIdx = 0
	}

	it.key = nil
	it.elem = nil
	return
}

// Replaces the table with one larger table or two split tables to fit more
// entries. Since the table is replaced, t is now stale and should not be
// modified.
func (t *table) rehash(m *Map) {
	// TODO(prattmic): SwissTables typically perform a "rehash in place"
	// operation which recovers capacity consumed by tombstones without growing
	// the table by reordering slots as necessary to maintain the probe
	// invariant while eliminating all tombstones.
	//
	// However, it is unclear how to make rehash in place work with
	// iteration. Since iteration simply walks through all slots in order
	// (with random start offset), reordering the slots would break
	// iteration.
	//
	// As an alternative, we could do a "resize" to new groups allocation
	// of the same size. This would eliminate the tombstones, but using a
	// new allocation, so the existing grow support in iteration would
	// continue to work.

	// TODO(prattmic): split table
	// TODO(prattmic): Avoid overflow (splitting the table will achieve this)

	newCapacity := 2 * t.capacity
	if newCapacity <= maxTableCapacity {
		t.grow(m, newCapacity)
		return
	}

	t.split(m)
}

// Bitmask for the last selection bit at this depth.
func localDepthMask(localDepth uint8) uintptr {
	if goarch.PtrSize == 4 {
		return uintptr(1) << (32 - localDepth)
	}
	return uintptr(1) << (64 - localDepth)
}

// split the table into two, installing the new tables in the map directory.
func (t *table) split(m *Map) {
	localDepth := t.localDepth
	localDepth++

	// TODO: is this the best capacity?
	left := newTable(t.typ, maxTableCapacity, -1, localDepth)
	right := newTable(t.typ, maxTableCapacity, -1, localDepth)

	// Split in half at the localDepth bit from the top.
	mask := localDepthMask(localDepth)

	for i := uint64(0); i <= t.groups.lengthMask; i++ {
		g := t.groups.group(i)
		for j := uint32(0); j < abi.SwissMapGroupSlots; j++ {
			if (g.ctrls().get(j) & ctrlEmpty) == ctrlEmpty {
				// Empty or deleted
				continue
			}
			key := g.key(j)
			elem := g.elem(j)
			hash := t.typ.Hasher(key, t.seed)
			var newTable *table
			if hash&mask == 0 {
				newTable = left
			} else {
				newTable = right
			}
			slotElem := newTable.uncheckedPutSlot(hash, key)
			typedmemmove(newTable.typ.Elem, slotElem, elem)
			newTable.used++
		}
	}

	m.installTableSplit(t, left, right)
}

// grow the capacity of the table by allocating a new table with a bigger array
// and uncheckedPutting each element of the table into the new table (we know
// that no insertion here will Put an already-present value), and discard the
// old table.
func (t *table) grow(m *Map, newCapacity uint16) {
	newTable := newTable(t.typ, uint64(newCapacity), t.index, t.localDepth)

	if t.capacity > 0 {
		for i := uint64(0); i <= t.groups.lengthMask; i++ {
			g := t.groups.group(i)
			for j := uint32(0); j < abi.SwissMapGroupSlots; j++ {
				if (g.ctrls().get(j) & ctrlEmpty) == ctrlEmpty {
					// Empty or deleted
					continue
				}
				key := g.key(j)
				elem := g.elem(j)
				hash := newTable.typ.Hasher(key, t.seed)
				slotElem := newTable.uncheckedPutSlot(hash, key)
				typedmemmove(newTable.typ.Elem, slotElem, elem)
				newTable.used++
			}
		}
	}

	newTable.checkInvariants()
	m.replaceTable(newTable)
}

// probeSeq maintains the state for a probe sequence that iterates through the
// groups in a table. The sequence is a triangular progression of the form
//
//	p(i) := (i^2 + i)/2 + hash (mod mask+1)
//
// The sequence effectively outputs the indexes of *groups*. The group
// machinery allows us to check an entire group with minimal branching.
//
// It turns out that this probe sequence visits every group exactly once if
// the number of groups is a power of two, since (i^2+i)/2 is a bijection in
// Z/(2^m). See https://en.wikipedia.org/wiki/Quadratic_probing
type probeSeq struct {
	mask   uint64
	offset uint64
	index  uint64
}

func makeProbeSeq(hash uintptr, mask uint64) probeSeq {
	return probeSeq{
		mask:   mask,
		offset: uint64(hash) & mask,
		index:  0,
	}
}

func (s probeSeq) next() probeSeq {
	s.index++
	s.offset = (s.offset + s.index) & s.mask
	return s
}

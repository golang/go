// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps_test

import (
	"internal/abi"
	"internal/runtime/maps"
	"math"
	"testing"
	"unsafe"
)

func TestMapShrink(t *testing.T) {
	m, typ := maps.NewTestMap[int, [16]uint64](0)
	m.Grow(typ, 896)
	groups := m.GroupCount()
	// Sparse churn should leave the explicit reservation available.
	for i := range 1024 {
		elem := [16]uint64{uint64(i)}
		m.Put(typ, unsafe.Pointer(&i), unsafe.Pointer(&elem))
		m.Delete(typ, unsafe.Pointer(&i))
	}
	if got := m.GroupCount(); got != groups {
		t.Fatalf("GroupCount() after sparse churn = %d, want %d", got, groups)
	}
	for i := 0; i < 512; i++ {
		elem := [16]uint64{uint64(i)}
		m.Put(typ, unsafe.Pointer(&i), unsafe.Pointer(&elem))
	}
	for i := 8; i < 512; i++ {
		m.Delete(typ, unsafe.Pointer(&i))
	}
	shrunk := m.GroupCount()
	if shrunk >= groups {
		t.Fatalf("GroupCount() after deletes = %d, want fewer than %d", shrunk, groups)
	}
	if got := m.Used(); got != 8 {
		t.Fatalf("Used() after deletes = %d, want 8", got)
	}
	for i := 0; i < 512; i++ {
		got, ok := m.Get(typ, unsafe.Pointer(&i))
		if i < 8 {
			if !ok || *(*[16]uint64)(got) != [16]uint64{uint64(i)} {
				t.Errorf("surviving entry %d missing or changed", i)
			}
		} else if ok {
			t.Errorf("deleted entry %d still present", i)
		}
	}

	m.Clear(typ)
	if got := m.GroupCount(); got != shrunk {
		t.Fatalf("GroupCount() after Clear = %d, want %d", got, shrunk)
	}
	for i := range 512 {
		elem := [16]uint64{uint64(i)}
		m.Put(typ, unsafe.Pointer(&i), unsafe.Pointer(&elem))
	}
	for i := 8; i < 512; i++ {
		m.Delete(typ, unsafe.Pointer(&i))
	}
	if got := m.GroupCount(); got != groups {
		t.Errorf("GroupCount() after refill and second drain = %d, want %d", got, groups)
	}
}

func TestMapGrowAfterShrink(t *testing.T) {
	for _, tc := range []struct {
		name      string
		mixed     bool
		wantSlots uint64
	}{
		{"all shrunk", false, 4 * 128},
		{"one full-size table", true, 1024 + 3*512},
	} {
		t.Run(tc.name, func(t *testing.T) {
			testMapGrowAfterShrink(t, tc.mixed, tc.wantSlots/abi.MapGroupSlots)
		})
	}
}

func testMapGrowAfterShrink(t *testing.T, mixed bool, wantGroups uint64) {
	type value [16]uint64
	const numTables = 4
	const entriesPerTable = 512
	const survivorsPerTable = 4
	m, mapType := maps.NewTestMap[uint32, value](numTables * entriesPerTable)
	typ := *mapType
	typ.Hasher = func(key unsafe.Pointer, seed uintptr) uintptr {
		k := uintptr(*(*uint32)(key))
		topShift := 30
		if maps.Use64BitHash {
			topShift = 62
		}
		return (k >> 10 << topShift) | ((k & 1023) << 7) | (k & 127)
	}
	put := func(tab, i int) {
		key := uint32(tab*1024 + i)
		elem := value{uint64(key)}
		m.Put(&typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))
	}
	for tab := range numTables {
		for i := range entriesPerTable {
			put(tab, i)
		}
		if mixed && tab == 0 {
			continue
		}
		for i := survivorsPerTable; i < entriesPerTable; i++ {
			key := uint32(tab*1024 + i)
			m.Delete(&typ, unsafe.Pointer(&key))
		}
	}

	// Reserve room for 100 more entries per table. The old directory
	// already covers the whole hash space with four ranges.
	const extraPerTable = 100
	m.Grow(&typ, numTables*extraPerTable)
	if got := m.GroupCount(); got != wantGroups {
		t.Fatalf("GroupCount() after Grow = %d, want %d", got, wantGroups)
	}
	for tab := range numTables {
		start := survivorsPerTable
		if mixed && tab == 0 {
			start = entriesPerTable
		}
		for i := start; i < start+extraPerTable; i++ {
			put(tab, i)
		}
	}
	if got := m.GroupCount(); got != wantGroups {
		t.Errorf("GroupCount() after refill = %d, want %d", got, wantGroups)
	}
}

// Exercise a shrink while iterating a table with multiple directory aliases.
func TestMapShrinkAliasedDuringIteration(t *testing.T) {
	type elem [16]uint64
	m, baseType := maps.NewTestMap[float64, elem](maps.MaxTableCapacity)
	typ := *baseType
	topMask := uintptr(1) << 31
	if maps.Use64BitHash {
		topMask <<= 32
	}

	const rightBase = 1 << 16
	const leftEntries = maps.MaxTableCapacity / 2
	const rightEntries = maps.MaxTableCapacity*maps.MaxAvgGroupLoad/abi.MapGroupSlots + 1
	const nanValue = 1 << 20

	// Select the table by key range. The right table will split on key
	// parity while the left table retains both of its directory aliases.
	nanHash := uintptr(3) << 7
	typ.Hasher = func(key unsafe.Pointer, seed uintptr) uintptr {
		k := *(*float64)(key)
		if math.IsNaN(k) {
			return nanHash
		}
		i := uintptr(k)
		hash := i<<7 | i&0x7f
		if i >= rightBase {
			hash |= topMask
			if i&1 != 0 {
				hash |= topMask >> 1
			}
		}
		return hash
	}

	put := func(k float64, v uint64) {
		value := elem{v}
		m.Put(&typ, unsafe.Pointer(&k), unsafe.Pointer(&value))
	}
	deleteKey := func(k float64) {
		m.Delete(&typ, unsafe.Pointer(&k))
	}
	for i := range leftEntries {
		put(float64(i), uint64(i))
	}
	put(math.NaN(), nanValue)
	for i := range rightEntries {
		put(rightBase+float64(i), uint64(rightBase+i))
	}

	if m.TableCount() != 4 || m.TableAt(0) != m.TableAt(1) {
		t.Fatal("test setup did not create a shared directory range")
	}
	old, right0, right1 := m.TableAt(0), m.TableAt(2), m.TableAt(3)

	var it maps.Iter
	it.Init(&typ, m)
	it.SetOffsets(0, 0)
	it.Next()
	if it.Key() == nil || *(*float64)(it.Key()) != 0 {
		t.Fatal("test setup did not begin iteration at key 0 in the shared table")
	}
	itClear := it

	// Rehashing a NaN can change its high bits. Resizing must keep it
	// in the same directory range, even with a changed hash.
	nanHash |= topMask
	for i := 7; i < leftEntries; i++ {
		deleteKey(float64(i))
	}
	cur := m.TableAt(0)
	if cur.GroupsLength() >= old.GroupsLength() {
		t.Fatalf("old table was not replaced by a smaller table in the same range: %d -> %d groups", old.GroupsLength(), cur.GroupsLength())
	}
	if m.TableCount() != 4 || m.TableAt(1) != cur || m.TableAt(2) != right0 || m.TableAt(3) != right1 {
		t.Fatal("shrink changed the directory ranges")
	}

	// The iterator still has the old table. Lookups in the new table
	// must reflect these changes to entries present in the snapshot.
	put(1, 99)
	deleteKey(2)

	checkRemaining := func(it *maps.Iter, sawZero bool) {
		t.Helper()
		seen := make(map[float64]bool)
		if sawZero {
			seen[0] = true
		}
		nanCount := 0
		for it.Next(); it.Key() != nil; it.Next() {
			k := *(*float64)(it.Key())
			v := *(*elem)(it.Elem())
			if math.IsNaN(k) {
				nanCount++
				if v != (elem{nanValue}) {
					t.Errorf("NaN value = %v, want %d", v, nanValue)
				}
				continue
			}
			if seen[k] {
				t.Errorf("duplicate key %v", k)
			}
			seen[k] = true
			if k == 2 || k >= 7 && k < rightBase {
				t.Errorf("returned deleted key %v", k)
			}
			want := uint64(k)
			if k == 1 {
				want = 99
			}
			if v != (elem{want}) {
				t.Errorf("value for key %v = %v, want %d", k, v, want)
			}
		}
		if nanCount != 1 || len(seen) != rightEntries+6 {
			t.Errorf("returned %d NaNs and %d ordinary keys, want 1 and %d", nanCount, len(seen), rightEntries+6)
		}
	}
	checkRemaining(&it, true)

	// A fresh iterator verifies that the NaN survived in the map itself.
	var fresh maps.Iter
	fresh.Init(&typ, m)
	checkRemaining(&fresh, false)

	m.Clear(&typ)
	itClear.Next()
	if itClear.Key() != nil {
		t.Error("iterator returned an entry after Clear")
	}
}

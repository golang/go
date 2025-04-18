// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps_test

import (
	"fmt"
	"internal/abi"
	"internal/runtime/maps"
	"math"
	"testing"
	"unsafe"
)

func TestCtrlSize(t *testing.T) {
	cs := unsafe.Sizeof(maps.CtrlGroup(0))
	if cs != abi.SwissMapGroupSlots {
		t.Errorf("ctrlGroup size got %d want abi.SwissMapGroupSlots %d", cs, abi.SwissMapGroupSlots)
	}
}

func TestMapPut(t *testing.T) {
	m, typ := maps.NewTestMap[uint32, uint64](8)

	key := uint32(0)
	elem := uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

		if maps.DebugLog {
			fmt.Printf("After put %d: %v\n", key, m)
		}
	}

	if m.Used() != 31 {
		t.Errorf("Used() used got %d want 31", m.Used())
	}

	key = uint32(0)
	elem = uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		got, ok := m.Get(typ, unsafe.Pointer(&key))
		if !ok {
			t.Errorf("Get(%d) got ok false want true", key)
		}
		gotElem := *(*uint64)(got)
		if gotElem != elem {
			t.Errorf("Get(%d) got elem %d want %d", key, gotElem, elem)
		}
	}
}

// Grow enough to cause a table split.
func TestMapSplit(t *testing.T) {
	m, typ := maps.NewTestMap[uint32, uint64](0)

	key := uint32(0)
	elem := uint64(256 + 0)

	for i := 0; i < 2*maps.MaxTableCapacity; i++ {
		key += 1
		elem += 1
		m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

		if maps.DebugLog {
			fmt.Printf("After put %d: %v\n", key, m)
		}
	}

	if m.Used() != 2*maps.MaxTableCapacity {
		t.Errorf("Used() used got %d want 31", m.Used())
	}

	key = uint32(0)
	elem = uint64(256 + 0)

	for i := 0; i < 2*maps.MaxTableCapacity; i++ {
		key += 1
		elem += 1
		got, ok := m.Get(typ, unsafe.Pointer(&key))
		if !ok {
			t.Errorf("Get(%d) got ok false want true", key)
		}
		gotElem := *(*uint64)(got)
		if gotElem != elem {
			t.Errorf("Get(%d) got elem %d want %d", key, gotElem, elem)
		}
	}
}

func TestMapDelete(t *testing.T) {
	m, typ := maps.NewTestMap[uint32, uint64](32)

	key := uint32(0)
	elem := uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

		if maps.DebugLog {
			fmt.Printf("After put %d: %v\n", key, m)
		}
	}

	key = uint32(0)
	elem = uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		m.Delete(typ, unsafe.Pointer(&key))
	}

	if m.Used() != 0 {
		t.Errorf("Used() used got %d want 0", m.Used())
	}

	key = uint32(0)
	elem = uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		_, ok := m.Get(typ, unsafe.Pointer(&key))
		if ok {
			t.Errorf("Get(%d) got ok true want false", key)
		}
	}
}

func TestTableClear(t *testing.T) {
	m, typ := maps.NewTestMap[uint32, uint64](32)

	key := uint32(0)
	elem := uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

		if maps.DebugLog {
			fmt.Printf("After put %d: %v\n", key, m)
		}
	}

	m.Clear(typ)

	if m.Used() != 0 {
		t.Errorf("Clear() used got %d want 0", m.Used())
	}

	key = uint32(0)
	elem = uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		_, ok := m.Get(typ, unsafe.Pointer(&key))
		if ok {
			t.Errorf("Get(%d) got ok true want false", key)
		}
	}
}

// +0.0 and -0.0 compare equal, but we must still must update the key slot when
// overwriting.
func TestTableKeyUpdate(t *testing.T) {
	m, typ := maps.NewTestMap[float64, uint64](8)

	zero := float64(0.0)
	negZero := math.Copysign(zero, -1.0)
	elem := uint64(0)

	m.Put(typ, unsafe.Pointer(&zero), unsafe.Pointer(&elem))
	if maps.DebugLog {
		fmt.Printf("After put %f: %v\n", zero, m)
	}

	elem = 1
	m.Put(typ, unsafe.Pointer(&negZero), unsafe.Pointer(&elem))
	if maps.DebugLog {
		fmt.Printf("After put %f: %v\n", negZero, m)
	}

	if m.Used() != 1 {
		t.Errorf("Used() used got %d want 1", m.Used())
	}

	it := new(maps.Iter)
	it.Init(typ, m)
	it.Next()
	keyPtr, elemPtr := it.Key(), it.Elem()
	if keyPtr == nil {
		t.Fatal("it.Key() got nil want key")
	}

	key := *(*float64)(keyPtr)
	elem = *(*uint64)(elemPtr)
	if math.Copysign(1.0, key) > 0 {
		t.Errorf("map key %f has positive sign", key)
	}
	if elem != 1 {
		t.Errorf("map elem got %d want 1", elem)
	}
}

// Put should reuse a deleted slot rather than consuming an empty slot.
func TestTablePutDelete(t *testing.T) {
	// Put will reuse the first deleted slot it encounters.
	//
	// This is awkward to test because Delete will only install ctrlDeleted
	// if the group is full, otherwise it goes straight to empty.
	//
	// So first we must add to the table continuously until we happen to
	// fill a group.

	// Avoid small maps, they have no tables.
	m, typ := maps.NewTestMap[uint32, uint32](16)

	key := uint32(0)
	elem := uint32(256 + 0)

	for {
		key += 1
		elem += 1

		m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

		// Normally a Put that fills a group would fill it with the
		// inserted key, so why search the whole map for a potentially
		// different key in a full group?
		//
		// Put may grow/split a table. Initial construction of the new
		// table(s) could result in a full group consisting of
		// arbitrary keys.
		fullKeyPtr := m.KeyFromFullGroup(typ)
		if fullKeyPtr != nil {
			// Found a full group.
			key = *(*uint32)(fullKeyPtr)
			elem = 256 + key
			break
		}
	}

	// Key is in a full group. Deleting it will result in a ctrlDeleted
	// slot.
	m.Delete(typ, unsafe.Pointer(&key))

	// Re-insert key. This should reuse the deleted slot rather than
	// consuming space.
	tabWant := m.TableFor(typ, unsafe.Pointer(&key))
	growthLeftWant := tabWant.GrowthLeft()

	m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

	tabGot := m.TableFor(typ, unsafe.Pointer(&key))
	growthLeftGot := tabGot.GrowthLeft()

	if tabGot != tabWant {
		// There shouldn't be a grow, as replacing a deleted slot
		// doesn't require more space.
		t.Errorf("Put(%d) grew table got %v want %v map %v", key, tabGot, tabWant, m)
	}

	if growthLeftGot != growthLeftWant {
		t.Errorf("GrowthLeft got %d want %d: map %v tab %v", growthLeftGot, growthLeftWant, m, tabGot)
	}
}

func TestTableIteration(t *testing.T) {
	m, typ := maps.NewTestMap[uint32, uint64](8)

	key := uint32(0)
	elem := uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

		if maps.DebugLog {
			fmt.Printf("After put %d: %v\n", key, m)
		}
	}

	got := make(map[uint32]uint64)

	it := new(maps.Iter)
	it.Init(typ, m)
	for {
		it.Next()
		keyPtr, elemPtr := it.Key(), it.Elem()
		if keyPtr == nil {
			break
		}

		key := *(*uint32)(keyPtr)
		elem := *(*uint64)(elemPtr)
		got[key] = elem
	}

	if len(got) != 31 {
		t.Errorf("Iteration got %d entries, want 31: %+v", len(got), got)
	}

	key = uint32(0)
	elem = uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		gotElem, ok := got[key]
		if !ok {
			t.Errorf("Iteration missing key %d", key)
			continue
		}
		if gotElem != elem {
			t.Errorf("Iteration key %d got elem %d want %d", key, gotElem, elem)
		}
	}
}

// Deleted keys shouldn't be visible in iteration.
func TestTableIterationDelete(t *testing.T) {
	m, typ := maps.NewTestMap[uint32, uint64](8)

	key := uint32(0)
	elem := uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

		if maps.DebugLog {
			fmt.Printf("After put %d: %v\n", key, m)
		}
	}

	got := make(map[uint32]uint64)
	first := true
	deletedKey := uint32(1)
	it := new(maps.Iter)
	it.Init(typ, m)
	for {
		it.Next()
		keyPtr, elemPtr := it.Key(), it.Elem()
		if keyPtr == nil {
			break
		}

		key := *(*uint32)(keyPtr)
		elem := *(*uint64)(elemPtr)
		got[key] = elem

		if first {
			first = false

			// If the key we intended to delete was the one we just
			// saw, pick another to delete.
			if key == deletedKey {
				deletedKey++
			}
			m.Delete(typ, unsafe.Pointer(&deletedKey))
		}
	}

	if len(got) != 30 {
		t.Errorf("Iteration got %d entries, want 30: %+v", len(got), got)
	}

	key = uint32(0)
	elem = uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1

		wantOK := true
		if key == deletedKey {
			wantOK = false
		}

		gotElem, gotOK := got[key]
		if gotOK != wantOK {
			t.Errorf("Iteration key %d got ok %v want ok %v", key, gotOK, wantOK)
			continue
		}
		if wantOK && gotElem != elem {
			t.Errorf("Iteration key %d got elem %d want %d", key, gotElem, elem)
		}
	}
}

// Deleted keys shouldn't be visible in iteration even after a grow.
func TestTableIterationGrowDelete(t *testing.T) {
	m, typ := maps.NewTestMap[uint32, uint64](8)

	key := uint32(0)
	elem := uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

		if maps.DebugLog {
			fmt.Printf("After put %d: %v\n", key, m)
		}
	}

	got := make(map[uint32]uint64)
	first := true
	deletedKey := uint32(1)
	it := new(maps.Iter)
	it.Init(typ, m)
	for {
		it.Next()
		keyPtr, elemPtr := it.Key(), it.Elem()
		if keyPtr == nil {
			break
		}

		key := *(*uint32)(keyPtr)
		elem := *(*uint64)(elemPtr)
		got[key] = elem

		if first {
			first = false

			// If the key we intended to delete was the one we just
			// saw, pick another to delete.
			if key == deletedKey {
				deletedKey++
			}

			// Double the number of elements to force a grow.
			key := uint32(32)
			elem := uint64(256 + 32)

			for i := 0; i < 31; i++ {
				key += 1
				elem += 1
				m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

				if maps.DebugLog {
					fmt.Printf("After put %d: %v\n", key, m)
				}
			}

			// Then delete from the grown map.
			m.Delete(typ, unsafe.Pointer(&deletedKey))
		}
	}

	// Don't check length: the number of new elements we'll see is
	// unspecified.

	// Check values only of the original pre-iteration entries.
	key = uint32(0)
	elem = uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1

		wantOK := true
		if key == deletedKey {
			wantOK = false
		}

		gotElem, gotOK := got[key]
		if gotOK != wantOK {
			t.Errorf("Iteration key %d got ok %v want ok %v", key, gotOK, wantOK)
			continue
		}
		if wantOK && gotElem != elem {
			t.Errorf("Iteration key %d got elem %d want %d", key, gotElem, elem)
		}
	}
}

func testTableIterationGrowDuplicate(t *testing.T, grow int) {
	m, typ := maps.NewTestMap[uint32, uint64](8)

	key := uint32(0)
	elem := uint64(256 + 0)

	for i := 0; i < 31; i++ {
		key += 1
		elem += 1
		m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

		if maps.DebugLog {
			fmt.Printf("After put %d: %v\n", key, m)
		}
	}

	got := make(map[uint32]uint64)
	it := new(maps.Iter)
	it.Init(typ, m)
	for i := 0; ; i++ {
		it.Next()
		keyPtr, elemPtr := it.Key(), it.Elem()
		if keyPtr == nil {
			break
		}

		key := *(*uint32)(keyPtr)
		elem := *(*uint64)(elemPtr)
		if elem != 256+uint64(key) {
			t.Errorf("iteration got key %d elem %d want elem %d", key, elem, 256+uint64(key))
		}
		if _, ok := got[key]; ok {
			t.Errorf("iteration got key %d more than once", key)
		}
		got[key] = elem

		// Grow halfway through iteration.
		if i == 16 {
			key := uint32(32)
			elem := uint64(256 + 32)

			for i := 0; i < grow; i++ {
				key += 1
				elem += 1
				m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

				if maps.DebugLog {
					fmt.Printf("After put %d: %v\n", key, m)
				}
			}
		}
	}

	// Don't check length: the number of new elements we'll see is
	// unspecified.
}

// Grow should not allow duplicate keys to appear.
func TestTableIterationGrowDuplicate(t *testing.T) {
	// Small grow, only enough to cause table grow.
	t.Run("grow", func(t *testing.T) { testTableIterationGrowDuplicate(t, 32) })

	// Large grow, to cause table split.
	t.Run("split", func(t *testing.T) { testTableIterationGrowDuplicate(t, 2*maps.MaxTableCapacity) })
}

func TestAlignUpPow2(t *testing.T) {
	tests := []struct {
		in       uint64
		want     uint64
		overflow bool
	}{
		{
			in:   0,
			want: 0,
		},
		{
			in:   3,
			want: 4,
		},
		{
			in:   4,
			want: 4,
		},
		{
			in:   1 << 63,
			want: 1 << 63,
		},
		{
			in:   (1 << 63) - 1,
			want: 1 << 63,
		},
		{
			in:       (1 << 63) + 1,
			overflow: true,
		},
	}

	for _, tc := range tests {
		got, overflow := maps.AlignUpPow2(tc.in)
		if got != tc.want {
			t.Errorf("alignUpPow2(%d) got %d, want %d", tc.in, got, tc.want)
		}
		if overflow != tc.overflow {
			t.Errorf("alignUpPow2(%d) got overflow %v, want %v", tc.in, overflow, tc.overflow)
		}
	}
}

// Verify that a map with zero-size slot is safe to use.
func TestMapZeroSizeSlot(t *testing.T) {
	m, typ := maps.NewTestMap[struct{}, struct{}](16)

	key := struct{}{}
	elem := struct{}{}

	m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

	if maps.DebugLog {
		fmt.Printf("After put %d: %v\n", key, m)
	}

	got, ok := m.Get(typ, unsafe.Pointer(&key))
	if !ok {
		t.Errorf("Get(%d) got ok false want true", key)
	}
	gotElem := *(*struct{})(got)
	if gotElem != elem {
		t.Errorf("Get(%d) got elem %d want %d", key, gotElem, elem)
	}

	tab := m.TableFor(typ, unsafe.Pointer(&key))
	start := tab.GroupsStart()
	length := tab.GroupsLength()
	end := unsafe.Pointer(uintptr(start) + length*typ.GroupSize - 1) // inclusive to ensure we have a valid pointer
	if uintptr(got) < uintptr(start) || uintptr(got) > uintptr(end) {
		t.Errorf("elem address outside groups allocation; got %p want [%p, %p]", got, start, end)
	}
}

func TestMapIndirect(t *testing.T) {
	type big [abi.SwissMapMaxKeyBytes + abi.SwissMapMaxElemBytes]byte

	m, typ := maps.NewTestMap[big, big](8)

	key := big{}
	elem := big{}
	elem[0] = 128

	for i := 0; i < 31; i++ {
		key[0] += 1
		elem[0] += 1
		m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

		if maps.DebugLog {
			fmt.Printf("After put %v: %v\n", key, m)
		}
	}

	if m.Used() != 31 {
		t.Errorf("Used() used got %d want 31", m.Used())
	}

	key = big{}
	elem = big{}
	elem[0] = 128

	for i := 0; i < 31; i++ {
		key[0] += 1
		elem[0] += 1
		got, ok := m.Get(typ, unsafe.Pointer(&key))
		if !ok {
			t.Errorf("Get(%v) got ok false want true", key)
		}
		gotElem := *(*big)(got)
		if gotElem != elem {
			t.Errorf("Get(%v) got elem %v want %v", key, gotElem, elem)
		}
	}
}

// Delete should clear element. See https://go.dev/issue/25936.
func TestMapDeleteClear(t *testing.T) {
	m, typ := maps.NewTestMap[int64, int64](8)

	key := int64(0)
	elem := int64(128)

	m.Put(typ, unsafe.Pointer(&key), unsafe.Pointer(&elem))

	if maps.DebugLog {
		fmt.Printf("After put %d: %v\n", key, m)
	}

	got, ok := m.Get(typ, unsafe.Pointer(&key))
	if !ok {
		t.Errorf("Get(%d) got ok false want true", key)
	}
	gotElem := *(*int64)(got)
	if gotElem != elem {
		t.Errorf("Get(%d) got elem %d want %d", key, gotElem, elem)
	}

	m.Delete(typ, unsafe.Pointer(&key))

	gotElem = *(*int64)(got)
	if gotElem != 0 {
		t.Errorf("Delete(%d) failed to clear element. got %d want 0", key, gotElem)
	}
}

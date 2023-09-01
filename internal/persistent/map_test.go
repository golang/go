// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package persistent

import (
	"fmt"
	"math/rand"
	"reflect"
	"sync/atomic"
	"testing"
)

type mapEntry struct {
	key   int
	value int
}

type validatedMap struct {
	impl     *Map[int, int]
	expected map[int]int      // current key-value mapping.
	deleted  map[mapEntry]int // maps deleted entries to their clock time of last deletion
	seen     map[mapEntry]int // maps seen entries to their clock time of last insertion
	clock    int
}

func TestSimpleMap(t *testing.T) {
	deletedEntries := make(map[mapEntry]int)
	seenEntries := make(map[mapEntry]int)

	m1 := &validatedMap{
		impl:     new(Map[int, int]),
		expected: make(map[int]int),
		deleted:  deletedEntries,
		seen:     seenEntries,
	}

	m3 := m1.clone()
	validateRef(t, m1, m3)
	m3.set(t, 8, 8)
	validateRef(t, m1, m3)
	m3.destroy()

	assertSameMap(t, entrySet(deletedEntries), map[mapEntry]struct{}{
		{key: 8, value: 8}: {},
	})

	validateRef(t, m1)
	m1.set(t, 1, 1)
	validateRef(t, m1)
	m1.set(t, 2, 2)
	validateRef(t, m1)
	m1.set(t, 3, 3)
	validateRef(t, m1)
	m1.remove(t, 2)
	validateRef(t, m1)
	m1.set(t, 6, 6)
	validateRef(t, m1)

	assertSameMap(t, entrySet(deletedEntries), map[mapEntry]struct{}{
		{key: 2, value: 2}: {},
		{key: 8, value: 8}: {},
	})

	m2 := m1.clone()
	validateRef(t, m1, m2)
	m1.set(t, 6, 60)
	validateRef(t, m1, m2)
	m1.remove(t, 1)
	validateRef(t, m1, m2)

	gotAllocs := int(testing.AllocsPerRun(10, func() {
		m1.impl.Delete(100)
		m1.impl.Delete(1)
	}))
	wantAllocs := 0
	if gotAllocs != wantAllocs {
		t.Errorf("wanted %d allocs, got %d", wantAllocs, gotAllocs)
	}

	for i := 10; i < 14; i++ {
		m1.set(t, i, i)
		validateRef(t, m1, m2)
	}

	m1.set(t, 10, 100)
	validateRef(t, m1, m2)

	m1.remove(t, 12)
	validateRef(t, m1, m2)

	m2.set(t, 4, 4)
	validateRef(t, m1, m2)
	m2.set(t, 5, 5)
	validateRef(t, m1, m2)

	m1.destroy()

	assertSameMap(t, entrySet(deletedEntries), map[mapEntry]struct{}{
		{key: 2, value: 2}:    {},
		{key: 6, value: 60}:   {},
		{key: 8, value: 8}:    {},
		{key: 10, value: 10}:  {},
		{key: 10, value: 100}: {},
		{key: 11, value: 11}:  {},
		{key: 12, value: 12}:  {},
		{key: 13, value: 13}:  {},
	})

	m2.set(t, 7, 7)
	validateRef(t, m2)

	m2.destroy()

	assertSameMap(t, entrySet(seenEntries), entrySet(deletedEntries))
}

func TestRandomMap(t *testing.T) {
	deletedEntries := make(map[mapEntry]int)
	seenEntries := make(map[mapEntry]int)

	m := &validatedMap{
		impl:     new(Map[int, int]),
		expected: make(map[int]int),
		deleted:  deletedEntries,
		seen:     seenEntries,
	}

	keys := make([]int, 0, 1000)
	for i := 0; i < 1000; i++ {
		key := rand.Intn(10000)
		m.set(t, key, key)
		keys = append(keys, key)

		if i%10 == 1 {
			index := rand.Intn(len(keys))
			last := len(keys) - 1
			key = keys[index]
			keys[index], keys[last] = keys[last], keys[index]
			keys = keys[:last]

			m.remove(t, key)
		}
	}

	m.destroy()
	assertSameMap(t, entrySet(seenEntries), entrySet(deletedEntries))
}

func entrySet(m map[mapEntry]int) map[mapEntry]struct{} {
	set := make(map[mapEntry]struct{})
	for k := range m {
		set[k] = struct{}{}
	}
	return set
}

func TestUpdate(t *testing.T) {
	deletedEntries := make(map[mapEntry]int)
	seenEntries := make(map[mapEntry]int)

	m1 := &validatedMap{
		impl:     new(Map[int, int]),
		expected: make(map[int]int),
		deleted:  deletedEntries,
		seen:     seenEntries,
	}
	m2 := m1.clone()

	m1.set(t, 1, 1)
	m1.set(t, 2, 2)
	m2.set(t, 2, 20)
	m2.set(t, 3, 3)
	m1.setAll(t, m2)

	m1.destroy()
	m2.destroy()
	assertSameMap(t, entrySet(seenEntries), entrySet(deletedEntries))
}

func validateRef(t *testing.T, maps ...*validatedMap) {
	t.Helper()

	actualCountByEntry := make(map[mapEntry]int32)
	nodesByEntry := make(map[mapEntry]map[*mapNode]struct{})
	expectedCountByEntry := make(map[mapEntry]int32)
	for i, m := range maps {
		dfsRef(m.impl.root, actualCountByEntry, nodesByEntry)
		dumpMap(t, fmt.Sprintf("%d:", i), m.impl.root)
	}
	for entry, nodes := range nodesByEntry {
		expectedCountByEntry[entry] = int32(len(nodes))
	}
	assertSameMap(t, expectedCountByEntry, actualCountByEntry)
}

func dfsRef(node *mapNode, countByEntry map[mapEntry]int32, nodesByEntry map[mapEntry]map[*mapNode]struct{}) {
	if node == nil {
		return
	}

	entry := mapEntry{key: node.key.(int), value: node.value.value.(int)}
	countByEntry[entry] = atomic.LoadInt32(&node.value.refCount)

	nodes, ok := nodesByEntry[entry]
	if !ok {
		nodes = make(map[*mapNode]struct{})
		nodesByEntry[entry] = nodes
	}
	nodes[node] = struct{}{}

	dfsRef(node.left, countByEntry, nodesByEntry)
	dfsRef(node.right, countByEntry, nodesByEntry)
}

func dumpMap(t *testing.T, prefix string, n *mapNode) {
	if n == nil {
		t.Logf("%s nil", prefix)
		return
	}
	t.Logf("%s {key: %v, value: %v (ref: %v), ref: %v, weight: %v}", prefix, n.key, n.value.value, n.value.refCount, n.refCount, n.weight)
	dumpMap(t, prefix+"l", n.left)
	dumpMap(t, prefix+"r", n.right)
}

func (vm *validatedMap) validate(t *testing.T) {
	t.Helper()

	validateNode(t, vm.impl.root)

	// Note: this validation may not make sense if maps were constructed using
	// SetAll operations. If this proves to be problematic, remove the clock,
	// deleted, and seen fields.
	for key, value := range vm.expected {
		entry := mapEntry{key: key, value: value}
		if deleteAt := vm.deleted[entry]; deleteAt > vm.seen[entry] {
			t.Fatalf("entry is deleted prematurely, key: %d, value: %d", key, value)
		}
	}

	actualMap := make(map[int]int, len(vm.expected))
	vm.impl.Range(func(key, value int) {
		if other, ok := actualMap[key]; ok {
			t.Fatalf("key is present twice, key: %d, first value: %d, second value: %d", key, value, other)
		}
		actualMap[key] = value
	})

	assertSameMap(t, actualMap, vm.expected)
}

func validateNode(t *testing.T, node *mapNode) {
	if node == nil {
		return
	}

	if node.left != nil {
		if node.key.(int) < node.left.key.(int) {
			t.Fatalf("left child has larger key: %v vs %v", node.left.key, node.key)
		}
		if node.left.weight > node.weight {
			t.Fatalf("left child has larger weight: %v vs %v", node.left.weight, node.weight)
		}
	}

	if node.right != nil {
		if node.right.key.(int) < node.key.(int) {
			t.Fatalf("right child has smaller key: %v vs %v", node.right.key, node.key)
		}
		if node.right.weight > node.weight {
			t.Fatalf("right child has larger weight: %v vs %v", node.right.weight, node.weight)
		}
	}

	validateNode(t, node.left)
	validateNode(t, node.right)
}

func (vm *validatedMap) setAll(t *testing.T, other *validatedMap) {
	vm.impl.SetAll(other.impl)

	// Note: this is buggy because we are not updating vm.clock, vm.deleted, or
	// vm.seen.
	for key, value := range other.expected {
		vm.expected[key] = value
	}
	vm.validate(t)
}

func (vm *validatedMap) set(t *testing.T, key, value int) {
	entry := mapEntry{key: key, value: value}

	vm.clock++
	vm.seen[entry] = vm.clock

	vm.impl.Set(key, value, func(deletedKey, deletedValue any) {
		if deletedKey != key || deletedValue != value {
			t.Fatalf("unexpected passed in deleted entry: %v/%v, expected: %v/%v", deletedKey, deletedValue, key, value)
		}
		// Not safe if closure shared between two validatedMaps.
		vm.deleted[entry] = vm.clock
	})
	vm.expected[key] = value
	vm.validate(t)

	gotValue, ok := vm.impl.Get(key)
	if !ok || gotValue != value {
		t.Fatalf("unexpected get result after insertion, key: %v, expected: %v, got: %v (%v)", key, value, gotValue, ok)
	}
}

func (vm *validatedMap) remove(t *testing.T, key int) {
	vm.clock++
	vm.impl.Delete(key)
	delete(vm.expected, key)
	vm.validate(t)

	gotValue, ok := vm.impl.Get(key)
	if ok {
		t.Fatalf("unexpected get result after removal, key: %v, got: %v", key, gotValue)
	}
}

func (vm *validatedMap) clone() *validatedMap {
	expected := make(map[int]int, len(vm.expected))
	for key, value := range vm.expected {
		expected[key] = value
	}

	return &validatedMap{
		impl:     vm.impl.Clone(),
		expected: expected,
		deleted:  vm.deleted,
		seen:     vm.seen,
	}
}

func (vm *validatedMap) destroy() {
	vm.impl.Destroy()
}

func assertSameMap(t *testing.T, map1, map2 any) {
	t.Helper()

	if !reflect.DeepEqual(map1, map2) {
		t.Fatalf("different maps:\n%v\nvs\n%v", map1, map2)
	}
}

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
	impl     *Map
	expected map[int]int
	deleted  map[mapEntry]struct{}
	seen     map[mapEntry]struct{}
}

func TestSimpleMap(t *testing.T) {
	deletedEntries := make(map[mapEntry]struct{})
	seenEntries := make(map[mapEntry]struct{})

	m1 := &validatedMap{
		impl: NewMap(func(a, b interface{}) bool {
			return a.(int) < b.(int)
		}),
		expected: make(map[int]int),
		deleted:  deletedEntries,
		seen:     seenEntries,
	}

	m3 := m1.clone()
	validateRef(t, m1, m3)
	m3.set(t, 8, 8)
	validateRef(t, m1, m3)
	m3.destroy()

	assertSameMap(t, deletedEntries, map[mapEntry]struct{}{
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

	assertSameMap(t, deletedEntries, map[mapEntry]struct{}{
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

	assertSameMap(t, deletedEntries, map[mapEntry]struct{}{
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

	assertSameMap(t, seenEntries, deletedEntries)
}

func TestRandomMap(t *testing.T) {
	deletedEntries := make(map[mapEntry]struct{})
	seenEntries := make(map[mapEntry]struct{})

	m := &validatedMap{
		impl: NewMap(func(a, b interface{}) bool {
			return a.(int) < b.(int)
		}),
		expected: make(map[int]int),
		deleted:  deletedEntries,
		seen:     seenEntries,
	}

	keys := make([]int, 0, 1000)
	for i := 0; i < 1000; i++ {
		key := rand.Int()
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
	assertSameMap(t, seenEntries, deletedEntries)
}

func TestUpdate(t *testing.T) {
	deletedEntries := make(map[mapEntry]struct{})
	seenEntries := make(map[mapEntry]struct{})

	m1 := &validatedMap{
		impl: NewMap(func(a, b interface{}) bool {
			return a.(int) < b.(int)
		}),
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
	assertSameMap(t, seenEntries, deletedEntries)
}

func (vm *validatedMap) onDelete(t *testing.T, key, value int) {
	entry := mapEntry{key: key, value: value}
	if _, ok := vm.deleted[entry]; ok {
		t.Fatalf("tried to delete entry twice, key: %d, value: %d", key, value)
	}
	vm.deleted[entry] = struct{}{}
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

	validateNode(t, vm.impl.root, vm.impl.less)

	for key, value := range vm.expected {
		entry := mapEntry{key: key, value: value}
		if _, ok := vm.deleted[entry]; ok {
			t.Fatalf("entry is deleted prematurely, key: %d, value: %d", key, value)
		}
	}

	actualMap := make(map[int]int, len(vm.expected))
	vm.impl.Range(func(key, value interface{}) {
		if other, ok := actualMap[key.(int)]; ok {
			t.Fatalf("key is present twice, key: %d, first value: %d, second value: %d", key, value, other)
		}
		actualMap[key.(int)] = value.(int)
	})

	assertSameMap(t, actualMap, vm.expected)
}

func validateNode(t *testing.T, node *mapNode, less func(a, b interface{}) bool) {
	if node == nil {
		return
	}

	if node.left != nil {
		if less(node.key, node.left.key) {
			t.Fatalf("left child has larger key: %v vs %v", node.left.key, node.key)
		}
		if node.left.weight > node.weight {
			t.Fatalf("left child has larger weight: %v vs %v", node.left.weight, node.weight)
		}
	}

	if node.right != nil {
		if less(node.right.key, node.key) {
			t.Fatalf("right child has smaller key: %v vs %v", node.right.key, node.key)
		}
		if node.right.weight > node.weight {
			t.Fatalf("right child has larger weight: %v vs %v", node.right.weight, node.weight)
		}
	}

	validateNode(t, node.left, less)
	validateNode(t, node.right, less)
}

func (vm *validatedMap) setAll(t *testing.T, other *validatedMap) {
	vm.impl.SetAll(other.impl)
	for key, value := range other.expected {
		vm.expected[key] = value
	}
	vm.validate(t)
}

func (vm *validatedMap) set(t *testing.T, key, value int) {
	vm.seen[mapEntry{key: key, value: value}] = struct{}{}
	vm.impl.Set(key, value, func(deletedKey, deletedValue interface{}) {
		if deletedKey != key || deletedValue != value {
			t.Fatalf("unexpected passed in deleted entry: %v/%v, expected: %v/%v", deletedKey, deletedValue, key, value)
		}
		vm.onDelete(t, key, value)
	})
	vm.expected[key] = value
	vm.validate(t)

	gotValue, ok := vm.impl.Get(key)
	if !ok || gotValue != value {
		t.Fatalf("unexpected get result after insertion, key: %v, expected: %v, got: %v (%v)", key, value, gotValue, ok)
	}
}

func (vm *validatedMap) remove(t *testing.T, key int) {
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

func assertSameMap(t *testing.T, map1, map2 interface{}) {
	t.Helper()

	if !reflect.DeepEqual(map1, map2) {
		t.Fatalf("different maps:\n%v\nvs\n%v", map1, map2)
	}
}

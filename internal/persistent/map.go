// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The persistent package defines various persistent data structures;
// that is, data structures that can be efficiently copied and modified
// in sublinear time.
package persistent

import (
	"fmt"
	"math/rand"
	"strings"
	"sync/atomic"
)

// Implementation details:
// * Each value is reference counted by nodes which hold it.
// * Each node is reference counted by its parent nodes.
// * Each map is considered a top-level parent node from reference counting perspective.
// * Each change does always effectively produce a new top level node.
//
// Functions which operate directly with nodes do have a notation in form of
// `foo(arg1:+n1, arg2:+n2) (ret1:+n3)`.
// Each argument is followed by a delta change to its reference counter.
// In case if no change is expected, the delta will be `-0`.

// Map is an associative mapping from keys to values, both represented as
// interface{}. Key comparison and iteration order is defined by a
// client-provided function that implements a strict weak order.
//
// Maps can be Cloned in constant time.
// Get, Store, and Delete operations are done on average in logarithmic time.
// Maps can be Updated in O(m log(n/m)) time for maps of size n and m, where m < n.
//
// Values are reference counted, and a client-supplied release function
// is called when a value is no longer referenced by a map or any clone.
//
// Internally the implementation is based on a randomized persistent treap:
// https://en.wikipedia.org/wiki/Treap.
type Map struct {
	less func(a, b interface{}) bool
	root *mapNode
}

func (m *Map) String() string {
	var buf strings.Builder
	buf.WriteByte('{')
	var sep string
	m.Range(func(k, v interface{}) {
		fmt.Fprintf(&buf, "%s%v: %v", sep, k, v)
		sep = ", "
	})
	buf.WriteByte('}')
	return buf.String()
}

type mapNode struct {
	key         interface{}
	value       *refValue
	weight      uint64
	refCount    int32
	left, right *mapNode
}

type refValue struct {
	refCount int32
	value    interface{}
	release  func(key, value interface{})
}

func newNodeWithRef(key, value interface{}, release func(key, value interface{})) *mapNode {
	return &mapNode{
		key: key,
		value: &refValue{
			value:    value,
			release:  release,
			refCount: 1,
		},
		refCount: 1,
		weight:   rand.Uint64(),
	}
}

func (node *mapNode) shallowCloneWithRef() *mapNode {
	atomic.AddInt32(&node.value.refCount, 1)
	return &mapNode{
		key:      node.key,
		value:    node.value,
		weight:   node.weight,
		refCount: 1,
	}
}

func (node *mapNode) incref() *mapNode {
	if node != nil {
		atomic.AddInt32(&node.refCount, 1)
	}
	return node
}

func (node *mapNode) decref() {
	if node == nil {
		return
	}
	if atomic.AddInt32(&node.refCount, -1) == 0 {
		if atomic.AddInt32(&node.value.refCount, -1) == 0 {
			if node.value.release != nil {
				node.value.release(node.key, node.value.value)
			}
			node.value.value = nil
			node.value.release = nil
		}
		node.left.decref()
		node.right.decref()
	}
}

// NewMap returns a new map whose keys are ordered by the given comparison
// function (a strict weak order). It is the responsibility of the caller to
// Destroy it at later time.
func NewMap(less func(a, b interface{}) bool) *Map {
	return &Map{
		less: less,
	}
}

// Clone returns a copy of the given map. It is a responsibility of the caller
// to Destroy it at later time.
func (pm *Map) Clone() *Map {
	return &Map{
		less: pm.less,
		root: pm.root.incref(),
	}
}

// Destroy destroys the map.
//
// After Destroy, the Map should not be used again.
func (pm *Map) Destroy() {
	// The implementation of these two functions is the same,
	// but their intent is different.
	pm.Clear()
}

// Clear removes all entries from the map.
func (pm *Map) Clear() {
	pm.root.decref()
	pm.root = nil
}

// Range calls f sequentially in ascending key order for all entries in the map.
func (pm *Map) Range(f func(key, value interface{})) {
	pm.root.forEach(f)
}

func (node *mapNode) forEach(f func(key, value interface{})) {
	if node == nil {
		return
	}
	node.left.forEach(f)
	f(node.key, node.value.value)
	node.right.forEach(f)
}

// Get returns the map value associated with the specified key, or nil if no entry
// is present. The ok result indicates whether an entry was found in the map.
func (pm *Map) Get(key interface{}) (interface{}, bool) {
	node := pm.root
	for node != nil {
		if pm.less(key, node.key) {
			node = node.left
		} else if pm.less(node.key, key) {
			node = node.right
		} else {
			return node.value.value, true
		}
	}
	return nil, false
}

// SetAll updates the map with key/value pairs from the other map, overwriting existing keys.
// It is equivalent to calling Set for each entry in the other map but is more efficient.
// Both maps must have the same comparison function, otherwise behavior is undefined.
func (pm *Map) SetAll(other *Map) {
	root := pm.root
	pm.root = union(root, other.root, pm.less, true)
	root.decref()
}

// Set updates the value associated with the specified key.
// If release is non-nil, it will be called with entry's key and value once the
// key is no longer contained in the map or any clone.
func (pm *Map) Set(key, value interface{}, release func(key, value interface{})) {
	first := pm.root
	second := newNodeWithRef(key, value, release)
	pm.root = union(first, second, pm.less, true)
	first.decref()
	second.decref()
}

// union returns a new tree which is a union of first and second one.
// If overwrite is set to true, second one would override a value for any duplicate keys.
//
// union(first:-0, second:-0) (result:+1)
// Union borrows both subtrees without affecting their refcount and returns a
// new reference that the caller is expected to call decref.
func union(first, second *mapNode, less func(a, b interface{}) bool, overwrite bool) *mapNode {
	if first == nil {
		return second.incref()
	}
	if second == nil {
		return first.incref()
	}

	if first.weight < second.weight {
		second, first, overwrite = first, second, !overwrite
	}

	left, mid, right := split(second, first.key, less, false)
	var result *mapNode
	if overwrite && mid != nil {
		result = mid.shallowCloneWithRef()
	} else {
		result = first.shallowCloneWithRef()
	}
	result.weight = first.weight
	result.left = union(first.left, left, less, overwrite)
	result.right = union(first.right, right, less, overwrite)
	left.decref()
	mid.decref()
	right.decref()
	return result
}

// split the tree midway by the key into three different ones.
// Return three new trees: left with all nodes with smaller than key, mid with
// the node matching the key, right with all nodes larger than key.
// If there are no nodes in one of trees, return nil instead of it.
// If requireMid is set (such as during deletion), then all return arguments
// are nil if mid is not found.
//
// split(n:-0) (left:+1, mid:+1, right:+1)
// Split borrows n without affecting its refcount, and returns three
// new references that the caller is expected to call decref.
func split(n *mapNode, key interface{}, less func(a, b interface{}) bool, requireMid bool) (left, mid, right *mapNode) {
	if n == nil {
		return nil, nil, nil
	}

	if less(n.key, key) {
		left, mid, right := split(n.right, key, less, requireMid)
		if requireMid && mid == nil {
			return nil, nil, nil
		}
		newN := n.shallowCloneWithRef()
		newN.left = n.left.incref()
		newN.right = left
		return newN, mid, right
	} else if less(key, n.key) {
		left, mid, right := split(n.left, key, less, requireMid)
		if requireMid && mid == nil {
			return nil, nil, nil
		}
		newN := n.shallowCloneWithRef()
		newN.left = right
		newN.right = n.right.incref()
		return left, mid, newN
	}
	mid = n.shallowCloneWithRef()
	return n.left.incref(), mid, n.right.incref()
}

// Delete deletes the value for a key.
func (pm *Map) Delete(key interface{}) {
	root := pm.root
	left, mid, right := split(root, key, pm.less, true)
	if mid == nil {
		return
	}
	pm.root = merge(left, right)
	left.decref()
	mid.decref()
	right.decref()
	root.decref()
}

// merge two trees while preserving the weight invariant.
// All nodes in left must have smaller keys than any node in right.
//
// merge(left:-0, right:-0) (result:+1)
// Merge borrows its arguments without affecting their refcount
// and returns a new reference that the caller is expected to call decref.
func merge(left, right *mapNode) *mapNode {
	switch {
	case left == nil:
		return right.incref()
	case right == nil:
		return left.incref()
	case left.weight > right.weight:
		root := left.shallowCloneWithRef()
		root.left = left.left.incref()
		root.right = merge(left.right, right)
		return root
	default:
		root := right.shallowCloneWithRef()
		root.left = merge(left, right.left)
		root.right = right.right.incref()
		return root
	}
}

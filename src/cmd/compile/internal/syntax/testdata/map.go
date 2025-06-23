// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package orderedmap provides an ordered map, implemented as a binary tree.
package orderedmap

import "chans"

// Map is an ordered map.
type Map[K, V any] struct {
	root    *node[K, V]
	compare func(K, K) int
}

// node is the type of a node in the binary tree.
type node[K, V any] struct {
	key         K
	val         V
	left, right *node[K, V]
}

// New returns a new map.
func New[K, V any](compare func(K, K) int) *Map[K, V] {
        return &Map[K, V]{compare: compare}
}

// find looks up key in the map, and returns either a pointer
// to the node holding key, or a pointer to the location where
// such a node would go.
func (m *Map[K, V]) find(key K) **node[K, V] {
	pn := &m.root
	for *pn != nil {
		switch cmp := m.compare(key, (*pn).key); {
		case cmp < 0:
			pn = &(*pn).left
		case cmp > 0:
			pn = &(*pn).right
		default:
			return pn
		}
	}
	return pn
}

// Insert inserts a new key/value into the map.
// If the key is already present, the value is replaced.
// Returns true if this is a new key, false if already present.
func (m *Map[K, V]) Insert(key K, val V) bool {
	pn := m.find(key)
	if *pn != nil {
		(*pn).val = val
		return false
	}
        *pn = &node[K, V]{key: key, val: val}
	return true
}

// Find returns the value associated with a key, or zero if not present.
// The found result reports whether the key was found.
func (m *Map[K, V]) Find(key K) (V, bool) {
	pn := m.find(key)
	if *pn == nil {
		var zero V // see the discussion of zero values, above
		return zero, false
	}
	return (*pn).val, true
}

// keyValue is a pair of key and value used when iterating.
type keyValue[K, V any] struct {
	key K
	val V
}

// InOrder returns an iterator that does an in-order traversal of the map.
func (m *Map[K, V]) InOrder() *Iterator[K, V] {
	sender, receiver := chans.Ranger[keyValue[K, V]]()
	var f func(*node[K, V]) bool
	f = func(n *node[K, V]) bool {
		if n == nil {
			return true
		}
		// Stop sending values if sender.Send returns false,
		// meaning that nothing is listening at the receiver end.
		return f(n.left) &&
                        sender.Send(keyValue[K, V]{n.key, n.val}) &&
			f(n.right)
	}
	go func() {
		f(m.root)
		sender.Close()
	}()
	return &Iterator[K, V]{receiver}
}

// Iterator is used to iterate over the map.
type Iterator[K, V any] struct {
	r *chans.Receiver[keyValue[K, V]]
}

// Next returns the next key and value pair, and a boolean indicating
// whether they are valid or whether we have reached the end.
func (it *Iterator[K, V]) Next() (K, V, bool) {
	keyval, ok := it.r.Next()
	if !ok {
		var zerok K
		var zerov V
		return zerok, zerov, false
	}
	return keyval.key, keyval.val, true
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "sort"

// A trie[V] maps keys to values V, similar to a map.
// A key is a list of integers; two keys collide if one of them is a prefix of the other.
// For instance, [1, 2] and [1, 2, 3] collide, but [1, 2, 3] and [1, 2, 4] do not.
// If all keys have length 1, a trie degenerates into an ordinary map[int]V.
type trie[V any] map[int]any // map value is either trie[V] (non-leaf node), or V (leaf node)

// insert inserts a value with a key into the trie; key must not be empty.
// If key doesn't collide with any other key in the trie, insert succeeds
// and returns (val, 0). Otherwise, insert fails and returns (alt, n) where
// alt is an unspecified but deterministically chosen value with a colliding
// key and n is the length > 0 of the common key prefix. The trie is not
// changed in this case.
func (tr trie[V]) insert(key []int, val V) (V, int) {
	// A key serves as the path from the trie root to its corresponding value.
	for l, index := range key {
		if v, exists := tr[index]; exists {
			// An entry already exists for this index; check its type to determine collision.
			switch v := v.(type) {
			case trie[V]:
				// Path continues.
				// Must check this case first in case V is any which would act as catch-all.
				tr = v
			case V:
				// Collision: An existing key ends here.
				// This means the existing key is a prefix of (or exactly equal to) key.
				return v, l + 1
			case nil:
				// Handle esoteric case where V is any and val is nil.
				var zero V
				return zero, l + 1
			default:
				panic("trie.insert: invalid entry")
			}
		} else {
			// Path doesn't exist yet, we need to build it.
			if l == len(key)-1 {
				// No prefix collision detected; insert val as a new leaf node.
				tr[index] = val
				return val, 0
			}
			node := make(trie[V])
			tr[index] = node
			tr = node
		}
	}

	if len(key) == 0 {
		panic("trie.insert: key must not be empty")
	}

	// Collision: path ends here, but the trie continues.
	// This means key is a prefix of an existing key.
	// Return a value from the subtrie.
	for len(tr) != 0 {
		// see switch above
		switch v := tr.pickValue().(type) {
		case trie[V]:
			tr = v
		case V:
			return v, len(key)
		case nil:
			var zero V
			return zero, len(key)
		default:
			panic("trie.insert: invalid entry")
		}
	}

	panic("trie.insert: unreachable")
}

// pickValue deterministically picks a value of the trie and returns it.
// Only called in case of a collision.
// The trie must not be empty.
func (tr trie[V]) pickValue() any {
	var keys []int
	for k := range tr {
		keys = append(keys, k)
	}
	sort.Ints(keys) // guarantee deterministic element pick
	return tr[keys[0]]
}

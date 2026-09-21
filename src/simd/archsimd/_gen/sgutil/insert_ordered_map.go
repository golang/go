// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgutil

import "iter"

type InsertMap[K comparable, V any] struct {
	m map[K]uint32 // Maps keys to their insertion index in the values slice.
	v []V          // Stores values in their insertion order.
	k []K          // Lazy slice of keys in insertion order, for iterators.
}

// Put inserts or updates a key-value pair in the map.
// If the key already exists, its value is updated while its insertion order
// remains unchanged. Returns the old value if there is one,
// and a boolean indicating whether an update occurred.
func (im *InsertMap[K, V]) Put(key K, val V) (old V, updated bool) {
	if im.m == nil {
		im.m = make(map[K]uint32)
	}

	var index uint32
	index, updated = im.m[key]
	if updated {
		// Key already exists; update its value in-place in the values slice.
		old = im.v[index]
		im.v[index] = val
		return
	}

	// Keep the keys slice synchronized if it has already been initialized.
	if len(im.v) > 0 && len(im.k) == len(im.v) {
		im.k = append(im.k, key)
	}

	// Record the new key's insertion index and append the value.
	im.m[key] = uint32(len(im.v))
	im.v = append(im.v, val)
	return
}

// updateKeys synchronizes the lazy keys slice (im.k) with the values slice (im.v).
// It populates im.k with keys at their respective insertion indices using
// the mapping stored in im.m.
func (im *InsertMap[K, V]) updateKeys() {
	if len(im.k) == len(im.v) {
		return
	}
	im.k = make([]K, len(im.v))
	for k, i := range im.m {
		im.k[i] = k
	}
}

// Contains reports whether the map contains the specified key.
func (im *InsertMap[K, V]) Contains(key K) bool {
	if im.m == nil {
		return false
	}
	_, ok := im.m[key]
	return ok
}

// Get returns the value associated with the specified key.
// If the key does not exist, the zero value of type V is returned.
func (im *InsertMap[K, V]) Get(key K) V {
	if im.m != nil {
		index, ok := im.m[key]
		if ok {
			return im.v[index]
		}
	}
	var v V
	return v
}

// GetOk returns the value associated with the specified key and
// a boolean indicating whether the key exists in the map.
// If the key is not in the map, the zero value is returned along with false.
func (im *InsertMap[K, V]) GetOk(key K) (V, bool) {
	if im.m != nil {
		index, ok := im.m[key]
		if ok {
			return im.v[index], true
		}
	}
	var v V
	return v, false
}

// Compare compares two keys based on their insertion order.
// It returns:
//   - -1 if 'a' was inserted before 'b'
//   - 1 if 'a' was inserted after 'b'
//   - 0 if 'a' and 'b' are equal, or if neither key exists in the map.
//
// If only one of the keys exists in the map, the existing key is considered
// to be "before" (less than) the non-existing key, returning -1 if 'a' exists,
// and 1 if 'b' exists.
func (im *InsertMap[K, V]) Compare(a, b K) int {
	if im.m == nil {
		return 0
	}
	ai, aok := im.m[a]
	bi, bok := im.m[b]
	if aok != bok {
		if aok {
			return -1
		}
		return 1
	}
	if !aok {
		return 0
	}
	if ai < bi {
		return -1
	}
	if ai > bi {
		return 1
	}
	return 0
}

// All returns an iterator over the key-value pairs of the map in insertion order.
func (im *InsertMap[K, V]) All() iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		im.updateKeys()
		for i, k := range im.k {
			if !yield(k, im.v[i]) {
				return
			}
		}
	}
}

// Keys returns an iterator over the keys of the map in insertion order.
func (im *InsertMap[K, V]) Keys() iter.Seq[K] {
	return func(yield func(K) bool) {
		im.updateKeys()
		for _, k := range im.k {
			if !yield(k) {
				return
			}
		}
	}
}

// Values returns an iterator over the values of the map in insertion order.
func (im *InsertMap[K, V]) Values() iter.Seq[V] {
	return func(yield func(V) bool) {
		for _, v := range im.v {
			if !yield(v) {
				return
			}
		}
	}
}

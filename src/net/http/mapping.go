// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

// A mapping is a collection of key-value pairs where the keys are unique.
// A zero mapping is empty and ready to use.
// A mapping tries to pick a representation that makes [mapping.find] most efficient.
type mapping[K comparable, V any] struct {
	s []entry[K, V] // for few pairs
	m map[K]V       // for many pairs
}

type entry[K comparable, V any] struct {
	key   K
	value V
}

// maxSlice is the maximum number of pairs for which a slice is used.
// It is a variable for benchmarking.
var maxSlice int = 8

// add adds a key-value pair to the mapping.
func (h *mapping[K, V]) add(k K, v V) {
	if h.m == nil && len(h.s) < maxSlice {
		h.s = append(h.s, entry[K, V]{k, v})
	} else {
		if h.m == nil {
			h.m = map[K]V{}
			for _, e := range h.s {
				h.m[e.key] = e.value
			}
			h.s = nil
		}
		h.m[k] = v
	}
}

// find returns the value corresponding to the given key.
// The second return value is false if there is no value
// with that key.
func (h *mapping[K, V]) find(k K) (v V, found bool) {
	if h == nil {
		return v, false
	}
	if h.m != nil {
		v, found = h.m[k]
		return v, found
	}
	for _, e := range h.s {
		if e.key == k {
			return e.value, true
		}
	}
	return v, false
}

// eachPair calls f for each pair in the mapping.
// If f returns false, pairs returns immediately.
func (h *mapping[K, V]) eachPair(f func(k K, v V) bool) {
	if h == nil {
		return
	}
	if h.m != nil {
		for k, v := range h.m {
			if !f(k, v) {
				return
			}
		}
	} else {
		for _, e := range h.s {
			if !f(e.key, e.value) {
				return
			}
		}
	}
}

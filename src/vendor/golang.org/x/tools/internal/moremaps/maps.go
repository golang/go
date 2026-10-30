// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package moremaps contains more functions for working with maps.
package moremaps

import (
	"cmp"
	"iter"
	"maps"
	"slices"
)

// Arbitrary returns an arbitrary (key, value) entry from the map and ok is true, if
// the map is not empty. Otherwise, it returns zero values for K and V, and false.
func Arbitrary[K comparable, V any](m map[K]V) (_ K, _ V, ok bool) {
	for k, v := range m {
		return k, v, true
	}
	return
}

// Group returns a new non-nil map containing the elements of s grouped by the
// keys returned from the key func.
func Group[K comparable, V any](s []V, key func(V) K) map[K][]V {
	m := make(map[K][]V)
	for _, v := range s {
		k := key(v)
		m[k] = append(m[k], v)
	}
	return m
}

// KeySlice returns the keys of the map M, like slices.Collect(maps.Keys(m)).
func KeySlice[M ~map[K]V, K comparable, V any](m M) []K {
	r := make([]K, 0, len(m))
	for k := range m {
		r = append(r, k)
	}
	return r
}

// ValueSlice returns the values of the map M, like slices.Collect(maps.Values(m)).
func ValueSlice[M ~map[K]V, K comparable, V any](m M) []V {
	r := make([]V, 0, len(m))
	for _, v := range m {
		r = append(r, v)
	}
	return r
}

// SameKeys reports whether x and y have equal sets of keys.
func SameKeys[K comparable, V1, V2 any](x map[K]V1, y map[K]V2) bool {
	ignoreValues := func(V1, V2) bool { return true }
	return maps.EqualFunc(x, y, ignoreValues)
}

// Sorted returns an iterator over the entries of m in key order.
func Sorted[M ~map[K]V, K cmp.Ordered, V any](m M) iter.Seq2[K, V] {
	// TODO(adonovan): use maps.Sorted if proposal #68598 is accepted.
	return func(yield func(K, V) bool) {
		keys := KeySlice(m)
		slices.Sort(keys)
		for _, k := range keys {
			if !yield(k, m[k]) {
				break
			}
		}
	}
}

// SortedFunc returns an iterator over the entries of m in the key order determined by cmp.
func SortedFunc[M ~map[K]V, K comparable, V any](m M, cmp func(x, y K) int) iter.Seq2[K, V] {
	// TODO(adonovan): use maps.SortedFunc if proposal #68598 is accepted.
	return func(yield func(K, V) bool) {
		keys := KeySlice(m)
		slices.SortFunc(keys, cmp)
		for _, k := range keys {
			if !yield(k, m[k]) {
				break
			}
		}
	}
}

// Delete is like delete(m, k) but reports whether deletion occurred.
func Delete[M ~map[K]V, K comparable, V any](m M, k K) bool {
	pre := len(m)
	delete(m, k)
	return pre != len(m)
}

// Entry is a key-value pair obtained from a map.
type Entry[K comparable, V any] struct {
	Key   K
	Value V
}

// Entries returns a new unordered array of the entries of a map.
func Entries[M ~map[K]V, K comparable, V any](m M) []Entry[K, V] {
	entries := make([]Entry[K, V], 0, len(m))
	for k, v := range m {
		entries = append(entries, Entry[K, V]{k, v})
	}
	return entries
}

// FromEntries returns a new map into which the entries have been inserted in order.
func FromEntries[K comparable, V any](entries []Entry[K, V]) map[K]V {
	m := make(map[K]V, len(entries))
	for _, e := range entries {
		m[e.Key] = e.Value
	}
	return m
}

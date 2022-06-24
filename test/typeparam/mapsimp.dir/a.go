// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// SliceEqual reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func SliceEqual[Elem comparable](s1, s2 []Elem) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, v1 := range s1 {
		v2 := s2[i]
		if v1 != v2 {
			isNaN := func(f Elem) bool { return f != f }
			if !isNaN(v1) || !isNaN(v2) {
				return false
			}
		}
	}
	return true
}

// Keys returns the keys of the map m.
// The keys will be an indeterminate order.
func Keys[K comparable, V any](m map[K]V) []K {
	r := make([]K, 0, len(m))
	for k := range m {
		r = append(r, k)
	}
	return r
}

// Values returns the values of the map m.
// The values will be in an indeterminate order.
func Values[K comparable, V any](m map[K]V) []V {
	r := make([]V, 0, len(m))
	for _, v := range m {
		r = append(r, v)
	}
	return r
}

// Equal reports whether two maps contain the same key/value pairs.
// Values are compared using ==.
func Equal[K, V comparable](m1, m2 map[K]V) bool {
	if len(m1) != len(m2) {
		return false
	}
	for k, v1 := range m1 {
		if v2, ok := m2[k]; !ok || v1 != v2 {
			return false
		}
	}
	return true
}

// Copy returns a copy of m.
func Copy[K comparable, V any](m map[K]V) map[K]V {
	r := make(map[K]V, len(m))
	for k, v := range m {
		r[k] = v
	}
	return r
}

// Add adds all key/value pairs in m2 to m1. Keys in m2 that are already
// present in m1 will be overwritten with the value in m2.
func Add[K comparable, V any](m1, m2 map[K]V) {
	for k, v := range m2 {
		m1[k] = v
	}
}

// Sub removes all keys in m2 from m1. Keys in m2 that are not present
// in m1 are ignored. The values in m2 are ignored.
func Sub[K comparable, V any](m1, m2 map[K]V) {
	for k := range m2 {
		delete(m1, k)
	}
}

// Intersect removes all keys from m1 that are not present in m2.
// Keys in m2 that are not in m1 are ignored. The values in m2 are ignored.
func Intersect[K comparable, V any](m1, m2 map[K]V) {
	for k := range m1 {
		if _, ok := m2[k]; !ok {
			delete(m1, k)
		}
	}
}

// Filter deletes any key/value pairs from m for which f returns false.
func Filter[K comparable, V any](m map[K]V, f func(K, V) bool) {
	for k, v := range m {
		if !f(k, v) {
			delete(m, k)
		}
	}
}

// TransformValues applies f to each value in m. The keys remain unchanged.
func TransformValues[K comparable, V any](m map[K]V, f func(V) V) {
	for k, v := range m {
		m[k] = f(v)
	}
}

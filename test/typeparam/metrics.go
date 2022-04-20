// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package metrics provides tracking arbitrary metrics composed of
// values of comparable types.
package main

import (
	"fmt"
	"sort"
	"sync"
)

// _Metric1 tracks metrics of values of some type.
type _Metric1[T comparable] struct {
	mu sync.Mutex
	m  map[T]int
}

// Add adds another instance of some value.
func (m *_Metric1[T]) Add(v T) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.m == nil {
		m.m = make(map[T]int)
	}
	m.m[v]++
}

// Count returns the number of instances we've seen of v.
func (m *_Metric1[T]) Count(v T) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.m[v]
}

// Metrics returns all the values we've seen, in an indeterminate order.
func (m *_Metric1[T]) Metrics() []T {
	return _Keys(m.m)
}

type key2[T1, T2 comparable] struct {
	f1 T1
	f2 T2
}

// _Metric2 tracks metrics of pairs of values.
type _Metric2[T1, T2 comparable] struct {
	mu sync.Mutex
	m  map[key2[T1, T2]]int
}

// Add adds another instance of some pair of values.
func (m *_Metric2[T1, T2]) Add(v1 T1, v2 T2) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.m == nil {
		m.m = make(map[key2[T1, T2]]int)
	}
	m.m[key2[T1, T2]{v1, v2}]++
}

// Count returns the number of instances we've seen of v1/v2.
func (m *_Metric2[T1, T2]) Count(v1 T1, v2 T2) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.m[key2[T1, T2]{v1, v2}]
}

// Metrics returns all the values we've seen, in an indeterminate order.
func (m *_Metric2[T1, T2]) Metrics() (r1 []T1, r2 []T2) {
	for _, k := range _Keys(m.m) {
		r1 = append(r1, k.f1)
		r2 = append(r2, k.f2)
	}
	return r1, r2
}

type key3[T1, T2, T3 comparable] struct {
	f1 T1
	f2 T2
	f3 T3
}

// _Metric3 tracks metrics of triplets of values.
type _Metric3[T1, T2, T3 comparable] struct {
	mu sync.Mutex
	m  map[key3[T1, T2, T3]]int
}

// Add adds another instance of some triplet of values.
func (m *_Metric3[T1, T2, T3]) Add(v1 T1, v2 T2, v3 T3) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.m == nil {
		m.m = make(map[key3[T1, T2, T3]]int)
	}
	m.m[key3[T1, T2, T3]{v1, v2, v3}]++
}

// Count returns the number of instances we've seen of v1/v2/v3.
func (m *_Metric3[T1, T2, T3]) Count(v1 T1, v2 T2, v3 T3) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.m[key3[T1, T2, T3]{v1, v2, v3}]
}

// Metrics returns all the values we've seen, in an indeterminate order.
func (m *_Metric3[T1, T2, T3]) Metrics() (r1 []T1, r2 []T2, r3 []T3) {
	for k := range m.m {
		r1 = append(r1, k.f1)
		r2 = append(r2, k.f2)
		r3 = append(r3, k.f3)
	}
	return r1, r2, r3
}

type S struct{ a, b, c string }

func TestMetrics() {
	m1 := _Metric1[string]{}
	if got := m1.Count("a"); got != 0 {
		panic(fmt.Sprintf("Count(%q) = %d, want 0", "a", got))
	}
	m1.Add("a")
	m1.Add("a")
	if got := m1.Count("a"); got != 2 {
		panic(fmt.Sprintf("Count(%q) = %d, want 2", "a", got))
	}
	if got, want := m1.Metrics(), []string{"a"}; !_SlicesEqual(got, want) {
		panic(fmt.Sprintf("Metrics = %v, want %v", got, want))
	}

	m2 := _Metric2[int, float64]{}
	m2.Add(1, 1)
	m2.Add(2, 2)
	m2.Add(3, 3)
	m2.Add(3, 3)
	k1, k2 := m2.Metrics()

	sort.Ints(k1)
	w1 := []int{1, 2, 3}
	if !_SlicesEqual(k1, w1) {
		panic(fmt.Sprintf("_Metric2.Metrics first slice = %v, want %v", k1, w1))
	}

	sort.Float64s(k2)
	w2 := []float64{1, 2, 3}
	if !_SlicesEqual(k2, w2) {
		panic(fmt.Sprintf("_Metric2.Metrics first slice = %v, want %v", k2, w2))
	}

	m3 := _Metric3[string, S, S]{}
	m3.Add("a", S{"d", "e", "f"}, S{"g", "h", "i"})
	m3.Add("a", S{"d", "e", "f"}, S{"g", "h", "i"})
	m3.Add("a", S{"d", "e", "f"}, S{"g", "h", "i"})
	m3.Add("b", S{"d", "e", "f"}, S{"g", "h", "i"})
	if got := m3.Count("a", S{"d", "e", "f"}, S{"g", "h", "i"}); got != 3 {
		panic(fmt.Sprintf("Count(%v, %v, %v) = %d, want 3", "a", S{"d", "e", "f"}, S{"g", "h", "i"}, got))
	}
}

func main() {
	TestMetrics()
}

// _Equal reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func _SlicesEqual[Elem comparable](s1, s2 []Elem) bool {
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

// _Keys returns the keys of the map m.
// The keys will be an indeterminate order.
func _Keys[K comparable, V any](m map[K]V) []K {
	r := make([]K, 0, len(m))
	for k := range m {
		r = append(r, k)
	}
	return r
}

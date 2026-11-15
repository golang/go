// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt_test

import (
	"logos/rt"
	"math"
	"slices"
	"testing"
)

type sortedPair struct{ A, B int }

func sortedKeys[K comparable, V any](m map[K]V) []K {
	var keys []K
	for k := range rt.SortedAll(m) {
		keys = append(keys, k)
	}
	return keys
}

func wantKeys[K comparable](t *testing.T, name string, got, want []K) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Errorf("%s: keys %v, want %v", name, got, want)
	}
}

func TestSortedAllOrdersEveryKeyKind(t *testing.T) {
	wantKeys(t, "string", sortedKeys(map[string]int{"b": 0, "a": 0, "ab": 0, "": 0, "B": 0}), []string{"", "B", "a", "ab", "b"})
	wantKeys(t, "int", sortedKeys(map[int]int{3: 0, -1: 0, 10: 0, -20: 0, 0: 0}), []int{-20, -1, 0, 3, 10})
	wantKeys(t, "int8", sortedKeys(map[int8]int{100: 0, -128: 0, 1: 0}), []int8{-128, 1, 100})
	wantKeys(t, "uint8", sortedKeys(map[uint8]int{200: 0, 3: 0, 100: 0}), []uint8{3, 100, 200})
	wantKeys(t, "float64", sortedKeys(map[float64]int{2.5: 0, -1.5: 0, 10: 0, 0.25: 0, -100: 0}), []float64{-100, -1.5, 0.25, 2.5, 10})
	wantKeys(t, "bool", sortedKeys(map[bool]int{true: 0, false: 0}), []bool{false, true})
	wantKeys(t, "complex128", sortedKeys(map[complex128]int{1 + 2i: 0, 1 + 1i: 0, 5i: 0, -1: 0}), []complex128{-1, 5i, 1 + 1i, 1 + 2i})
	wantKeys(t, "struct", sortedKeys(map[sortedPair]int{{2, 1}: 0, {1, 5}: 0, {1, 2}: 0, {3, 0}: 0}), []sortedPair{{1, 2}, {1, 5}, {2, 1}, {3, 0}})
	wantKeys(t, "array", sortedKeys(map[[2]int]int{{2, 1}: 0, {1, 5}: 0, {1, 2}: 0}), [][2]int{{1, 2}, {1, 5}, {2, 1}})
}

func TestSortedAllYieldsValuesWithKeys(t *testing.T) {
	m := map[string]int{"c": 3, "a": 1, "b": 2}
	var got []int
	for k, v := range rt.SortedAll(m) {
		if m[k] != v {
			t.Fatalf("key %q yielded value %d, want %d", k, v, m[k])
		}
		got = append(got, v)
	}
	wantKeys(t, "values", got, []int{1, 2, 3})
}

func TestSortedAllStopsWhenYieldReturnsFalse(t *testing.T) {
	count := 0
	for range rt.SortedAll(map[int]int{1: 0, 2: 0, 3: 0}) {
		count++
		break
	}
	if count != 1 {
		t.Fatalf("visited %d entries after break, want 1", count)
	}
}

func TestSortedAllRetainsCapturedEntries(t *testing.T) {
	m := map[int]int{1: 0, 2: 0, 3: 0}
	var got []int
	for k := range rt.SortedAll(m) {
		got = append(got, k)
		delete(m, 3)
	}
	wantKeys(t, "captured before delete", got, []int{1, 2, 3})
}

func TestSortedAllOfNilMapVisitsNothing(t *testing.T) {
	var m map[string]int
	if keys := sortedKeys(m); len(keys) != 0 {
		t.Fatalf("nil map visited %v", keys)
	}
}

func TestSortedAllMixedDynamicTypes(t *testing.T) {
	m := map[any]int{1: 1, 2: 2, "a": 3, "b": 4, nil: 5, [1]int{1}: 6, [2]int{1, 2}: 7, struct{ A int }{1}: 8, struct{ B string }{"b"}: 9}
	for range 64 {
		seen := make(map[any]bool)
		ints, strs := []int{}, []string{}
		for k, v := range rt.SortedAll(m) {
			if seen[k] || m[k] != v {
				t.Fatalf("duplicate or mismatched entry %v: %d", k, v)
			}
			seen[k] = true
			switch k := k.(type) {
			case int:
				ints = append(ints, k)
			case string:
				strs = append(strs, k)
			}
		}
		if len(seen) != len(m) {
			t.Fatalf("visited %d entries, want %d", len(seen), len(m))
		}
		wantKeys(t, "dynamic ints", ints, []int{1, 2})
		wantKeys(t, "dynamic strings", strs, []string{"a", "b"})
	}
}

func TestSortedAllRetainsNaNEntries(t *testing.T) {
	nan := math.NaN()
	m := map[float64]int{1: 3}
	m[nan] = 1
	m[nan] = 2
	seen := make(map[int]bool)
	nans := 0
	for k, v := range rt.SortedAll(m) {
		if seen[v] || v < 1 || v > 3 {
			t.Fatalf("duplicate or unexpected value %d", v)
		}
		seen[v] = true
		if math.IsNaN(k) {
			nans++
			if v == 3 {
				t.Fatal("NaN key lost its value")
			}
		} else if k != 1 || v != 3 {
			t.Fatalf("unexpected finite entry %v: %d", k, v)
		}
	}
	if len(seen) != 3 || nans != 2 {
		t.Fatalf("visited %d entries and %d NaNs, want 3 and 2", len(seen), nans)
	}
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt_test

import (
	"cmp"
	"iter"
	"logos/rt"
	"math"
	"slices"
	"testing"
)

type sortedPair struct{ A, B int }

type sortedName string

type sortedFlag bool

func sortedKeys[K comparable, V any](seq iter.Seq2[K, V]) []K {
	var keys []K
	for k := range seq {
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

func eachRange[K cmp.Ordered, V any](t *testing.T, check func(t *testing.T, ranged func(map[K]V) iter.Seq2[K, V])) {
	t.Helper()
	t.Run("compare", func(t *testing.T) {
		check(t, func(m map[K]V) iter.Seq2[K, V] { return rt.SortedAll(m, rt.CompareOrdered) })
	})
	t.Run("dynamic", func(t *testing.T) {
		check(t, rt.SortedAllDynamic[K, V])
	})
}

func TestSortedAllOrdersEveryKeyKind(t *testing.T) {
	wantKeys(t, "string", sortedKeys(rt.SortedAll(map[string]int{"b": 0, "a": 0, "ab": 0, "": 0, "B": 0}, rt.CompareOrdered)), []string{"", "B", "a", "ab", "b"})
	wantKeys(t, "int", sortedKeys(rt.SortedAll(map[int]int{3: 0, -1: 0, 10: 0, -20: 0, 0: 0}, rt.CompareOrdered)), []int{-20, -1, 0, 3, 10})
	wantKeys(t, "int8", sortedKeys(rt.SortedAll(map[int8]int{100: 0, -128: 0, 1: 0}, rt.CompareOrdered)), []int8{-128, 1, 100})
	wantKeys(t, "uint8", sortedKeys(rt.SortedAll(map[uint8]int{200: 0, 3: 0, 100: 0}, rt.CompareOrdered)), []uint8{3, 100, 200})
	wantKeys(t, "float64", sortedKeys(rt.SortedAll(map[float64]int{2.5: 0, -1.5: 0, 10: 0, 0.25: 0, -100: 0}, rt.CompareOrdered)), []float64{-100, -1.5, 0.25, 2.5, 10})
	wantKeys(t, "bool", sortedKeys(rt.SortedAll(map[bool]int{true: 0, false: 0}, rt.CompareBool)), []bool{false, true})
	wantKeys(t, "complex128", sortedKeys(rt.SortedAll(map[complex128]int{1 + 2i: 0, 1 + 1i: 0, 5i: 0, -1: 0}, rt.CompareComplex)), []complex128{-1, 5i, 1 + 1i, 1 + 2i})
}

func TestSortedAllDynamicOrdersEveryKeyKind(t *testing.T) {
	wantKeys(t, "string", sortedKeys(rt.SortedAllDynamic(map[string]int{"b": 0, "a": 0, "ab": 0, "": 0, "B": 0})), []string{"", "B", "a", "ab", "b"})
	wantKeys(t, "int", sortedKeys(rt.SortedAllDynamic(map[int]int{3: 0, -1: 0, 10: 0, -20: 0, 0: 0})), []int{-20, -1, 0, 3, 10})
	wantKeys(t, "int8", sortedKeys(rt.SortedAllDynamic(map[int8]int{100: 0, -128: 0, 1: 0})), []int8{-128, 1, 100})
	wantKeys(t, "uint8", sortedKeys(rt.SortedAllDynamic(map[uint8]int{200: 0, 3: 0, 100: 0})), []uint8{3, 100, 200})
	wantKeys(t, "float64", sortedKeys(rt.SortedAllDynamic(map[float64]int{2.5: 0, -1.5: 0, 10: 0, 0.25: 0, -100: 0})), []float64{-100, -1.5, 0.25, 2.5, 10})
	wantKeys(t, "bool", sortedKeys(rt.SortedAllDynamic(map[bool]int{true: 0, false: 0})), []bool{false, true})
	wantKeys(t, "complex128", sortedKeys(rt.SortedAllDynamic(map[complex128]int{1 + 2i: 0, 1 + 1i: 0, 5i: 0, -1: 0})), []complex128{-1, 5i, 1 + 1i, 1 + 2i})
	wantKeys(t, "struct", sortedKeys(rt.SortedAllDynamic(map[sortedPair]int{{2, 1}: 0, {1, 5}: 0, {1, 2}: 0, {3, 0}: 0})), []sortedPair{{1, 2}, {1, 5}, {2, 1}, {3, 0}})
	wantKeys(t, "array", sortedKeys(rt.SortedAllDynamic(map[[2]int]int{{2, 1}: 0, {1, 5}: 0, {1, 2}: 0})), [][2]int{{1, 2}, {1, 5}, {2, 1}})
}

func TestCompareOrderedMatchesCmpCompare(t *testing.T) {
	nan := math.NaN()
	negZero := math.Copysign(0, -1)
	for _, tc := range []struct {
		a, b float64
		want int
	}{
		{1, 2, -1}, {2, 1, 1}, {1, 1, 0}, {nan, 1, -1}, {1, nan, 1}, {nan, nan, 0}, {negZero, 0, 0},
	} {
		if got := rt.CompareOrdered(tc.a, tc.b); got != tc.want {
			t.Errorf("CompareOrdered(%v, %v) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
	if got := rt.CompareOrdered(sortedName("a"), sortedName("b")); got != -1 {
		t.Errorf("CompareOrdered on a named string = %d, want -1", got)
	}
	if got := rt.CompareOrdered(uint64(1)<<63, uint64(1)); got != 1 {
		t.Errorf("CompareOrdered on a large unsigned value = %d, want 1", got)
	}
}

func TestCompareBoolOrdersFalseFirst(t *testing.T) {
	for _, tc := range []struct {
		a, b bool
		want int
	}{
		{false, true, -1}, {true, false, 1}, {false, false, 0}, {true, true, 0},
	} {
		if got := rt.CompareBool(tc.a, tc.b); got != tc.want {
			t.Errorf("CompareBool(%v, %v) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
	if got := rt.CompareBool(sortedFlag(false), sortedFlag(true)); got != -1 {
		t.Errorf("CompareBool on a named bool = %d, want -1", got)
	}
}

func TestCompareComplexOrdersRealThenImaginary(t *testing.T) {
	nan := math.NaN()
	for _, tc := range []struct {
		a, b complex128
		want int
	}{
		{1 + 5i, 2 + 1i, -1}, {2 + 1i, 1 + 5i, 1}, {1 + 1i, 1 + 2i, -1}, {1 + 2i, 1 + 1i, 1}, {1 + 1i, 1 + 1i, 0},
		{complex(nan, 1), 0, -1}, {complex(1, nan), complex(1, 0), -1},
	} {
		if got := rt.CompareComplex(tc.a, tc.b); got != tc.want {
			t.Errorf("CompareComplex(%v, %v) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
	if got := rt.CompareComplex(complex64(1+2i), complex64(1+3i)); got != -1 {
		t.Errorf("CompareComplex on complex64 = %d, want -1", got)
	}
}

func TestSortedAllYieldsValuesWithKeys(t *testing.T) {
	eachRange(t, func(t *testing.T, ranged func(map[string]int) iter.Seq2[string, int]) {
		m := map[string]int{"c": 3, "a": 1, "b": 2}
		var got []int
		for k, v := range ranged(m) {
			if m[k] != v {
				t.Fatalf("key %q yielded value %d, want %d", k, v, m[k])
			}
			got = append(got, v)
		}
		wantKeys(t, "values", got, []int{1, 2, 3})
	})
}

func TestSortedAllStopsWhenYieldReturnsFalse(t *testing.T) {
	eachRange(t, func(t *testing.T, ranged func(map[string]int) iter.Seq2[string, int]) {
		count := 0
		for range ranged(map[string]int{"a": 0, "b": 0, "c": 0}) {
			count++
			break
		}
		if count != 1 {
			t.Fatalf("visited %d entries after break, want 1", count)
		}
	})
}

func TestSortedAllRetainsCapturedEntries(t *testing.T) {
	eachRange(t, func(t *testing.T, ranged func(map[string]int) iter.Seq2[string, int]) {
		m := map[string]int{"a": 0, "b": 0, "c": 0}
		var got []string
		for k := range ranged(m) {
			got = append(got, k)
			delete(m, "c")
		}
		wantKeys(t, "captured before delete", got, []string{"a", "b", "c"})
	})
}

func TestSortedAllOfNilMapVisitsNothing(t *testing.T) {
	eachRange(t, func(t *testing.T, ranged func(map[string]int) iter.Seq2[string, int]) {
		var m map[string]int
		if keys := sortedKeys(ranged(m)); len(keys) != 0 {
			t.Fatalf("nil map visited %v", keys)
		}
	})
}

func TestSortedAllNilCompareKeepsEveryEntry(t *testing.T) {
	a, b, c := 1, 2, 3
	m := map[*int]int{&a: 10, &b: 20, &c: 30}
	seen := make(map[*int]int)
	for k, v := range rt.SortedAll(m, nil) {
		if _, dup := seen[k]; dup {
			t.Fatalf("pointer key %p visited twice", k)
		}
		seen[k] = v
	}
	if len(seen) != len(m) {
		t.Fatalf("visited %d entries, want %d", len(seen), len(m))
	}
	for k, v := range m {
		if seen[k] != v {
			t.Fatalf("pointer key %p yielded value %d, want %d", k, seen[k], v)
		}
	}
}

func TestSortedAllMixedDynamicTypes(t *testing.T) {
	m := map[any]int{1: 1, 2: 2, "a": 3, "b": 4, nil: 5, [1]int{1}: 6, [2]int{1, 2}: 7, struct{ A int }{1}: 8, struct{ B string }{"b"}: 9}
	for range 64 {
		seen := make(map[any]bool)
		ints, strs := []int{}, []string{}
		for k, v := range rt.SortedAllDynamic(m) {
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
	eachRange(t, func(t *testing.T, ranged func(map[float64]int) iter.Seq2[float64, int]) {
		nan := math.NaN()
		m := map[float64]int{1: 3}
		m[nan] = 1
		m[nan] = 2
		seen := make(map[int]bool)
		nans := 0
		for k, v := range ranged(m) {
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
	})
}

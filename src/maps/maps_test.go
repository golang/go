// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps

import (
	"math"
	"strconv"
	"testing"
)

var m1 = map[int]int{1: 2, 2: 4, 4: 8, 8: 16}
var m2 = map[int]string{1: "2", 2: "4", 4: "8", 8: "16"}

func TestEqual(t *testing.T) {
	if !Equal(m1, m1) {
		t.Errorf("Equal(%v, %v) = false, want true", m1, m1)
	}
	if Equal(m1, (map[int]int)(nil)) {
		t.Errorf("Equal(%v, nil) = true, want false", m1)
	}
	if Equal((map[int]int)(nil), m1) {
		t.Errorf("Equal(nil, %v) = true, want false", m1)
	}
	if !Equal[map[int]int, map[int]int](nil, nil) {
		t.Error("Equal(nil, nil) = false, want true")
	}
	if ms := map[int]int{1: 2}; Equal(m1, ms) {
		t.Errorf("Equal(%v, %v) = true, want false", m1, ms)
	}

	// Comparing NaN for equality is expected to fail.
	mf := map[int]float64{1: 0, 2: math.NaN()}
	if Equal(mf, mf) {
		t.Errorf("Equal(%v, %v) = true, want false", mf, mf)
	}
}

// equal is simply ==.
func equal[T comparable](v1, v2 T) bool {
	return v1 == v2
}

// equalNaN is like == except that all NaNs are equal.
func equalNaN[T comparable](v1, v2 T) bool {
	isNaN := func(f T) bool { return f != f }
	return v1 == v2 || (isNaN(v1) && isNaN(v2))
}

// equalIntStr compares ints and strings.
func equalIntStr(v1 int, v2 string) bool {
	return strconv.Itoa(v1) == v2
}

func TestEqualFunc(t *testing.T) {
	if !EqualFunc(m1, m1, equal[int]) {
		t.Errorf("EqualFunc(%v, %v, equal) = false, want true", m1, m1)
	}
	if EqualFunc(m1, (map[int]int)(nil), equal[int]) {
		t.Errorf("EqualFunc(%v, nil, equal) = true, want false", m1)
	}
	if EqualFunc((map[int]int)(nil), m1, equal[int]) {
		t.Errorf("EqualFunc(nil, %v, equal) = true, want false", m1)
	}
	if !EqualFunc[map[int]int, map[int]int](nil, nil, equal[int]) {
		t.Error("EqualFunc(nil, nil, equal) = false, want true")
	}
	if ms := map[int]int{1: 2}; EqualFunc(m1, ms, equal[int]) {
		t.Errorf("EqualFunc(%v, %v, equal) = true, want false", m1, ms)
	}

	// Comparing NaN for equality is expected to fail.
	mf := map[int]float64{1: 0, 2: math.NaN()}
	if EqualFunc(mf, mf, equal[float64]) {
		t.Errorf("EqualFunc(%v, %v, equal) = true, want false", mf, mf)
	}
	// But it should succeed using equalNaN.
	if !EqualFunc(mf, mf, equalNaN[float64]) {
		t.Errorf("EqualFunc(%v, %v, equalNaN) = false, want true", mf, mf)
	}

	if !EqualFunc(m1, m2, equalIntStr) {
		t.Errorf("EqualFunc(%v, %v, equalIntStr) = false, want true", m1, m2)
	}
}

func TestClone(t *testing.T) {
	mc := Clone(m1)
	if !Equal(mc, m1) {
		t.Errorf("Clone(%v) = %v, want %v", m1, mc, m1)
	}
	mc[16] = 32
	if Equal(mc, m1) {
		t.Errorf("Equal(%v, %v) = true, want false", mc, m1)
	}
}

func TestCloneNil(t *testing.T) {
	var m1 map[string]int
	mc := Clone(m1)
	if mc != nil {
		t.Errorf("Clone(%v) = %v, want %v", m1, mc, m1)
	}
}

func TestCopy(t *testing.T) {
	mc := Clone(m1)
	Copy(mc, mc)
	if !Equal(mc, m1) {
		t.Errorf("Copy(%v, %v) = %v, want %v", m1, m1, mc, m1)
	}
	Copy(mc, map[int]int{16: 32})
	want := map[int]int{1: 2, 2: 4, 4: 8, 8: 16, 16: 32}
	if !Equal(mc, want) {
		t.Errorf("Copy result = %v, want %v", mc, want)
	}

	type M1 map[int]bool
	type M2 map[int]bool
	Copy(make(M1), make(M2))
}

// rangeMap returns a map with keys [lo, hi), each mapped to k+off.
func rangeMap(lo, hi, off int) map[int]int {
	m := make(map[int]int)
	for k := lo; k < hi; k++ {
		m[k] = k + off
	}
	return m
}

// Copy presizes dst when src is larger; the result must not depend on
// how the keys overlap.
func TestCopyPresize(t *testing.T) {
	const srcOff = 1_000_000
	tests := []struct {
		name         string
		dstLo, dstHi int
		srcLo, srcHi int
	}{
		{"empty dst", 0, 0, 0, 1000},
		{"small dst, disjoint", 0, 10, 10, 1010},
		{"src superset", 0, 10, 0, 1000},
		{"src subset", 0, 1000, 0, 10},
		{"identical keys", 0, 1000, 0, 1000},
		{"disjoint, equal size", 0, 500, 500, 1000},
		{"partial overlap", 0, 600, 300, 1300},
		{"small maps", 0, 3, 3, 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := rangeMap(tt.dstLo, tt.dstHi, 0)
			src := rangeMap(tt.srcLo, tt.srcHi, srcOff)
			want := Clone(dst)
			for k, v := range src {
				want[k] = v
			}
			Copy(dst, src)
			if !Equal(dst, want) {
				t.Errorf("Copy result has len %d, want len %d with src values winning", len(dst), len(want))
			}
		})
	}
}

// Copy presizes dst, which must not disturb an iteration over dst:
// NaN keys cannot be looked up, so each must stay on the same side of
// the iterator.
func TestCopyNaNDuringRange(t *testing.T) {
	const nans = 64
	dst := make(map[float64]int)
	for i := range nans {
		dst[math.NaN()] = i
	}
	for i := range 5000 {
		dst[float64(-1-i)] = -1
	}
	src := make(map[float64]int)
	for i := range 20000 {
		src[float64(i)] = -1
	}

	var seen [nans]int
	copied := false
	for k, v := range dst {
		if !copied {
			copied = true
			Copy(dst, src)
		}
		if k != k {
			seen[v]++
		}
	}
	for i, n := range seen {
		if n != 1 {
			t.Errorf("NaN %d seen %d times, want 1", i, n)
		}
	}
}

func TestCopyNilDst(t *testing.T) {
	// Nothing to copy: no panic, as before presizing.
	Copy(map[int]int(nil), map[int]int{})

	defer func() {
		if recover() == nil {
			t.Error("Copy(nil, non-empty) did not panic")
		}
	}()
	Copy(map[int]int(nil), map[int]int{1: 2})
}

func BenchmarkCopy(b *testing.B) {
	for _, n := range []int{4, 64, 1024, 65536} {
		for _, bm := range []struct {
			name     string
			dst, src map[int]int
		}{
			// Copy presizes: len(src) > len(dst).
			{"empty", nil, rangeMap(0, n, 0)},
			{"small+disjoint", rangeMap(0, 4, 0), rangeMap(4, n+4, 0)},
			{"half-overlap", rangeMap(0, n/2, 0), rangeMap(0, n, 0)},
			// No presizing: len(src) <= len(dst).
			{"identical", rangeMap(0, n, 0), rangeMap(0, n, 0)},
			{"disjoint-equal", rangeMap(0, n, 0), rangeMap(n, 2*n, 0)},
		} {
			b.Run(bm.name+"/n="+strconv.Itoa(n), func(b *testing.B) {
				// Build fresh destinations in batches outside the
				// timer: stopping it per iteration costs far more
				// than a small Copy.
				dsts := make([]map[int]int, max(1, min(1000, (1<<20)/n)))
				fill := func() {
					b.StopTimer()
					for i := range dsts {
						if bm.dst == nil {
							dsts[i] = make(map[int]int)
						} else {
							dsts[i] = Clone(bm.dst)
						}
					}
					b.StartTimer()
				}
				fill()
				for i := 0; i < b.N; i++ {
					j := i % len(dsts)
					if j == 0 && i > 0 {
						fill()
					}
					Copy(dsts[j], bm.src)
				}
			})
		}
	}
}

func TestDeleteFunc(t *testing.T) {
	mc := Clone(m1)
	DeleteFunc(mc, func(int, int) bool { return false })
	if !Equal(mc, m1) {
		t.Errorf("DeleteFunc(%v, true) = %v, want %v", m1, mc, m1)
	}
	DeleteFunc(mc, func(k, v int) bool { return k > 3 })
	want := map[int]int{1: 2, 2: 4}
	if !Equal(mc, want) {
		t.Errorf("DeleteFunc result = %v, want %v", mc, want)
	}
}

var n map[int]int

func BenchmarkMapClone(b *testing.B) {
	var m = make(map[int]int)
	for i := 0; i < 1000000; i++ {
		m[i] = i
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n = Clone(m)
	}
}

func TestCloneWithDelete(t *testing.T) {
	var m = make(map[int]int)
	for i := 0; i < 32; i++ {
		m[i] = i
	}
	for i := 8; i < 32; i++ {
		delete(m, i)
	}
	m2 := Clone(m)
	if len(m2) != 8 {
		t.Errorf("len2(m2) = %d, want %d", len(m2), 8)
	}
	for i := 0; i < 8; i++ {
		if m2[i] != m[i] {
			t.Errorf("m2[%d] = %d, want %d", i, m2[i], m[i])
		}
	}
}

func TestCloneWithMapAssign(t *testing.T) {
	var m = make(map[int]int)
	const N = 25
	for i := 0; i < N; i++ {
		m[i] = i
	}
	m2 := Clone(m)
	if len(m2) != N {
		t.Errorf("len2(m2) = %d, want %d", len(m2), N)
	}
	for i := 0; i < N; i++ {
		if m2[i] != m[i] {
			t.Errorf("m2[%d] = %d, want %d", i, m2[i], m[i])
		}
	}
}

func TestCloneLarge(t *testing.T) {
	// See issue 64474.
	type K [17]float64 // > 128 bytes
	type V [17]float64

	var zero float64
	negZero := -zero

	for tst := 0; tst < 3; tst++ {
		// Initialize m with a key and value.
		m := map[K]V{}
		var k1 K
		var v1 V
		m[k1] = v1

		switch tst {
		case 0: // nothing, just a 1-entry map
		case 1:
			// Add more entries to make it 2 buckets
			// 1 entry already
			// 7 more fill up 1 bucket
			// 1 more to grow to 2 buckets
			for i := 0; i < 7+1; i++ {
				m[K{float64(i) + 1}] = V{}
			}
		case 2:
			// Capture the map mid-grow
			// 1 entry already
			// 7 more fill up 1 bucket
			// 5 more (13 total) fill up 2 buckets
			// 13 more (26 total) fill up 4 buckets
			// 1 more to start the 4->8 bucket grow
			for i := 0; i < 7+5+13+1; i++ {
				m[K{float64(i) + 1}] = V{}
			}
		}

		// Clone m, which should freeze the map's contents.
		c := Clone(m)

		// Update m with new key and value.
		k2, v2 := k1, v1
		k2[0] = negZero
		v2[0] = 1.0
		m[k2] = v2

		// Make sure c still has its old key and value.
		for k, v := range c {
			if math.Signbit(k[0]) {
				t.Errorf("tst%d: sign bit of key changed; got %v want %v", tst, k, k1)
			}
			if v != v1 {
				t.Errorf("tst%d: value changed; got %v want %v", tst, v, v1)
			}
		}
	}
}

func TestGrow(t *testing.T) {
	m := map[int]int{1: 2, 2: 4}
	Grow(m, 1000)

	if !Equal(m, map[int]int{1: 2, 2: 4}) {
		t.Errorf("Grow changed the contents of m: got %v", m)
	}

	for i := 0; i < 1000; i++ {
		m[i] = i
	}
	if len(m) != 1000 {
		t.Errorf("len(m) after inserts got %d want 1000", len(m))
	}
	for i := 0; i < 1000; i++ {
		if got := m[i]; got != i {
			t.Errorf("m[%d] got %d want %d", i, got, i)
		}
	}
}

func TestGrowZero(t *testing.T) {
	m := map[int]int{1: 2}
	Grow(m, 0)
	if !Equal(m, map[int]int{1: 2}) {
		t.Errorf("Grow(m, 0) changed the contents of m: got %v", m)
	}

	// Growing a nil map by zero must not panic.
	var nilMap map[int]int
	Grow(nilMap, 0)
}

func TestGrowNegative(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Grow(m, -1) did not panic")
		}
	}()
	Grow(map[int]int{}, -1)
}

func TestGrowNil(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Grow(nil, 1) did not panic")
		}
	}()
	var m map[int]int
	Grow(m, 1)
}

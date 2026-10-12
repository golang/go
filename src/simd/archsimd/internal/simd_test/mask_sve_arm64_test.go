// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

// SVE mask accessor tests. A mask is built from a uint64 of predicate bits
// through its memory representation (see maskBits), and each accessor is
// checked against a lane-by-lane reference over a set of lane patterns.

package simd_test

import (
	"math"
	"simd/archsimd"
	"testing"
	"unsafe"
)

type sveMask interface {
	archsimd.Mask8s | archsimd.Mask16s | archsimd.Mask32s | archsimd.Mask64s
}

// maskFromBits returns the mask whose memory representation is bits.
func maskFromBits[M sveMask](bits uint64) M {
	return *(*M)(unsafe.Pointer(&bits))
}

// lanesOf returns the n lanes of bits for lanes laneBytes wide.
func lanesOf(bits uint64, laneBytes, n int) []bool {
	lanes := make([]bool, n)
	for i := range lanes {
		lanes[i] = bits>>(i*laneBytes)&1 != 0
	}
	return lanes
}

// bitsOf is the inverse of lanesOf, with the bits between lanes zero.
func bitsOf(lanes []bool, laneBytes int) uint64 {
	var bits uint64
	for i, l := range lanes {
		if l {
			bits |= 1 << (i * laneBytes)
		}
	}
	return bits
}

// maskPatterns returns lane patterns to test for n lanes laneBytes wide, as
// predicate bits: none, all, each single lane, each prefix and suffix, and
// alternating lanes. For lanes wider than a byte it adds the same patterns
// with every bit between lanes set, bits a mask ignores.
func maskPatterns(laneBytes, n int) []uint64 {
	var lanes [][]bool
	add := func(f func(i int) bool) {
		l := make([]bool, n)
		for i := range l {
			l[i] = f(i)
		}
		lanes = append(lanes, l)
	}
	add(func(int) bool { return false })
	add(func(int) bool { return true })
	add(func(i int) bool { return i%2 == 0 })
	add(func(i int) bool { return i%2 == 1 })
	for k := 0; k < n; k++ {
		add(func(i int) bool { return i == k })
		add(func(i int) bool { return i < k })
		add(func(i int) bool { return i >= k })
	}
	var pats []uint64
	for _, l := range lanes {
		pats = append(pats, bitsOf(l, laneBytes))
	}
	if laneBytes > 1 {
		var between uint64
		for i := range n * laneBytes {
			if i%laneBytes != 0 {
				between |= 1 << i
			}
		}
		for _, l := range lanes {
			pats = append(pats, bitsOf(l, laneBytes)|between)
		}
	}
	return pats
}

// testMaskToMask checks a mask-to-mask accessor against want, which maps
// input lanes to output lanes. The result must have only lane bits set.
func testMaskToMask[M sveMask](t *testing.T, name string, laneBytes int, op func(M) M, want func([]bool) []bool) {
	t.Helper()
	n := archsimd.Int8s{}.Len() / laneBytes
	for _, bits := range maskPatterns(laneBytes, n) {
		got := maskBits(op(maskFromBits[M](bits)))
		if w := bitsOf(want(lanesOf(bits, laneBytes, n)), laneBytes); got != w {
			t.Errorf("%s(%#x) = %#x, want %#x", name, bits, got, w)
		}
	}
}

// testMaskToBool checks a mask-to-bool accessor against want.
func testMaskToBool[M sveMask](t *testing.T, name string, laneBytes int, op func(M) bool, want func([]bool) bool) {
	t.Helper()
	n := archsimd.Int8s{}.Len() / laneBytes
	for _, bits := range maskPatterns(laneBytes, n) {
		if got, w := op(maskFromBits[M](bits)), want(lanesOf(bits, laneBytes, n)); got != w {
			t.Errorf("%s(%#x) = %v, want %v", name, bits, got, w)
		}
	}
}

func firstWant(lanes []bool) []bool {
	out := make([]bool, len(lanes))
	for i, l := range lanes {
		if l {
			out[i] = true
			break
		}
	}
	return out
}

func TestMaskFirstSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testMaskToMask(t, "Mask8s.First", 1, archsimd.Mask8s.First, firstWant)
	testMaskToMask(t, "Mask16s.First", 2, archsimd.Mask16s.First, firstWant)
	testMaskToMask(t, "Mask32s.First", 4, archsimd.Mask32s.First, firstWant)
	testMaskToMask(t, "Mask64s.First", 8, archsimd.Mask64s.First, firstWant)
}

func nextWant(lanes []bool) []bool {
	last := -1
	for i, l := range lanes {
		if l {
			last = i
		}
	}
	out := make([]bool, len(lanes))
	if last+1 < len(lanes) {
		out[last+1] = true
	}
	return out
}

func TestMaskNextSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testMaskToMask(t, "Mask8s.Next", 1, archsimd.Mask8s.Next, nextWant)
	testMaskToMask(t, "Mask16s.Next", 2, archsimd.Mask16s.Next, nextWant)
	testMaskToMask(t, "Mask32s.Next", 4, archsimd.Mask32s.Next, nextWant)
	testMaskToMask(t, "Mask64s.Next", 8, archsimd.Mask64s.Next, nextWant)
}

func allWant(lanes []bool) bool {
	for _, l := range lanes {
		if !l {
			return false
		}
	}
	return true
}

func TestMaskAllSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testMaskToBool(t, "Mask8s.All", 1, archsimd.Mask8s.All, allWant)
	testMaskToBool(t, "Mask16s.All", 2, archsimd.Mask16s.All, allWant)
	testMaskToBool(t, "Mask32s.All", 4, archsimd.Mask32s.All, allWant)
	testMaskToBool(t, "Mask64s.All", 8, archsimd.Mask64s.All, allWant)
}

func noneWant(lanes []bool) bool {
	for _, l := range lanes {
		if l {
			return false
		}
	}
	return true
}

func TestMaskNoneSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testMaskToBool(t, "Mask8s.None", 1, archsimd.Mask8s.None, noneWant)
	testMaskToBool(t, "Mask16s.None", 2, archsimd.Mask16s.None, noneWant)
	testMaskToBool(t, "Mask32s.None", 4, archsimd.Mask32s.None, noneWant)
	testMaskToBool(t, "Mask64s.None", 8, archsimd.Mask64s.None, noneWant)
}

func anyWant(lanes []bool) bool { return !noneWant(lanes) }

func TestMaskAnySVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testMaskToBool(t, "Mask8s.Any", 1, archsimd.Mask8s.Any, anyWant)
	testMaskToBool(t, "Mask16s.Any", 2, archsimd.Mask16s.Any, anyWant)
	testMaskToBool(t, "Mask32s.Any", 4, archsimd.Mask32s.Any, anyWant)
	testMaskToBool(t, "Mask64s.Any", 8, archsimd.Mask64s.Any, anyWant)
}

func testMaskToIntAndBack[M sveMask, T signed, V any](t *testing.T, name string, laneBytes int,
	toInt func(M) V, toMask func(V) M, store func(V, []T), load func([]T) V) {
	t.Helper()
	n := archsimd.Int8s{}.Len() / laneBytes
	for _, bits := range maskPatterns(laneBytes, n) {
		m := maskFromBits[M](bits)
		v := toInt(m)
		got := make([]T, n)
		store(v, got)
		lanes := lanesOf(bits, laneBytes, n)
		for i, l := range lanes {
			var want T
			if l {
				want = -1
			}
			if got[i] != want {
				t.Errorf("%s.ToInt(%#x)[%d] = %d, want %d", name, bits, i, got[i], want)
			}
		}
		if gotBits, wantBits := maskBits(toMask(v)), bitsOf(lanes, laneBytes); gotBits != wantBits {
			t.Errorf("%s.ToInt(%#x).ToMask() = %#x, want %#x", name, bits, gotBits, wantBits)
		}
		// Check non-canonical non-zero values for ToMask.
		for i := range got {
			if got[i] != 0 {
				got[i] = T(i + 1)
			}
		}
		if gotBits, wantBits := maskBits(toMask(load(got))), bitsOf(lanes, laneBytes); gotBits != wantBits {
			t.Errorf("%s non-canonical ToMask(%#x) = %#x, want %#x", name, bits, gotBits, wantBits)
		}
	}
}

func TestMaskToIntAndToMaskSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testMaskToIntAndBack(t, "Mask8s", 1, archsimd.Mask8s.ToInt8s, archsimd.Int8s.ToMask, archsimd.Int8s.Store, archsimd.LoadInt8s)
	testMaskToIntAndBack(t, "Mask16s", 2, archsimd.Mask16s.ToInt16s, archsimd.Int16s.ToMask, archsimd.Int16s.Store, archsimd.LoadInt16s)
	testMaskToIntAndBack(t, "Mask32s", 4, archsimd.Mask32s.ToInt32s, archsimd.Int32s.ToMask, archsimd.Int32s.Store, archsimd.LoadInt32s)
	testMaskToIntAndBack(t, "Mask64s", 8, archsimd.Mask64s.ToInt64s, archsimd.Int64s.ToMask, archsimd.Int64s.Store, archsimd.LoadInt64s)
}

// testMaskFromCount checks a FromCount constructor against the prefix of
// lanes it should activate, through the mask's memory representation.
func testMaskFromCount[M sveMask](t *testing.T, name string, laneBytes, n int, fromCount func(int) M) {
	t.Helper()
	for _, c := range []int{math.MinInt, -1, 0, 1, n / 2, n - 1, n, n + 1, math.MaxInt} {
		m := fromCount(c)
		got := *(*uint64)(unsafe.Pointer(&m))
		lanes := make([]bool, n)
		for i := range lanes {
			lanes[i] = i < c
		}
		if want := bitsOf(lanes, laneBytes); got != want {
			t.Errorf("%s(%d) = %#x, want %#x", name, c, got, want)
		}
	}
}

func TestMaskFromCountSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testMaskFromCount(t, "Mask8sFromCount", 1, archsimd.Int8s{}.Len(), archsimd.Mask8sFromCount)
	testMaskFromCount(t, "Mask16sFromCount", 2, archsimd.Int16s{}.Len(), archsimd.Mask16sFromCount)
	testMaskFromCount(t, "Mask32sFromCount", 4, archsimd.Int32s{}.Len(), archsimd.Mask32sFromCount)
	testMaskFromCount(t, "Mask64sFromCount", 8, archsimd.Int64s{}.Len(), archsimd.Mask64sFromCount)
}

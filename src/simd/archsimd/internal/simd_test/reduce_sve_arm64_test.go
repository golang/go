// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"math"
	"simd/archsimd"
	"testing"
)

// testSVEReduceSum checks a float ReduceSum. The values are small integers,
// so every partial sum is exact and the result does not depend on the order
// FADDV adds the lanes in.
func testSVEReduceSum[T float32 | float64, V any](t *testing.T, name string, n int, load func([]T) V, sum func(V) T) {
	t.Helper()
	xs := make([]T, n)
	for start := -20; start <= 20; start += 7 {
		var want T
		for i := range xs {
			xs[i] = T(start + 3*i)
			want += xs[i]
		}
		if got := sum(load(xs)); got != want {
			t.Errorf("%s(%v) = %v, want %v", name, xs, got, want)
		}
	}

	inf := T(math.Inf(1))
	for _, c := range []struct {
		lane    int
		val     T
		isNaN   bool
		wantInf bool
	}{
		{0, T(math.NaN()), true, false},
		{n - 1, T(math.NaN()), true, false},
		{n / 2, inf, false, true},
	} {
		for i := range xs {
			xs[i] = 1
		}
		xs[c.lane] = c.val
		got := sum(load(xs))
		switch {
		case c.isNaN && got == got:
			t.Errorf("%s with NaN in lane %d = %v, want NaN", name, c.lane, got)
		case c.wantInf && got != inf:
			t.Errorf("%s with +Inf in lane %d = %v, want +Inf", name, c.lane, got)
		}
	}
}

func TestReduceSumSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEReduceSum(t, "Float32s.ReduceSum", archsimd.Float32s{}.Len(), archsimd.LoadFloat32s, archsimd.Float32s.ReduceSum)
	testSVEReduceSum(t, "Float64s.ReduceSum", archsimd.Float64s{}.Len(), archsimd.LoadFloat64s, archsimd.Float64s.ReduceSum)
	testSVEReduceInt(t, "Int8s.ReduceSum", archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.Int8s.ReduceSum, add[int8])
	testSVEReduceInt(t, "Int16s.ReduceSum", archsimd.Int16s{}.Len(), archsimd.LoadInt16s, archsimd.Int16s.ReduceSum, add[int16])
	testSVEReduceInt(t, "Int32s.ReduceSum", archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.Int32s.ReduceSum, add[int32])
	testSVEReduceInt(t, "Int64s.ReduceSum", archsimd.Int64s{}.Len(), archsimd.LoadInt64s, archsimd.Int64s.ReduceSum, add[int64])
	testSVEReduceInt(t, "Uint8s.ReduceSum", archsimd.Uint8s{}.Len(), archsimd.LoadUint8s, archsimd.Uint8s.ReduceSum, add[uint8])
	testSVEReduceInt(t, "Uint16s.ReduceSum", archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.Uint16s.ReduceSum, add[uint16])
	testSVEReduceInt(t, "Uint32s.ReduceSum", archsimd.Uint32s{}.Len(), archsimd.LoadUint32s, archsimd.Uint32s.ReduceSum, add[uint32])
	testSVEReduceInt(t, "Uint64s.ReduceSum", archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.Uint64s.ReduceSum, add[uint64])
}

// testSVEReduceInt checks an integer reduction against reduceSlice with fn.
// The edge values are -1, the minimum and the maximum (for unsigned types: the
// maximum, the top bit alone and the bits below it). Each edge value and zero
// fills every lane, which makes a sum wrap, and each edge value is placed in
// each lane of every fill. Ramps count up from negative starts. A signed result
// is compared as int64, which sign-extends it after the call.
func testSVEReduceInt[T integer, V any](t *testing.T, name string, n int, load func([]T) V, reduce func(V) T, fn func(a, b T) T) {
	t.Helper()
	xs := make([]T, n)
	check := func() {
		t.Helper()
		if got, w := reduce(load(xs)), reduceSlice(xs, fn); int64(got) != int64(w) {
			t.Errorf("%s(%v) = %v, want %v", name, xs, got, w)
		}
	}
	ones := ^T(0)
	top := ones
	for top<<1 != 0 {
		top <<= 1
	}
	edges := []T{ones, top, top - 1}
	for _, fill := range append([]T{0}, edges...) {
		for i := range xs {
			xs[i] = fill
		}
		check()
		for _, v := range edges {
			for lane := range xs {
				for i := range xs {
					xs[i] = fill
				}
				xs[lane] = v
				check()
			}
		}
	}
	for start := -20; start <= 20; start += 7 {
		for i := range xs {
			xs[i] = T(start + 3*i)
		}
		check()
	}
}

// testSVEReduceMax checks a float ReduceMax with the maximum in each lane,
// with NaN, and with signed zeros.
func testSVEReduceMax[T float32 | float64, V any](t *testing.T, name string, n int, load func([]T) V, reduce func(V) T) {
	t.Helper()
	xs := make([]T, n)
	for lane := range xs {
		for i := range xs {
			xs[i] = T(-3*i - 1)
		}
		xs[lane] = 5
		if got := reduce(load(xs)); got != 5 {
			t.Errorf("%s(%v) = %v, want 5", name, xs, got)
		}
	}

	for i := range xs {
		xs[i] = T(math.Inf(-1))
	}
	if got := reduce(load(xs)); got != T(math.Inf(-1)) {
		t.Errorf("%s(all -Inf) = %v, want -Inf", name, got)
	}

	for _, lane := range []int{0, n / 2, n - 1} {
		for i := range xs {
			xs[i] = T(i)
		}
		xs[lane] = T(math.NaN())
		if got := reduce(load(xs)); got == got {
			t.Errorf("%s with NaN in lane %d = %v, want NaN", name, lane, got)
		}
	}

	negZero := T(math.Copysign(0, -1))
	for _, lane := range []int{0, n - 1} {
		for i := range xs {
			xs[i] = negZero
		}
		xs[lane] = 0
		if got := reduce(load(xs)); got != 0 || math.Signbit(float64(got)) {
			t.Errorf("%s of signed zeros with +0 in lane %d = %v, want +0", name, lane, got)
		}
	}
}

func TestReduceMaxSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEReduceMax(t, "Float32s.ReduceMax", archsimd.Float32s{}.Len(), archsimd.LoadFloat32s, archsimd.Float32s.ReduceMax)
	testSVEReduceMax(t, "Float64s.ReduceMax", archsimd.Float64s{}.Len(), archsimd.LoadFloat64s, archsimd.Float64s.ReduceMax)
	testSVEReduceInt(t, "Int8s.ReduceMax", archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.Int8s.ReduceMax, max_[int8])
	testSVEReduceInt(t, "Int16s.ReduceMax", archsimd.Int16s{}.Len(), archsimd.LoadInt16s, archsimd.Int16s.ReduceMax, max_[int16])
	testSVEReduceInt(t, "Int32s.ReduceMax", archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.Int32s.ReduceMax, max_[int32])
	testSVEReduceInt(t, "Int64s.ReduceMax", archsimd.Int64s{}.Len(), archsimd.LoadInt64s, archsimd.Int64s.ReduceMax, max_[int64])
	testSVEReduceInt(t, "Uint8s.ReduceMax", archsimd.Uint8s{}.Len(), archsimd.LoadUint8s, archsimd.Uint8s.ReduceMax, max_[uint8])
	testSVEReduceInt(t, "Uint16s.ReduceMax", archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.Uint16s.ReduceMax, max_[uint16])
	testSVEReduceInt(t, "Uint32s.ReduceMax", archsimd.Uint32s{}.Len(), archsimd.LoadUint32s, archsimd.Uint32s.ReduceMax, max_[uint32])
	testSVEReduceInt(t, "Uint64s.ReduceMax", archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.Uint64s.ReduceMax, max_[uint64])
}

// testSVEReduceMin checks a float ReduceMin with the minimum in each lane,
// with NaN, and with signed zeros.
func testSVEReduceMin[T float32 | float64, V any](t *testing.T, name string, n int, load func([]T) V, reduce func(V) T) {
	t.Helper()
	xs := make([]T, n)
	for lane := range xs {
		for i := range xs {
			xs[i] = T(3*i + 1)
		}
		xs[lane] = -5
		if got := reduce(load(xs)); got != -5 {
			t.Errorf("%s(%v) = %v, want -5", name, xs, got)
		}
	}

	for i := range xs {
		xs[i] = T(math.Inf(1))
	}
	if got := reduce(load(xs)); got != T(math.Inf(1)) {
		t.Errorf("%s(all +Inf) = %v, want +Inf", name, got)
	}

	for _, lane := range []int{0, n / 2, n - 1} {
		for i := range xs {
			xs[i] = T(i)
		}
		xs[lane] = T(math.NaN())
		if got := reduce(load(xs)); got == got {
			t.Errorf("%s with NaN in lane %d = %v, want NaN", name, lane, got)
		}
	}

	negZero := T(math.Copysign(0, -1))
	for _, lane := range []int{0, n - 1} {
		for i := range xs {
			xs[i] = 0
		}
		xs[lane] = negZero
		if got := reduce(load(xs)); got != 0 || !math.Signbit(float64(got)) {
			t.Errorf("%s of signed zeros with -0 in lane %d = %v, want -0", name, lane, got)
		}
	}
}

func TestReduceMinSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEReduceMin(t, "Float32s.ReduceMin", archsimd.Float32s{}.Len(), archsimd.LoadFloat32s, archsimd.Float32s.ReduceMin)
	testSVEReduceMin(t, "Float64s.ReduceMin", archsimd.Float64s{}.Len(), archsimd.LoadFloat64s, archsimd.Float64s.ReduceMin)
	testSVEReduceInt(t, "Int8s.ReduceMin", archsimd.Int8s{}.Len(), archsimd.LoadInt8s, archsimd.Int8s.ReduceMin, min_[int8])
	testSVEReduceInt(t, "Int16s.ReduceMin", archsimd.Int16s{}.Len(), archsimd.LoadInt16s, archsimd.Int16s.ReduceMin, min_[int16])
	testSVEReduceInt(t, "Int32s.ReduceMin", archsimd.Int32s{}.Len(), archsimd.LoadInt32s, archsimd.Int32s.ReduceMin, min_[int32])
	testSVEReduceInt(t, "Int64s.ReduceMin", archsimd.Int64s{}.Len(), archsimd.LoadInt64s, archsimd.Int64s.ReduceMin, min_[int64])
	testSVEReduceInt(t, "Uint8s.ReduceMin", archsimd.Uint8s{}.Len(), archsimd.LoadUint8s, archsimd.Uint8s.ReduceMin, min_[uint8])
	testSVEReduceInt(t, "Uint16s.ReduceMin", archsimd.Uint16s{}.Len(), archsimd.LoadUint16s, archsimd.Uint16s.ReduceMin, min_[uint16])
	testSVEReduceInt(t, "Uint32s.ReduceMin", archsimd.Uint32s{}.Len(), archsimd.LoadUint32s, archsimd.Uint32s.ReduceMin, min_[uint32])
	testSVEReduceInt(t, "Uint64s.ReduceMin", archsimd.Uint64s{}.Len(), archsimd.LoadUint64s, archsimd.Uint64s.ReduceMin, min_[uint64])
}

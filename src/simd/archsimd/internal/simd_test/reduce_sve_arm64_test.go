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
}

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

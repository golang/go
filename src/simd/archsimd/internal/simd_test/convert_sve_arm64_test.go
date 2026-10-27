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

// fcvtzs32 is FCVTZS on one float32: truncate toward zero, saturate out of
// range values, and convert NaN to 0.
func fcvtzs32(x float32) int32 {
	switch {
	case x != x:
		return 0
	case x >= math.MaxInt32:
		return math.MaxInt32
	case x <= math.MinInt32:
		return math.MinInt32
	}
	return int32(x)
}

func TestConvertToInt32SVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	n := archsimd.Float32s{}.Len()
	vals := []float32{0, 0.5, -0.5, 1.5, -1.5, 2.999, -2.999, 1e9, -1e9, 3e9, -3e9,
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), float32(math.Copysign(0, -1))}
	xs := make([]float32, n)
	got := make([]int32, n)
	for start := 0; start < len(vals); start += n {
		for i := range xs {
			xs[i] = vals[(start+i)%len(vals)]
		}
		archsimd.LoadFloat32s(xs).ConvertToInt32().Store(got)
		for i, x := range xs {
			if want := fcvtzs32(x); got[i] != want {
				t.Errorf("ConvertToInt32(%v) = %d, want %d", x, got[i], want)
			}
		}
	}

	// A select over the conversion folds into its merging form.
	ys := make([]int32, n)
	for i := range xs {
		xs[i] = float32(i) + 0.75
		ys[i] = -int32(i) - 100
	}
	m := archsimd.LoadFloat32s(xs).Greater(archsimd.BroadcastFloat32s(float32(n) / 2))
	archsimd.LoadFloat32s(xs).ConvertToInt32().IfElse(m, archsimd.LoadInt32s(ys)).Store(got)
	for i, x := range xs {
		want := ys[i]
		if x > float32(n)/2 {
			want = int32(i)
		}
		if got[i] != want {
			t.Errorf("ConvertToInt32 under IfElse: lane %d = %d, want %d", i, got[i], want)
		}
	}
}

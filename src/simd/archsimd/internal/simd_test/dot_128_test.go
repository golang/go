// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || wasm)

package simd_test

import (
	"math"
	"simd/archsimd"
	"testing"
)

func dotProductPairsSlice(x, y []int16) []int32 {
	z := make([]int32, len(x)/2)
	for i := range z {
		// Products of int16s always fit in an int32, but their sum can wrap.
		z[i] = int32(x[2*i])*int32(y[2*i]) + int32(x[2*i+1])*int32(y[2*i+1])
	}
	return z
}

func TestDotProductPairs(t *testing.T) {
	forSlicePair(t, int16s, 8, func(x, y []int16) bool {
		t.Helper()
		got := make([]int32, 4)
		archsimd.LoadInt16x8(x).DotProductPairs(archsimd.LoadInt16x8(y)).Store(got)
		return checkSlicesLogInput(t, got, dotProductPairsSlice(x, y), 0.0, func() { t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y) })
	})

	// The only case that overflows is when all four inputs to a sum are
	// MinInt16, which wraps to MinInt32. Changing any one of them by one
	// gives the largest sum that doesn't overflow.
	m := int16(math.MinInt16)
	x := []int16{m, m, m, m, m, m, 1, 2}
	y := []int16{m, m, m, m, m, m + 1, 3, 4}
	got := make([]int32, 4)
	archsimd.LoadInt16x8(x).DotProductPairs(archsimd.LoadInt16x8(y)).Store(got)
	checkSlices(t, got, []int32{math.MinInt32, math.MinInt32, 1<<31 - 1<<15, 11})
}

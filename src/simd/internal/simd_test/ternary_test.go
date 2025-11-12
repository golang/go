// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd"
	"testing"
)

func TestFMA(t *testing.T) {
	if simd.X86.AVX512() {
		testFloat32x4TernaryFlaky(t, simd.Float32x4.MulAdd, fmaSlice[float32], 0.001)
		testFloat32x8TernaryFlaky(t, simd.Float32x8.MulAdd, fmaSlice[float32], 0.001)
		testFloat32x16TernaryFlaky(t, simd.Float32x16.MulAdd, fmaSlice[float32], 0.001)
		testFloat64x2Ternary(t, simd.Float64x2.MulAdd, fmaSlice[float64])
		testFloat64x4Ternary(t, simd.Float64x4.MulAdd, fmaSlice[float64])
		testFloat64x8Ternary(t, simd.Float64x8.MulAdd, fmaSlice[float64])
	}
}

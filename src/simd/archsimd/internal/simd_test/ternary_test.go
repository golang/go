// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestFMA(t *testing.T) {
	if archsimd.X86.AVX512() {
		testFloat32x4TernaryFlaky(t, archsimd.Float32x4.MulAdd, fmaSlice[float32], 0.001)
		testFloat32x8TernaryFlaky(t, archsimd.Float32x8.MulAdd, fmaSlice[float32], 0.001)
		testFloat32x16TernaryFlaky(t, archsimd.Float32x16.MulAdd, fmaSlice[float32], 0.001)
		testFloat64x2Ternary(t, archsimd.Float64x2.MulAdd, fmaSlice[float64])
		testFloat64x4Ternary(t, archsimd.Float64x4.MulAdd, fmaSlice[float64])
		testFloat64x8Ternary(t, archsimd.Float64x8.MulAdd, fmaSlice[float64])
	}
}

func TestFMAAlternating(t *testing.T) {
	if !archsimd.X86.FMA() {
		t.Skip("FMA is not available")
	}
	// Small integers make the product and sum exact in both float types.
	t.Run("Float32x4", func(t *testing.T) {
		x := archsimd.LoadFloat32x4([]float32{1, 2, 3, 4})
		y := archsimd.LoadFloat32x4([]float32{10, 10, 10, 10})
		z := archsimd.LoadFloat32x4([]float32{1, 1, 1, 1})
		var got [4]float32
		x.MulAddEvenSubOdd(y, z).Store(got[:])
		checkSlices(t, got[:], []float32{11, 19, 31, 39})
		x.MulAddOddSubEven(y, z).Store(got[:])
		checkSlices(t, got[:], []float32{9, 21, 29, 41})
	})
	t.Run("Float64x4", func(t *testing.T) {
		x := archsimd.LoadFloat64x4([]float64{1, 2, 3, 4})
		y := archsimd.LoadFloat64x4([]float64{10, 10, 10, 10})
		z := archsimd.LoadFloat64x4([]float64{1, 1, 1, 1})
		var got [4]float64
		x.MulAddEvenSubOdd(y, z).Store(got[:])
		checkSlices(t, got[:], []float64{11, 19, 31, 39})
		x.MulAddOddSubEven(y, z).Store(got[:])
		checkSlices(t, got[:], []float64{9, 21, 29, 41})
	})
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestSqrt(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Sqrt, sqrtSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Sqrt, sqrtSlice[float64])
}

func TestRound(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Round, roundSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Round, roundSlice[float64])
}

func TestFloor(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Floor, floorSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Floor, floorSlice[float64])
}

func TestCeil(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Ceil, ceilSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Ceil, ceilSlice[float64])
}

func TestTrunc(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Trunc, truncSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Trunc, truncSlice[float64])
}

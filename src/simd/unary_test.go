// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd"
	"testing"
)

func TestCeil(t *testing.T) {
	testFloat32x4Unary(t, simd.Float32x4.Ceil, ceilSlice[float32])
	testFloat32x8Unary(t, simd.Float32x8.Ceil, ceilSlice[float32])
	testFloat64x2Unary(t, simd.Float64x2.Ceil, ceilSlice[float64])
	testFloat64x4Unary(t, simd.Float64x4.Ceil, ceilSlice[float64])
	if simd.HasAVX512() {
		// testFloat32x16Unary(t, simd.Float32x16.Ceil, ceilSlice[float32]) // missing
		// testFloat64x8Unary(t, simd.Float64x8.Ceil, ceilSlice[float64])   // missing
	}
}

func TestFloor(t *testing.T) {
	testFloat32x4Unary(t, simd.Float32x4.Floor, floorSlice[float32])
	testFloat32x8Unary(t, simd.Float32x8.Floor, floorSlice[float32])
	testFloat64x2Unary(t, simd.Float64x2.Floor, floorSlice[float64])
	testFloat64x4Unary(t, simd.Float64x4.Floor, floorSlice[float64])
	if simd.HasAVX512() {
		// testFloat32x16Unary(t, simd.Float32x16.Floor, floorSlice[float32]) // missing
		// testFloat64x8Unary(t, simd.Float64x8.Floor, floorSlice[float64])   // missing
	}
}

func TestTrunc(t *testing.T) {
	testFloat32x4Unary(t, simd.Float32x4.Trunc, truncSlice[float32])
	testFloat32x8Unary(t, simd.Float32x8.Trunc, truncSlice[float32])
	testFloat64x2Unary(t, simd.Float64x2.Trunc, truncSlice[float64])
	testFloat64x4Unary(t, simd.Float64x4.Trunc, truncSlice[float64])
	if simd.HasAVX512() {
		// testFloat32x16Unary(t, simd.Float32x16.Trunc, truncSlice[float32]) // missing
		// testFloat64x8Unary(t, simd.Float64x8.Trunc, truncSlice[float64])   // missing
	}
}

func TestRound(t *testing.T) {
	testFloat32x4Unary(t, simd.Float32x4.Round, roundSlice[float32])
	testFloat32x8Unary(t, simd.Float32x8.Round, roundSlice[float32])
	testFloat64x2Unary(t, simd.Float64x2.Round, roundSlice[float64])
	testFloat64x4Unary(t, simd.Float64x4.Round, roundSlice[float64])
	if simd.HasAVX512() {
		// testFloat32x16Unary(t, simd.Float32x16.Round, roundSlice[float32]) // missing
		// testFloat64x8Unary(t, simd.Float64x8.Round, roundSlice[float64])   // missing
	}
}

func TestSqrt(t *testing.T) {
	testFloat32x4Unary(t, simd.Float32x4.Sqrt, sqrtSlice[float32])
	testFloat32x8Unary(t, simd.Float32x8.Sqrt, sqrtSlice[float32])
	testFloat64x2Unary(t, simd.Float64x2.Sqrt, sqrtSlice[float64])
	testFloat64x4Unary(t, simd.Float64x4.Sqrt, sqrtSlice[float64])
	if simd.HasAVX512() {
		testFloat32x16Unary(t, simd.Float32x16.Sqrt, sqrtSlice[float32])
		testFloat64x8Unary(t, simd.Float64x8.Sqrt, sqrtSlice[float64])
	}
}

func TestAbsolute(t *testing.T) {
	testInt8x16Unary(t, simd.Int8x16.Absolute, map1[int8](abs))
	testInt8x32Unary(t, simd.Int8x32.Absolute, map1[int8](abs))
	testInt16x8Unary(t, simd.Int16x8.Absolute, map1[int16](abs))
	testInt16x16Unary(t, simd.Int16x16.Absolute, map1[int16](abs))
	testInt32x4Unary(t, simd.Int32x4.Absolute, map1[int32](abs))
	testInt32x8Unary(t, simd.Int32x8.Absolute, map1[int32](abs))
	if simd.HasAVX512() {
		testInt8x64Unary(t, simd.Int8x64.Absolute, map1[int8](abs))
		testInt16x32Unary(t, simd.Int16x32.Absolute, map1[int16](abs))
		testInt32x16Unary(t, simd.Int32x16.Absolute, map1[int32](abs))
		testInt64x2Unary(t, simd.Int64x2.Absolute, map1[int64](abs))
		testInt64x4Unary(t, simd.Int64x4.Absolute, map1[int64](abs))
		testInt64x8Unary(t, simd.Int64x8.Absolute, map1[int64](abs))
	}
}

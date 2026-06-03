// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// This is a subset of the tests in unary_test.go, but notice also
// that amd64 does not support OnesCount for 128-bit vectors except
// on AVX512.

func TestCeil(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Ceil, ceilSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Ceil, ceilSlice[float64])
}

func TestFloor(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Floor, floorSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Floor, floorSlice[float64])
}

func TestTrunc(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Trunc, truncSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Trunc, truncSlice[float64])
}

func TestRound(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Round, roundSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Round, roundSlice[float64])
}

func TestSqrt(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Sqrt, sqrtSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Sqrt, sqrtSlice[float64])
}

func TestNot(t *testing.T) {
	testInt8x16Unary(t, archsimd.Int8x16.Not, map1[int8](not))
	testInt32x4Unary(t, archsimd.Int32x4.Not, map1[int32](not))
	testInt64x2Unary(t, archsimd.Int64x2.Not, map1[int64](not))
}

func TestAbsolute(t *testing.T) {
	testInt8x16Unary(t, archsimd.Int8x16.Abs, map1[int8](abs))
	testInt16x8Unary(t, archsimd.Int16x8.Abs, map1[int16](abs))
	testInt32x4Unary(t, archsimd.Int32x4.Abs, map1[int32](abs))
}

func TestOnesCount(t *testing.T) {
	testInt8x16Unary(t, archsimd.Int8x16.OnesCount, map1[int8](onesCount))
	testInt16x8Unary(t, archsimd.Int16x8.OnesCount, map1[int16](onesCount))
	testInt32x4Unary(t, archsimd.Int32x4.OnesCount, map1[int32](onesCount))
}

// func TestConvert(t *testing.T) {
// 	testFloat64x2ConvertToFloat32(t, archsimd.Float64x2.ConvertToFloat32, map1n[float64](toFloat32, 4))
// 	testFloat32x4ConvertToFloat64(t, archsimd.Float32x4.ConvertToFloat64, map1[float32](toFloat64))

// 	testFloat32x4ConvertToInt32(t, archsimd.Float32x4.ConvertToInt32, map1[float32](floatToInt32_x86))
// 	testFloat64x2ConvertToInt32(t, archsimd.Float64x2.ConvertToInt32, map1n[float64](floatToInt32_x86, 4))

// 	testInt32x4ConvertToFloat32(t, archsimd.Int32x4.ConvertToFloat32, map1[int32](toFloat32))
// 	testInt32x4ConvertToFloat64(t, archsimd.Int32x4.ConvertToFloat64, map1[int32](toFloat64))
// }

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || wasm || arm64)

package simd_test

import (
	"runtime"
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

func TestAbs(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Abs, map1[float32](abs))
	testFloat64x2Unary(t, archsimd.Float64x2.Abs, map1[float64](abs))
	testInt8x16Unary(t, archsimd.Int8x16.Abs, map1[int8](abs))
	testInt16x8Unary(t, archsimd.Int16x8.Abs, map1[int16](abs))
	testInt32x4Unary(t, archsimd.Int32x4.Abs, map1[int32](abs))
	if runtime.GOARCH != "amd64" || archsimd.X86.AVX512() {
		testInt64x2Unary(t, archsimd.Int64x2.Abs, map1[int64](abs))
	}
}

func TestNeg(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Neg, map1[float32](neg))
	testFloat64x2Unary(t, archsimd.Float64x2.Neg, map1[float64](neg))
	testInt8x16Unary(t, archsimd.Int8x16.Neg, map1[int8](neg))
	testInt16x8Unary(t, archsimd.Int16x8.Neg, map1[int16](neg))
	testInt32x4Unary(t, archsimd.Int32x4.Neg, map1[int32](neg))
	testInt64x2Unary(t, archsimd.Int64x2.Neg, map1[int64](neg))
}

func TestOnesCount(t *testing.T) {
	if runtime.GOARCH == "amd64" && !archsimd.X86.AVX512BITALG() {
		t.Skip("OnesCount on 128-bit 8-bit vectors on amd64 requires AVX512BITALG")
	}
	testInt8x16Unary(t, archsimd.Int8x16.OnesCount, map1[int8](onesCount))
	testUint8x16Unary(t, archsimd.Uint8x16.OnesCount, map1[uint8](onesCount))
}

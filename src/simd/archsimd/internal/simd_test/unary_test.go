// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"math"
	"simd/archsimd"
	"testing"
)

func TestCeil(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Ceil, ceilSlice[float32])
	testFloat32x8Unary(t, archsimd.Float32x8.Ceil, ceilSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Ceil, ceilSlice[float64])
	testFloat64x4Unary(t, archsimd.Float64x4.Ceil, ceilSlice[float64])
	if archsimd.X86.AVX512() {
		// testFloat32x16Unary(t, archsimd.Float32x16.Ceil, ceilSlice[float32]) // missing
		// testFloat64x8Unary(t, archsimd.Float64x8.Ceil, ceilSlice[float64])   // missing
	}
}

func TestFloor(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Floor, floorSlice[float32])
	testFloat32x8Unary(t, archsimd.Float32x8.Floor, floorSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Floor, floorSlice[float64])
	testFloat64x4Unary(t, archsimd.Float64x4.Floor, floorSlice[float64])
	if archsimd.X86.AVX512() {
		// testFloat32x16Unary(t, archsimd.Float32x16.Floor, floorSlice[float32]) // missing
		// testFloat64x8Unary(t, archsimd.Float64x8.Floor, floorSlice[float64])   // missing
	}
}

func TestTrunc(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Trunc, truncSlice[float32])
	testFloat32x8Unary(t, archsimd.Float32x8.Trunc, truncSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Trunc, truncSlice[float64])
	testFloat64x4Unary(t, archsimd.Float64x4.Trunc, truncSlice[float64])
	if archsimd.X86.AVX512() {
		// testFloat32x16Unary(t, archsimd.Float32x16.Trunc, truncSlice[float32]) // missing
		// testFloat64x8Unary(t, archsimd.Float64x8.Trunc, truncSlice[float64])   // missing
	}
}

func TestRound(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.RoundToEven, roundSlice[float32])
	testFloat32x8Unary(t, archsimd.Float32x8.RoundToEven, roundSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.RoundToEven, roundSlice[float64])
	testFloat64x4Unary(t, archsimd.Float64x4.RoundToEven, roundSlice[float64])
	if archsimd.X86.AVX512() {
		// testFloat32x16Unary(t, archsimd.Float32x16.Round, roundSlice[float32]) // missing
		// testFloat64x8Unary(t, archsimd.Float64x8.Round, roundSlice[float64])   // missing
	}
}

func TestSqrt(t *testing.T) {
	testFloat32x4Unary(t, archsimd.Float32x4.Sqrt, sqrtSlice[float32])
	testFloat32x8Unary(t, archsimd.Float32x8.Sqrt, sqrtSlice[float32])
	testFloat64x2Unary(t, archsimd.Float64x2.Sqrt, sqrtSlice[float64])
	testFloat64x4Unary(t, archsimd.Float64x4.Sqrt, sqrtSlice[float64])
	if archsimd.X86.AVX512() {
		testFloat32x16Unary(t, archsimd.Float32x16.Sqrt, sqrtSlice[float32])
		testFloat64x8Unary(t, archsimd.Float64x8.Sqrt, sqrtSlice[float64])
	}
}

func TestNot(t *testing.T) {
	testInt8x16Unary(t, archsimd.Int8x16.Not, map1[int8](not))
	testInt16x8Unary(t, archsimd.Int16x8.Not, map1[int16](not))
	testInt32x4Unary(t, archsimd.Int32x4.Not, map1[int32](not))

	if archsimd.X86.AVX2() {
		testInt8x32Unary(t, archsimd.Int8x32.Not, map1[int8](not))
		testInt16x16Unary(t, archsimd.Int16x16.Not, map1[int16](not))
		testInt32x8Unary(t, archsimd.Int32x8.Not, map1[int32](not))
	}
}

func TestAbsolute(t *testing.T) {
	testInt8x16Unary(t, archsimd.Int8x16.Abs, map1[int8](abs))
	testInt16x8Unary(t, archsimd.Int16x8.Abs, map1[int16](abs))
	testInt32x4Unary(t, archsimd.Int32x4.Abs, map1[int32](abs))

	if archsimd.X86.AVX2() {
		testInt8x32Unary(t, archsimd.Int8x32.Abs, map1[int8](abs))
		testInt16x16Unary(t, archsimd.Int16x16.Abs, map1[int16](abs))
		testInt32x8Unary(t, archsimd.Int32x8.Abs, map1[int32](abs))
	}
	if archsimd.X86.AVX512() {
		testInt8x64Unary(t, archsimd.Int8x64.Abs, map1[int8](abs))
		testInt16x32Unary(t, archsimd.Int16x32.Abs, map1[int16](abs))
		testInt32x16Unary(t, archsimd.Int32x16.Abs, map1[int32](abs))
		testInt64x2Unary(t, archsimd.Int64x2.Abs, map1[int64](abs))
		testInt64x4Unary(t, archsimd.Int64x4.Abs, map1[int64](abs))
		testInt64x8Unary(t, archsimd.Int64x8.Abs, map1[int64](abs))
	}
}

func TestCeilScaledResidue(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Needs AVX512")
	}
	testFloat64x8UnaryFlaky(t,
		func(x archsimd.Float64x8) archsimd.Float64x8 { return x.CeilScaledResidue(0) },
		map1(ceilResidueForPrecision[float64](0)),
		0.001)
	testFloat64x8UnaryFlaky(t,
		func(x archsimd.Float64x8) archsimd.Float64x8 { return x.CeilScaledResidue(1) },
		map1(ceilResidueForPrecision[float64](1)),
		0.001)
	testFloat64x8Unary(t,
		func(x archsimd.Float64x8) archsimd.Float64x8 { return x.Sub(x.CeilScaled(0)) },
		map1[float64](func(x float64) float64 { return x - math.Ceil(x) }))
}

func TestToUint32(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Needs AVX512")
	}
	testFloat32x4ConvertToUint32(t, archsimd.Float32x4.ConvertToUint32, map1[float32](toUint32))
	testFloat32x8ConvertToUint32(t, archsimd.Float32x8.ConvertToUint32, map1[float32](toUint32))
	testFloat32x16ConvertToUint32(t, archsimd.Float32x16.ConvertToUint32, map1[float32](toUint32))
}

func TestToInt32(t *testing.T) {
	testFloat32x4ConvertToInt32(t, archsimd.Float32x4.ConvertToInt32, map1[float32](toInt32))
	testFloat32x8ConvertToInt32(t, archsimd.Float32x8.ConvertToInt32, map1[float32](toInt32))
}

func TestConverts(t *testing.T) {
	testUint8x16ConvertToUint16(t, archsimd.Uint8x16.ExtendToUint16, map1[uint8](toUint16))
	testUint16x8ConvertToUint32(t, archsimd.Uint16x8.ExtendToUint32, map1[uint16](toUint32))
}

func TestConvertsAVX512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Needs AVX512")
	}
	testUint8x32ConvertToUint16(t, archsimd.Uint8x32.ExtendToUint16, map1[uint8](toUint16))
}

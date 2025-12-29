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

func TestConvert(t *testing.T) {
	testFloat64x2ConvertToFloat32(t, archsimd.Float64x2.ConvertToFloat32, map1n[float64](toFloat32, 4))
	testFloat64x4ConvertToFloat32(t, archsimd.Float64x4.ConvertToFloat32, map1[float64](toFloat32))
	testFloat32x4ConvertToFloat64(t, archsimd.Float32x4.ConvertToFloat64, map1[float32](toFloat64))

	testFloat32x4ConvertToInt32(t, archsimd.Float32x4.ConvertToInt32, map1[float32](floatToInt32_x86))
	testFloat32x8ConvertToInt32(t, archsimd.Float32x8.ConvertToInt32, map1[float32](floatToInt32_x86))
	testFloat64x2ConvertToInt32(t, archsimd.Float64x2.ConvertToInt32, map1n[float64](floatToInt32_x86, 4))
	testFloat64x4ConvertToInt32(t, archsimd.Float64x4.ConvertToInt32, map1[float64](floatToInt32_x86))

	testInt32x4ConvertToFloat32(t, archsimd.Int32x4.ConvertToFloat32, map1[int32](toFloat32))
	testInt32x8ConvertToFloat32(t, archsimd.Int32x8.ConvertToFloat32, map1[int32](toFloat32))
	testInt32x4ConvertToFloat64(t, archsimd.Int32x4.ConvertToFloat64, map1[int32](toFloat64))

	if archsimd.X86.AVX512() {
		testFloat32x8ConvertToFloat64(t, archsimd.Float32x8.ConvertToFloat64, map1[float32](toFloat64))
		testFloat64x8ConvertToFloat32(t, archsimd.Float64x8.ConvertToFloat32, map1[float64](toFloat32))

		testFloat32x16ConvertToInt32(t, archsimd.Float32x16.ConvertToInt32, map1[float32](floatToInt32_x86))
		testFloat64x8ConvertToInt32(t, archsimd.Float64x8.ConvertToInt32, map1[float64](floatToInt32_x86))
		testFloat32x4ConvertToInt64(t, archsimd.Float32x4.ConvertToInt64, map1[float32](floatToInt64_x86))
		testFloat32x8ConvertToInt64(t, archsimd.Float32x8.ConvertToInt64, map1[float32](floatToInt64_x86))
		testFloat64x2ConvertToInt64(t, archsimd.Float64x2.ConvertToInt64, map1[float64](floatToInt64_x86))
		testFloat64x4ConvertToInt64(t, archsimd.Float64x4.ConvertToInt64, map1[float64](floatToInt64_x86))
		testFloat64x8ConvertToInt64(t, archsimd.Float64x8.ConvertToInt64, map1[float64](floatToInt64_x86))

		testFloat32x4ConvertToUint32(t, archsimd.Float32x4.ConvertToUint32, map1[float32](floatToUint32_x86))
		testFloat32x8ConvertToUint32(t, archsimd.Float32x8.ConvertToUint32, map1[float32](floatToUint32_x86))
		testFloat32x16ConvertToUint32(t, archsimd.Float32x16.ConvertToUint32, map1[float32](floatToUint32_x86))
		testFloat64x2ConvertToUint32(t, archsimd.Float64x2.ConvertToUint32, map1n[float64](floatToUint32_x86, 4))
		testFloat64x4ConvertToUint32(t, archsimd.Float64x4.ConvertToUint32, map1[float64](floatToUint32_x86))
		testFloat64x8ConvertToUint32(t, archsimd.Float64x8.ConvertToUint32, map1[float64](floatToUint32_x86))
		testFloat32x4ConvertToUint64(t, archsimd.Float32x4.ConvertToUint64, map1[float32](floatToUint64_x86))
		testFloat32x8ConvertToUint64(t, archsimd.Float32x8.ConvertToUint64, map1[float32](floatToUint64_x86))
		testFloat64x2ConvertToUint64(t, archsimd.Float64x2.ConvertToUint64, map1[float64](floatToUint64_x86))
		testFloat64x4ConvertToUint64(t, archsimd.Float64x4.ConvertToUint64, map1[float64](floatToUint64_x86))
		testFloat64x8ConvertToUint64(t, archsimd.Float64x8.ConvertToUint64, map1[float64](floatToUint64_x86))

		testInt32x16ConvertToFloat32(t, archsimd.Int32x16.ConvertToFloat32, map1[int32](toFloat32))
		testInt64x2ConvertToFloat32(t, archsimd.Int64x2.ConvertToFloat32, map1n[int64](toFloat32, 4))
		testInt64x4ConvertToFloat32(t, archsimd.Int64x4.ConvertToFloat32, map1[int64](toFloat32))
		testInt64x8ConvertToFloat32(t, archsimd.Int64x8.ConvertToFloat32, map1[int64](toFloat32))
		testInt64x2ConvertToFloat64(t, archsimd.Int64x2.ConvertToFloat64, map1[int64](toFloat64))
		testInt64x4ConvertToFloat64(t, archsimd.Int64x4.ConvertToFloat64, map1[int64](toFloat64))
		testInt64x8ConvertToFloat64(t, archsimd.Int64x8.ConvertToFloat64, map1[int64](toFloat64))

		testUint32x4ConvertToFloat32(t, archsimd.Uint32x4.ConvertToFloat32, map1[uint32](toFloat32))
		testUint32x8ConvertToFloat32(t, archsimd.Uint32x8.ConvertToFloat32, map1[uint32](toFloat32))
		testUint32x16ConvertToFloat32(t, archsimd.Uint32x16.ConvertToFloat32, map1[uint32](toFloat32))
		testUint64x2ConvertToFloat32(t, archsimd.Uint64x2.ConvertToFloat32, map1n[uint64](toFloat32, 4))
		testUint64x4ConvertToFloat32(t, archsimd.Uint64x4.ConvertToFloat32, map1[uint64](toFloat32))
		testUint64x8ConvertToFloat32(t, archsimd.Uint64x8.ConvertToFloat32, map1[uint64](toFloat32))
		testUint32x4ConvertToFloat64(t, archsimd.Uint32x4.ConvertToFloat64, map1[uint32](toFloat64))
		testUint32x8ConvertToFloat64(t, archsimd.Uint32x8.ConvertToFloat64, map1[uint32](toFloat64))
		testUint64x2ConvertToFloat64(t, archsimd.Uint64x2.ConvertToFloat64, map1[uint64](toFloat64))
		testUint64x4ConvertToFloat64(t, archsimd.Uint64x4.ConvertToFloat64, map1[uint64](toFloat64))
		testUint64x8ConvertToFloat64(t, archsimd.Uint64x8.ConvertToFloat64, map1[uint64](toFloat64))
	}
}

func TestExtend(t *testing.T) {
	if archsimd.X86.AVX2() {
		testInt8x16ConvertToInt16(t, archsimd.Int8x16.ExtendToInt16, map1[int8](toInt16))
		testInt16x8ConvertToInt32(t, archsimd.Int16x8.ExtendToInt32, map1[int16](toInt32))
		testInt32x4ConvertToInt64(t, archsimd.Int32x4.ExtendToInt64, map1[int32](toInt64))
		testUint8x16ConvertToUint16(t, archsimd.Uint8x16.ExtendToUint16, map1[uint8](toUint16))
		testUint16x8ConvertToUint32(t, archsimd.Uint16x8.ExtendToUint32, map1[uint16](toUint32))
		testUint32x4ConvertToUint64(t, archsimd.Uint32x4.ExtendToUint64, map1[uint32](toUint64))
	}

	if archsimd.X86.AVX512() {
		testInt8x32ConvertToInt16(t, archsimd.Int8x32.ExtendToInt16, map1[int8](toInt16))
		testInt8x16ConvertToInt32(t, archsimd.Int8x16.ExtendToInt32, map1[int8](toInt32))
		testInt16x16ConvertToInt32(t, archsimd.Int16x16.ExtendToInt32, map1[int16](toInt32))
		testInt16x8ConvertToInt64(t, archsimd.Int16x8.ExtendToInt64, map1[int16](toInt64))
		testInt32x8ConvertToInt64(t, archsimd.Int32x8.ExtendToInt64, map1[int32](toInt64))
		testUint8x32ConvertToUint16(t, archsimd.Uint8x32.ExtendToUint16, map1[uint8](toUint16))
		testUint8x16ConvertToUint32(t, archsimd.Uint8x16.ExtendToUint32, map1[uint8](toUint32))
		testUint16x16ConvertToUint32(t, archsimd.Uint16x16.ExtendToUint32, map1[uint16](toUint32))
		testUint16x8ConvertToUint64(t, archsimd.Uint16x8.ExtendToUint64, map1[uint16](toUint64))
		testUint32x8ConvertToUint64(t, archsimd.Uint32x8.ExtendToUint64, map1[uint32](toUint64))
	}
}

func TestExtendLo(t *testing.T) {
	testInt8x16ConvertLoToInt64x2(t, archsimd.Int8x16.ExtendLo2ToInt64, map1n[int8](toInt64, 2))
	testInt16x8ConvertLoToInt64x2(t, archsimd.Int16x8.ExtendLo2ToInt64, map1n[int16](toInt64, 2))
	testInt32x4ConvertLoToInt64x2(t, archsimd.Int32x4.ExtendLo2ToInt64, map1n[int32](toInt64, 2))
	testUint8x16ConvertLoToUint64x2(t, archsimd.Uint8x16.ExtendLo2ToUint64, map1n[uint8](toUint64, 2))
	testUint16x8ConvertLoToUint64x2(t, archsimd.Uint16x8.ExtendLo2ToUint64, map1n[uint16](toUint64, 2))
	testUint32x4ConvertLoToUint64x2(t, archsimd.Uint32x4.ExtendLo2ToUint64, map1n[uint32](toUint64, 2))
	testInt8x16ConvertLoToInt32x4(t, archsimd.Int8x16.ExtendLo4ToInt32, map1n[int8](toInt32, 4))
	testInt16x8ConvertLoToInt32x4(t, archsimd.Int16x8.ExtendLo4ToInt32, map1n[int16](toInt32, 4))
	testUint8x16ConvertLoToUint32x4(t, archsimd.Uint8x16.ExtendLo4ToUint32, map1n[uint8](toUint32, 4))
	testUint16x8ConvertLoToUint32x4(t, archsimd.Uint16x8.ExtendLo4ToUint32, map1n[uint16](toUint32, 4))
	testInt8x16ConvertLoToInt16x8(t, archsimd.Int8x16.ExtendLo8ToInt16, map1n[int8](toInt16, 8))
	testUint8x16ConvertLoToUint16x8(t, archsimd.Uint8x16.ExtendLo8ToUint16, map1n[uint8](toUint16, 8))

	if archsimd.X86.AVX2() {
		testInt8x16ConvertLoToInt64x4(t, archsimd.Int8x16.ExtendLo4ToInt64, map1n[int8](toInt64, 4))
		testInt16x8ConvertLoToInt64x4(t, archsimd.Int16x8.ExtendLo4ToInt64, map1n[int16](toInt64, 4))
		testUint8x16ConvertLoToUint64x4(t, archsimd.Uint8x16.ExtendLo4ToUint64, map1n[uint8](toUint64, 4))
		testUint16x8ConvertLoToUint64x4(t, archsimd.Uint16x8.ExtendLo4ToUint64, map1n[uint16](toUint64, 4))
		testInt8x16ConvertLoToInt32x8(t, archsimd.Int8x16.ExtendLo8ToInt32, map1n[int8](toInt32, 8))
		testUint8x16ConvertLoToUint32x8(t, archsimd.Uint8x16.ExtendLo8ToUint32, map1n[uint8](toUint32, 8))
	}

	if archsimd.X86.AVX512() {
		testInt8x16ConvertToInt64(t, archsimd.Int8x16.ExtendLo8ToInt64, map1n[int8](toInt64, 8))
		testUint8x16ConvertToUint64(t, archsimd.Uint8x16.ExtendLo8ToUint64, map1n[uint8](toUint64, 8))
	}
}

func TestTruncate(t *testing.T) {
	if archsimd.X86.AVX512() {
		testInt16x8ConvertToInt8(t, archsimd.Int16x8.TruncateToInt8, map1n[int16](toInt8, 16))
		testInt16x16ConvertToInt8(t, archsimd.Int16x16.TruncateToInt8, map1[int16](toInt8))
		testInt16x32ConvertToInt8(t, archsimd.Int16x32.TruncateToInt8, map1[int16](toInt8))
		testInt32x4ConvertToInt8(t, archsimd.Int32x4.TruncateToInt8, map1n[int32](toInt8, 16))
		testInt32x8ConvertToInt8(t, archsimd.Int32x8.TruncateToInt8, map1n[int32](toInt8, 16))
		testInt32x16ConvertToInt8(t, archsimd.Int32x16.TruncateToInt8, map1[int32](toInt8))
		testInt64x2ConvertToInt8(t, archsimd.Int64x2.TruncateToInt8, map1n[int64](toInt8, 16))
		testInt64x4ConvertToInt8(t, archsimd.Int64x4.TruncateToInt8, map1n[int64](toInt8, 16))
		testInt64x8ConvertToInt8(t, archsimd.Int64x8.TruncateToInt8, map1n[int64](toInt8, 16))
		testInt32x4ConvertToInt16(t, archsimd.Int32x4.TruncateToInt16, map1n[int32](toInt16, 8))
		testInt32x8ConvertToInt16(t, archsimd.Int32x8.TruncateToInt16, map1[int32](toInt16))
		testInt32x16ConvertToInt16(t, archsimd.Int32x16.TruncateToInt16, map1[int32](toInt16))
		testInt64x2ConvertToInt16(t, archsimd.Int64x2.TruncateToInt16, map1n[int64](toInt16, 8))
		testInt64x4ConvertToInt16(t, archsimd.Int64x4.TruncateToInt16, map1n[int64](toInt16, 8))
		testInt64x8ConvertToInt16(t, archsimd.Int64x8.TruncateToInt16, map1[int64](toInt16))
		testInt64x2ConvertToInt32(t, archsimd.Int64x2.TruncateToInt32, map1n[int64](toInt32, 4))
		testInt64x4ConvertToInt32(t, archsimd.Int64x4.TruncateToInt32, map1[int64](toInt32))
		testInt64x8ConvertToInt32(t, archsimd.Int64x8.TruncateToInt32, map1[int64](toInt32))

		testUint16x8ConvertToUint8(t, archsimd.Uint16x8.TruncateToUint8, map1n[uint16](toUint8, 16))
		testUint16x16ConvertToUint8(t, archsimd.Uint16x16.TruncateToUint8, map1[uint16](toUint8))
		testUint16x32ConvertToUint8(t, archsimd.Uint16x32.TruncateToUint8, map1[uint16](toUint8))
		testUint32x4ConvertToUint8(t, archsimd.Uint32x4.TruncateToUint8, map1n[uint32](toUint8, 16))
		testUint32x8ConvertToUint8(t, archsimd.Uint32x8.TruncateToUint8, map1n[uint32](toUint8, 16))
		testUint32x16ConvertToUint8(t, archsimd.Uint32x16.TruncateToUint8, map1[uint32](toUint8))
		testUint64x2ConvertToUint8(t, archsimd.Uint64x2.TruncateToUint8, map1n[uint64](toUint8, 16))
		testUint64x4ConvertToUint8(t, archsimd.Uint64x4.TruncateToUint8, map1n[uint64](toUint8, 16))
		testUint64x8ConvertToUint8(t, archsimd.Uint64x8.TruncateToUint8, map1n[uint64](toUint8, 16))
		testUint32x4ConvertToUint16(t, archsimd.Uint32x4.TruncateToUint16, map1n[uint32](toUint16, 8))
		testUint32x8ConvertToUint16(t, archsimd.Uint32x8.TruncateToUint16, map1[uint32](toUint16))
		testUint32x16ConvertToUint16(t, archsimd.Uint32x16.TruncateToUint16, map1[uint32](toUint16))
		testUint64x2ConvertToUint16(t, archsimd.Uint64x2.TruncateToUint16, map1n[uint64](toUint16, 8))
		testUint64x4ConvertToUint16(t, archsimd.Uint64x4.TruncateToUint16, map1n[uint64](toUint16, 8))
		testUint64x8ConvertToUint16(t, archsimd.Uint64x8.TruncateToUint16, map1[uint64](toUint16))
		testUint64x2ConvertToUint32(t, archsimd.Uint64x2.TruncateToUint32, map1n[uint64](toUint32, 4))
		testUint64x4ConvertToUint32(t, archsimd.Uint64x4.TruncateToUint32, map1[uint64](toUint32))
		testUint64x8ConvertToUint32(t, archsimd.Uint64x8.TruncateToUint32, map1[uint64](toUint32))
	}
}

func TestSaturate(t *testing.T) {
	if archsimd.X86.AVX512() {
		testInt16x8ConvertToInt8(t, archsimd.Int16x8.SaturateToInt8, map1n[int16](satToInt8, 16))
		testInt16x16ConvertToInt8(t, archsimd.Int16x16.SaturateToInt8, map1[int16](satToInt8))
		testInt16x32ConvertToInt8(t, archsimd.Int16x32.SaturateToInt8, map1[int16](satToInt8))
		testInt32x4ConvertToInt8(t, archsimd.Int32x4.SaturateToInt8, map1n[int32](satToInt8, 16))
		testInt32x8ConvertToInt8(t, archsimd.Int32x8.SaturateToInt8, map1n[int32](satToInt8, 16))
		testInt32x16ConvertToInt8(t, archsimd.Int32x16.SaturateToInt8, map1[int32](satToInt8))
		testInt64x2ConvertToInt8(t, archsimd.Int64x2.SaturateToInt8, map1n[int64](satToInt8, 16))
		testInt64x4ConvertToInt8(t, archsimd.Int64x4.SaturateToInt8, map1n[int64](satToInt8, 16))
		testInt64x8ConvertToInt8(t, archsimd.Int64x8.SaturateToInt8, map1n[int64](satToInt8, 16))
		testInt32x4ConvertToInt16(t, archsimd.Int32x4.SaturateToInt16, map1n[int32](satToInt16, 8))
		testInt32x8ConvertToInt16(t, archsimd.Int32x8.SaturateToInt16, map1[int32](satToInt16))
		testInt32x16ConvertToInt16(t, archsimd.Int32x16.SaturateToInt16, map1[int32](satToInt16))
		testInt64x2ConvertToInt16(t, archsimd.Int64x2.SaturateToInt16, map1n[int64](satToInt16, 8))
		testInt64x4ConvertToInt16(t, archsimd.Int64x4.SaturateToInt16, map1n[int64](satToInt16, 8))
		testInt64x8ConvertToInt16(t, archsimd.Int64x8.SaturateToInt16, map1[int64](satToInt16))
		testInt64x2ConvertToInt32(t, archsimd.Int64x2.SaturateToInt32, map1n[int64](satToInt32, 4))
		testInt64x4ConvertToInt32(t, archsimd.Int64x4.SaturateToInt32, map1[int64](satToInt32))
		testInt64x8ConvertToInt32(t, archsimd.Int64x8.SaturateToInt32, map1[int64](satToInt32))

		testUint16x8ConvertToUint8(t, archsimd.Uint16x8.SaturateToUint8, map1n[uint16](satToUint8, 16))
		testUint16x16ConvertToUint8(t, archsimd.Uint16x16.SaturateToUint8, map1[uint16](satToUint8))
		testUint16x32ConvertToUint8(t, archsimd.Uint16x32.SaturateToUint8, map1[uint16](satToUint8))
		testUint32x4ConvertToUint8(t, archsimd.Uint32x4.SaturateToUint8, map1n[uint32](satToUint8, 16))
		testUint32x8ConvertToUint8(t, archsimd.Uint32x8.SaturateToUint8, map1n[uint32](satToUint8, 16))
		testUint32x16ConvertToUint8(t, archsimd.Uint32x16.SaturateToUint8, map1[uint32](satToUint8))
		testUint64x2ConvertToUint8(t, archsimd.Uint64x2.SaturateToUint8, map1n[uint64](satToUint8, 16))
		testUint64x4ConvertToUint8(t, archsimd.Uint64x4.SaturateToUint8, map1n[uint64](satToUint8, 16))
		testUint64x8ConvertToUint8(t, archsimd.Uint64x8.SaturateToUint8, map1n[uint64](satToUint8, 16))
		testUint32x4ConvertToUint16(t, archsimd.Uint32x4.SaturateToUint16, map1n[uint32](satToUint16, 8))
		testUint32x8ConvertToUint16(t, archsimd.Uint32x8.SaturateToUint16, map1[uint32](satToUint16))
		testUint32x16ConvertToUint16(t, archsimd.Uint32x16.SaturateToUint16, map1[uint32](satToUint16))
		testUint64x2ConvertToUint16(t, archsimd.Uint64x2.SaturateToUint16, map1n[uint64](satToUint16, 8))
		testUint64x4ConvertToUint16(t, archsimd.Uint64x4.SaturateToUint16, map1n[uint64](satToUint16, 8))
		testUint64x8ConvertToUint16(t, archsimd.Uint64x8.SaturateToUint16, map1[uint64](satToUint16))
		testUint64x2ConvertToUint32(t, archsimd.Uint64x2.SaturateToUint32, map1n[uint64](satToUint32, 4))
		testUint64x4ConvertToUint32(t, archsimd.Uint64x4.SaturateToUint32, map1[uint64](satToUint32))
		testUint64x8ConvertToUint32(t, archsimd.Uint64x8.SaturateToUint32, map1[uint64](satToUint32))
	}
}

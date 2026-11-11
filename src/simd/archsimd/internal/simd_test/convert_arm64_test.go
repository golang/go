// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestConvertARM64(t *testing.T) {
	testFloat64x2ConvertToFloat32(t, archsimd.Float64x2.ConvertToFloat32, map1[float64](toFloat32))
	testFloat32x4ConvertToFloat64(t, archsimd.Float32x4.ConvertToFloat64, map1n[float32](toFloat64, 2))

	testFloat32x4ConvertToInt32(t, archsimd.Float32x4.ConvertToInt32, map1[float32](floatToInt32_arm64))
	testFloat64x2ConvertToInt64(t, archsimd.Float64x2.ConvertToInt64, map1[float64](floatToInt64_arm64))

	testFloat32x4ConvertToUint32(t, archsimd.Float32x4.ConvertToUint32, map1[float32](floatToUint32_arm64))
	testFloat64x2ConvertToUint64(t, archsimd.Float64x2.ConvertToUint64, map1[float64](floatToUint64_arm64))

	testInt32x4ConvertToFloat32(t, archsimd.Int32x4.ConvertToFloat32, map1[int32](toFloat32))
	testInt64x2ConvertToFloat64(t, archsimd.Int64x2.ConvertToFloat64, map1[int64](toFloat64))

	testUint32x4ConvertToFloat32(t, archsimd.Uint32x4.ConvertToFloat32, map1[uint32](toFloat32))
	testUint64x2ConvertToFloat64(t, archsimd.Uint64x2.ConvertToFloat64, map1[uint64](toFloat64))
}

func TestTruncateARM64(t *testing.T) {
	testInt16x8ConvertToInt8(t, archsimd.Int16x8.TruncToInt8, map1[int16](toInt8))
	testInt32x4ConvertToInt16(t, archsimd.Int32x4.TruncToInt16, map1[int32](toInt16))
	testInt64x2ConvertToInt32(t, archsimd.Int64x2.TruncToInt32, map1[int64](toInt32))

	testUint16x8ConvertToUint8(t, archsimd.Uint16x8.TruncToUint8, map1[uint16](toUint8))
	testUint32x4ConvertToUint16(t, archsimd.Uint32x4.TruncToUint16, map1[uint32](toUint16))
	testUint64x2ConvertToUint32(t, archsimd.Uint64x2.TruncToUint32, map1[uint64](toUint32))
}

func TestExtendARM64(t *testing.T) {
	testInt8x16ConvertToInt16(t, archsimd.Int8x16.ExtendToInt16, map1n[int8](toInt16, 8))
	testInt16x8ConvertToInt32(t, archsimd.Int16x8.ExtendToInt32, map1n[int16](toInt32, 4))
	testInt32x4ConvertToInt64(t, archsimd.Int32x4.ExtendToInt64, map1n[int32](toInt64, 2))

	testUint8x16ConvertToUint16(t, archsimd.Uint8x16.ExtendToUint16, map1n[uint8](toUint16, 8))
	testUint16x8ConvertToUint32(t, archsimd.Uint16x8.ExtendToUint32, map1n[uint16](toUint32, 4))
	testUint32x4ConvertToUint64(t, archsimd.Uint32x4.ExtendToUint64, map1n[uint32](toUint64, 2))
}

func TestSaturateARM64(t *testing.T) {
	testInt16x8ConvertToInt8(t, archsimd.Int16x8.SaturateToInt8, map1[int16](satToInt8))
	testInt32x4ConvertToInt16(t, archsimd.Int32x4.SaturateToInt16, map1[int32](satToInt16))
	testInt64x2ConvertToInt32(t, archsimd.Int64x2.SaturateToInt32, map1[int64](satToInt32))

	testUint16x8ConvertToUint8(t, archsimd.Uint16x8.SaturateToUint8, map1[uint16](satToUint8))
	testUint32x4ConvertToUint16(t, archsimd.Uint32x4.SaturateToUint16, map1[uint32](satToUint16))
	testUint64x2ConvertToUint32(t, archsimd.Uint64x2.SaturateToUint32, map1[uint64](satToUint32))

	testInt16x8ConvertToUint8(t, archsimd.Int16x8.SaturateToUint8, map1[int16](satToUint8))
	testInt32x4ConvertToUint16(t, archsimd.Int32x4.SaturateToUint16, map1[int32](satToUint16))
	testInt64x2ConvertToUint32(t, archsimd.Int64x2.SaturateToUint32, map1[int64](satToUint32))
}

// TestExtendHiARM64 tests x.HiToLo().ExtendTo*(), which compiles to a single
// instruction that widens the high half of x.
func TestExtendHiARM64(t *testing.T) {
	testInt8x16ConvertToInt16(t, func(x archsimd.Int8x16) archsimd.Int16x8 { return x.HiToLo().ExtendToInt16() }, hi[int8](toInt16))
	testInt16x8ConvertToInt32(t, func(x archsimd.Int16x8) archsimd.Int32x4 { return x.HiToLo().ExtendToInt32() }, hi[int16](toInt32))
	testInt32x4ConvertToInt64(t, func(x archsimd.Int32x4) archsimd.Int64x2 { return x.HiToLo().ExtendToInt64() }, hi[int32](toInt64))

	testUint8x16ConvertToUint16(t, func(x archsimd.Uint8x16) archsimd.Uint16x8 { return x.HiToLo().ExtendToUint16() }, hi[uint8](toUint16))
	testUint16x8ConvertToUint32(t, func(x archsimd.Uint16x8) archsimd.Uint32x4 { return x.HiToLo().ExtendToUint32() }, hi[uint16](toUint32))
	testUint32x4ConvertToUint64(t, func(x archsimd.Uint32x4) archsimd.Uint64x2 { return x.HiToLo().ExtendToUint64() }, hi[uint32](toUint64))

	// Widening float conversions fold the same way, into FCVTL2.
	testFloat32x4ConvertToFloat64(t, func(x archsimd.Float32x4) archsimd.Float64x2 { return x.HiToLo().ConvertToFloat64() }, hi[float32](toFloat64))
}

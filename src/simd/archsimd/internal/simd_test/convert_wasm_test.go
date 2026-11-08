// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestConvertWasm(t *testing.T) {
	testFloat64x2ConvertToFloat32(t, archsimd.Float64x2.ConvertToFloat32, map1[float64](toFloat32))
	testFloat32x4ConvertToFloat64(t, archsimd.Float32x4.ConvertToFloat64, map1n[float32](toFloat64, 2))
	testInt32x4ConvertToFloat64(t, archsimd.Int32x4.ConvertToFloat64, map1n[int32](toFloat64, 2))
	testUint32x4ConvertToFloat64(t, archsimd.Uint32x4.ConvertToFloat64, map1n[uint32](toFloat64, 2))
	// Wasm float-to-int conversions saturate, like arm64's.
	testFloat64x2ConvertToInt32(t, archsimd.Float64x2.ConvertToInt32, map1[float64](floatToInt32_arm64))
	testFloat64x2ConvertToUint32(t, archsimd.Float64x2.ConvertToUint32, map1[float64](floatToUint32_arm64))
}

func TestExtendWasm(t *testing.T) {
	testInt8x16ConvertToInt16(t, archsimd.Int8x16.ExtendToInt16, map1n[int8](toInt16, 8))
	testInt16x8ConvertToInt32(t, archsimd.Int16x8.ExtendToInt32, map1n[int16](toInt32, 4))
	testInt32x4ConvertToInt64(t, archsimd.Int32x4.ExtendToInt64, map1n[int32](toInt64, 2))

	testUint8x16ConvertToUint16(t, archsimd.Uint8x16.ExtendToUint16, map1n[uint8](toUint16, 8))
	testUint16x8ConvertToUint32(t, archsimd.Uint16x8.ExtendToUint32, map1n[uint16](toUint32, 4))
	testUint32x4ConvertToUint64(t, archsimd.Uint32x4.ExtendToUint64, map1n[uint32](toUint64, 2))
}

func TestTruncateWasm(t *testing.T) {
	testInt16x8ConvertToInt8(t, archsimd.Int16x8.TruncToInt8, map1[int16](toInt8))
	testInt32x4ConvertToInt16(t, archsimd.Int32x4.TruncToInt16, map1[int32](toInt16))
	testInt64x2ConvertToInt32(t, archsimd.Int64x2.TruncToInt32, map1[int64](toInt32))

	testUint16x8ConvertToUint8(t, archsimd.Uint16x8.TruncToUint8, map1[uint16](toUint8))
	testUint32x4ConvertToUint16(t, archsimd.Uint32x4.TruncToUint16, map1[uint32](toUint16))
	testUint64x2ConvertToUint32(t, archsimd.Uint64x2.TruncToUint32, map1[uint64](toUint32))
}

func TestSaturateWasm(t *testing.T) {
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

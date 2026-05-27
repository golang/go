// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestConvertArm64(t *testing.T) {
	testFloat64x2ConvertToFloat32x4(t, archsimd.Float64x2.ConvertToFloat32, map1[float64](toFloat32))
	testFloat32x4ConvertToFloat64x2(t, archsimd.Float32x4.ConvertLo2ToFloat64, map1n[float32](toFloat64, 2))

	testFloat32x4ConvertToInt32x4(t, archsimd.Float32x4.ConvertToInt32, map1[float32](floatToInt32_arm64))
	testFloat64x2ConvertToInt64x2(t, archsimd.Float64x2.ConvertToInt64, map1[float64](floatToInt64_arm64))

	testFloat32x4ConvertToUint32x4(t, archsimd.Float32x4.ConvertToUint32, map1[float32](floatToUint32_arm64))
	testFloat64x2ConvertToUint64x2(t, archsimd.Float64x2.ConvertToUint64, map1[float64](floatToUint64_arm64))

	testInt32x4ConvertToFloat32x4(t, archsimd.Int32x4.ConvertToFloat32, map1[int32](toFloat32))
	testInt64x2ConvertToFloat64x2(t, archsimd.Int64x2.ConvertToFloat64, map1[int64](toFloat64))

	testUint32x4ConvertToFloat32x4(t, archsimd.Uint32x4.ConvertToFloat32, map1[uint32](toFloat32))
	testUint64x2ConvertToFloat64x2(t, archsimd.Uint64x2.ConvertToFloat64, map1[uint64](toFloat64))
}

func TestTruncateArm64(t *testing.T) {
	testInt16x8ConvertToInt8x16(t, archsimd.Int16x8.TruncToInt8, map1[int16](toInt8))
	testInt32x4ConvertToInt16x8(t, archsimd.Int32x4.TruncToInt16, map1[int32](toInt16))
	testInt64x2ConvertToInt32x4(t, archsimd.Int64x2.TruncToInt32, map1[int64](toInt32))

	testUint16x8ConvertToUint8x16(t, archsimd.Uint16x8.TruncToUint8, map1[uint16](toUint8))
	testUint32x4ConvertToUint16x8(t, archsimd.Uint32x4.TruncToUint16, map1[uint32](toUint16))
	testUint64x2ConvertToUint32x4(t, archsimd.Uint64x2.TruncToUint32, map1[uint64](toUint32))
}

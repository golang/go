// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestMulAdd(t *testing.T) {
	testFloat32x4TernaryFlaky(t, archsimd.Float32x4.MulAdd, fmaSlice[float32], 0.001)
	testFloat64x2TernaryFlaky(t, archsimd.Float64x2.MulAdd, fmaSlice[float64], 0.001)
	testInt8x16Ternary(t, archsimd.Int8x16.MulAdd, imaSlice[int8])
	testInt16x8Ternary(t, archsimd.Int16x8.MulAdd, imaSlice[int16])
	testInt32x4Ternary(t, archsimd.Int32x4.MulAdd, imaSlice[int32])
	testUint8x16Ternary(t, archsimd.Uint8x16.MulAdd, imaSlice[uint8])
	testUint16x8Ternary(t, archsimd.Uint16x8.MulAdd, imaSlice[uint16])
	testUint32x4Ternary(t, archsimd.Uint32x4.MulAdd, imaSlice[uint32])
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestAnd(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.And, andSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.And, andSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.And, andSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.And, andSlice[int64])

	testUint8x16Binary(t, archsimd.Uint8x16.And, andSlice[uint8])
	testUint16x8Binary(t, archsimd.Uint16x8.And, andSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.And, andSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.And, andSlice[uint64])
}

func TestOr(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.Or, orSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.Or, orSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Or, orSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Or, orSlice[int64])

	testUint8x16Binary(t, archsimd.Uint8x16.Or, orSlice[uint8])
	testUint16x8Binary(t, archsimd.Uint16x8.Or, orSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Or, orSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Or, orSlice[uint64])
}

func TestXor(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.Xor, xorSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.Xor, xorSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Xor, xorSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Xor, xorSlice[int64])

	testUint8x16Binary(t, archsimd.Uint8x16.Xor, xorSlice[uint8])
	testUint16x8Binary(t, archsimd.Uint16x8.Xor, xorSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Xor, xorSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Xor, xorSlice[uint64])
}

func TestAndNot(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.AndNot, andNotSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.AndNot, andNotSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.AndNot, andNotSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.AndNot, andNotSlice[int64])

	testUint8x16Binary(t, archsimd.Uint8x16.AndNot, andNotSlice[uint8])
	testUint16x8Binary(t, archsimd.Uint16x8.AndNot, andNotSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.AndNot, andNotSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.AndNot, andNotSlice[uint64])
}

func TestOrNot(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.OrNot, orNotSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.OrNot, orNotSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.OrNot, orNotSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.OrNot, orNotSlice[int64])

	testUint8x16Binary(t, archsimd.Uint8x16.OrNot, orNotSlice[uint8])
	testUint16x8Binary(t, archsimd.Uint16x8.OrNot, orNotSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.OrNot, orNotSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.OrNot, orNotSlice[uint64])
}

func TestNot(t *testing.T) {
	testInt8x16Unary(t, archsimd.Int8x16.Not, notSlice[int8])
	testInt16x8Unary(t, archsimd.Int16x8.Not, notSlice[int16])
	testInt32x4Unary(t, archsimd.Int32x4.Not, notSlice[int32])
	testInt64x2Unary(t, archsimd.Int64x2.Not, notSlice[int64])

	testUint8x16Unary(t, archsimd.Uint8x16.Not, notSlice[uint8])
	testUint16x8Unary(t, archsimd.Uint16x8.Not, notSlice[uint16])
	testUint32x4Unary(t, archsimd.Uint32x4.Not, notSlice[uint32])
	testUint64x2Unary(t, archsimd.Uint64x2.Not, notSlice[uint64])
}

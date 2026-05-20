// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestAdd(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Add, addSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Add, addSlice[float64])

	testInt16x8Binary(t, archsimd.Int16x8.Add, addSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Add, addSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Add, addSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Add, addSlice[int8])

	testUint32x4Binary(t, archsimd.Uint32x4.Add, addSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Add, addSlice[uint64])
	testUint16x8Binary(t, archsimd.Uint16x8.Add, addSlice[uint16])
	testUint8x16Binary(t, archsimd.Uint8x16.Add, addSlice[uint8])
}

func TestSub(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Sub, subSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Sub, subSlice[float64])

	testInt32x4Binary(t, archsimd.Int32x4.Sub, subSlice[int32])
	testInt16x8Binary(t, archsimd.Int16x8.Sub, subSlice[int16])
	testInt64x2Binary(t, archsimd.Int64x2.Sub, subSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Sub, subSlice[int8])

	testUint32x4Binary(t, archsimd.Uint32x4.Sub, subSlice[uint32])
	testUint16x8Binary(t, archsimd.Uint16x8.Sub, subSlice[uint16])
	testUint64x2Binary(t, archsimd.Uint64x2.Sub, subSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.Sub, subSlice[uint8])
}

func TestMax(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.Max, maxSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Max, maxSlice[int32])

	testInt64x2Binary(t, archsimd.Int64x2.Max, maxSlice[int64])

	testInt8x16Binary(t, archsimd.Int8x16.Max, maxSlice[int8])

	testUint16x8Binary(t, archsimd.Uint16x8.Max, maxSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Max, maxSlice[uint32])

	testUint64x2Binary(t, archsimd.Uint64x2.Max, maxSlice[uint64])

	testUint8x16Binary(t, archsimd.Uint8x16.Max, maxSlice[uint8])
}

func TestMin(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.Min, minSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Min, minSlice[int32])

	testInt64x2Binary(t, archsimd.Int64x2.Min, minSlice[int64])

	testInt8x16Binary(t, archsimd.Int8x16.Min, minSlice[int8])

	testUint16x8Binary(t, archsimd.Uint16x8.Min, minSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Min, minSlice[uint32])

	testUint64x2Binary(t, archsimd.Uint64x2.Min, minSlice[uint64])

	testUint8x16Binary(t, archsimd.Uint8x16.Min, minSlice[uint8])
}

func TestAnd(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.And, andSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.And, andSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.And, andSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.And, andSlice[int8])

	testUint16x8Binary(t, archsimd.Uint16x8.And, andSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.And, andSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.And, andSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.And, andSlice[uint8])
}

func TestAndNot(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.AndNot, andNotSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.AndNot, andNotSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.AndNot, andNotSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.AndNot, andNotSlice[int8])

	testUint8x16Binary(t, archsimd.Uint8x16.AndNot, andNotSlice[uint8])
	testUint16x8Binary(t, archsimd.Uint16x8.AndNot, andNotSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.AndNot, andNotSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.AndNot, andNotSlice[uint64])
}

func TestXor(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.Xor, xorSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Xor, xorSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Xor, xorSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Xor, xorSlice[int8])

	testUint16x8Binary(t, archsimd.Uint16x8.Xor, xorSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Xor, xorSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Xor, xorSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.Xor, xorSlice[uint8])
}

func TestOr(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.Or, orSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Or, orSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Or, orSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Or, orSlice[int8])

	testUint16x8Binary(t, archsimd.Uint16x8.Or, orSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Or, orSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Or, orSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.Or, orSlice[uint8])
}

func TestMul(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Mul, mulSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Mul, mulSlice[float64])

	testInt8x16Binary(t, archsimd.Int8x16.Mul, mulSlice[int8])
	testUint8x16Binary(t, archsimd.Uint8x16.Mul, mulSlice[uint8])
	testInt16x8Binary(t, archsimd.Int16x8.Mul, mulSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Mul, mulSlice[int32])

	if archsimd.X86.AVX512() {
		testInt64x2Binary(t, archsimd.Int64x2.Mul, mulSlice[int64])
	}
}

func TestDiv(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Div, divSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Div, divSlice[float64])
}

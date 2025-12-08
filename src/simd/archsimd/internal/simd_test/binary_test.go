// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestAdd(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Add, addSlice[float32])
	testFloat32x8Binary(t, archsimd.Float32x8.Add, addSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Add, addSlice[float64])
	testFloat64x4Binary(t, archsimd.Float64x4.Add, addSlice[float64])

	testInt16x16Binary(t, archsimd.Int16x16.Add, addSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.Add, addSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Add, addSlice[int32])
	testInt32x8Binary(t, archsimd.Int32x8.Add, addSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Add, addSlice[int64])
	testInt64x4Binary(t, archsimd.Int64x4.Add, addSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Add, addSlice[int8])
	testInt8x32Binary(t, archsimd.Int8x32.Add, addSlice[int8])

	testUint16x16Binary(t, archsimd.Uint16x16.Add, addSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.Add, addSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Add, addSlice[uint32])
	testUint32x8Binary(t, archsimd.Uint32x8.Add, addSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Add, addSlice[uint64])
	testUint64x4Binary(t, archsimd.Uint64x4.Add, addSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.Add, addSlice[uint8])
	testUint8x32Binary(t, archsimd.Uint8x32.Add, addSlice[uint8])

	if archsimd.X86.AVX512() {
		testFloat32x16Binary(t, archsimd.Float32x16.Add, addSlice[float32])
		testFloat64x8Binary(t, archsimd.Float64x8.Add, addSlice[float64])
		testInt8x64Binary(t, archsimd.Int8x64.Add, addSlice[int8])
		testInt16x32Binary(t, archsimd.Int16x32.Add, addSlice[int16])
		testInt32x16Binary(t, archsimd.Int32x16.Add, addSlice[int32])
		testInt64x8Binary(t, archsimd.Int64x8.Add, addSlice[int64])
		testUint8x64Binary(t, archsimd.Uint8x64.Add, addSlice[uint8])
		testUint16x32Binary(t, archsimd.Uint16x32.Add, addSlice[uint16])
		testUint32x16Binary(t, archsimd.Uint32x16.Add, addSlice[uint32])
		testUint64x8Binary(t, archsimd.Uint64x8.Add, addSlice[uint64])
	}
}

func TestSub(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Sub, subSlice[float32])
	testFloat32x8Binary(t, archsimd.Float32x8.Sub, subSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Sub, subSlice[float64])
	testFloat64x4Binary(t, archsimd.Float64x4.Sub, subSlice[float64])

	testInt16x16Binary(t, archsimd.Int16x16.Sub, subSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.Sub, subSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Sub, subSlice[int32])
	testInt32x8Binary(t, archsimd.Int32x8.Sub, subSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Sub, subSlice[int64])
	testInt64x4Binary(t, archsimd.Int64x4.Sub, subSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Sub, subSlice[int8])
	testInt8x32Binary(t, archsimd.Int8x32.Sub, subSlice[int8])

	testUint16x16Binary(t, archsimd.Uint16x16.Sub, subSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.Sub, subSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Sub, subSlice[uint32])
	testUint32x8Binary(t, archsimd.Uint32x8.Sub, subSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Sub, subSlice[uint64])
	testUint64x4Binary(t, archsimd.Uint64x4.Sub, subSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.Sub, subSlice[uint8])
	testUint8x32Binary(t, archsimd.Uint8x32.Sub, subSlice[uint8])

	if archsimd.X86.AVX512() {
		testFloat32x16Binary(t, archsimd.Float32x16.Sub, subSlice[float32])
		testFloat64x8Binary(t, archsimd.Float64x8.Sub, subSlice[float64])
		testInt8x64Binary(t, archsimd.Int8x64.Sub, subSlice[int8])
		testInt16x32Binary(t, archsimd.Int16x32.Sub, subSlice[int16])
		testInt32x16Binary(t, archsimd.Int32x16.Sub, subSlice[int32])
		testInt64x8Binary(t, archsimd.Int64x8.Sub, subSlice[int64])
		testUint8x64Binary(t, archsimd.Uint8x64.Sub, subSlice[uint8])
		testUint16x32Binary(t, archsimd.Uint16x32.Sub, subSlice[uint16])
		testUint32x16Binary(t, archsimd.Uint32x16.Sub, subSlice[uint32])
		testUint64x8Binary(t, archsimd.Uint64x8.Sub, subSlice[uint64])
	}
}

func TestMax(t *testing.T) {
	// testFloat32x4Binary(t, archsimd.Float32x4.Max, maxSlice[float32]) // nan is wrong
	// testFloat32x8Binary(t, archsimd.Float32x8.Max, maxSlice[float32]) // nan is wrong
	// testFloat64x2Binary(t, archsimd.Float64x2.Max, maxSlice[float64]) // nan is wrong
	// testFloat64x4Binary(t, archsimd.Float64x4.Max, maxSlice[float64]) // nan is wrong

	testInt16x16Binary(t, archsimd.Int16x16.Max, maxSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.Max, maxSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Max, maxSlice[int32])
	testInt32x8Binary(t, archsimd.Int32x8.Max, maxSlice[int32])

	if archsimd.X86.AVX512() {
		testInt64x2Binary(t, archsimd.Int64x2.Max, maxSlice[int64])
		testInt64x4Binary(t, archsimd.Int64x4.Max, maxSlice[int64])
	}

	testInt8x16Binary(t, archsimd.Int8x16.Max, maxSlice[int8])
	testInt8x32Binary(t, archsimd.Int8x32.Max, maxSlice[int8])

	testUint16x16Binary(t, archsimd.Uint16x16.Max, maxSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.Max, maxSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Max, maxSlice[uint32])
	testUint32x8Binary(t, archsimd.Uint32x8.Max, maxSlice[uint32])

	if archsimd.X86.AVX512() {
		testUint64x2Binary(t, archsimd.Uint64x2.Max, maxSlice[uint64])
		testUint64x4Binary(t, archsimd.Uint64x4.Max, maxSlice[uint64])
	}

	testUint8x16Binary(t, archsimd.Uint8x16.Max, maxSlice[uint8])
	testUint8x32Binary(t, archsimd.Uint8x32.Max, maxSlice[uint8])

	if archsimd.X86.AVX512() {
		// testFloat32x16Binary(t, archsimd.Float32x16.Max, maxSlice[float32]) // nan is wrong
		// testFloat64x8Binary(t, archsimd.Float64x8.Max, maxSlice[float64]) // nan is wrong
		testInt8x64Binary(t, archsimd.Int8x64.Max, maxSlice[int8])
		testInt16x32Binary(t, archsimd.Int16x32.Max, maxSlice[int16])
		testInt32x16Binary(t, archsimd.Int32x16.Max, maxSlice[int32])
		testInt64x8Binary(t, archsimd.Int64x8.Max, maxSlice[int64])
		testUint8x64Binary(t, archsimd.Uint8x64.Max, maxSlice[uint8])
		testUint16x32Binary(t, archsimd.Uint16x32.Max, maxSlice[uint16])
		testUint32x16Binary(t, archsimd.Uint32x16.Max, maxSlice[uint32])
		testUint64x8Binary(t, archsimd.Uint64x8.Max, maxSlice[uint64])
	}
}

func TestMin(t *testing.T) {
	// testFloat32x4Binary(t, archsimd.Float32x4.Min, minSlice[float32]) // nan is wrong
	// testFloat32x8Binary(t, archsimd.Float32x8.Min, minSlice[float32]) // nan is wrong
	// testFloat64x2Binary(t, archsimd.Float64x2.Min, minSlice[float64]) // nan is wrong
	// testFloat64x4Binary(t, archsimd.Float64x4.Min, minSlice[float64]) // nan is wrong

	testInt16x16Binary(t, archsimd.Int16x16.Min, minSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.Min, minSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Min, minSlice[int32])
	testInt32x8Binary(t, archsimd.Int32x8.Min, minSlice[int32])

	if archsimd.X86.AVX512() {
		testInt64x2Binary(t, archsimd.Int64x2.Min, minSlice[int64])
		testInt64x4Binary(t, archsimd.Int64x4.Min, minSlice[int64])
	}

	testInt8x16Binary(t, archsimd.Int8x16.Min, minSlice[int8])
	testInt8x32Binary(t, archsimd.Int8x32.Min, minSlice[int8])

	testUint16x16Binary(t, archsimd.Uint16x16.Min, minSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.Min, minSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Min, minSlice[uint32])
	testUint32x8Binary(t, archsimd.Uint32x8.Min, minSlice[uint32])

	if archsimd.X86.AVX512() {
		testUint64x2Binary(t, archsimd.Uint64x2.Min, minSlice[uint64])
		testUint64x4Binary(t, archsimd.Uint64x4.Min, minSlice[uint64])
	}

	testUint8x16Binary(t, archsimd.Uint8x16.Min, minSlice[uint8])
	testUint8x32Binary(t, archsimd.Uint8x32.Min, minSlice[uint8])

	if archsimd.X86.AVX512() {
		// testFloat32x16Binary(t, archsimd.Float32x16.Min, minSlice[float32]) // nan is wrong
		// testFloat64x8Binary(t, archsimd.Float64x8.Min, minSlice[float64]) // nan is wrong
		testInt8x64Binary(t, archsimd.Int8x64.Min, minSlice[int8])
		testInt16x32Binary(t, archsimd.Int16x32.Min, minSlice[int16])
		testInt32x16Binary(t, archsimd.Int32x16.Min, minSlice[int32])
		testInt64x8Binary(t, archsimd.Int64x8.Min, minSlice[int64])
		testUint8x64Binary(t, archsimd.Uint8x64.Min, minSlice[uint8])
		testUint16x32Binary(t, archsimd.Uint16x32.Min, minSlice[uint16])
		testUint32x16Binary(t, archsimd.Uint32x16.Min, minSlice[uint32])
		testUint64x8Binary(t, archsimd.Uint64x8.Min, minSlice[uint64])
	}
}

func TestAnd(t *testing.T) {
	testInt16x16Binary(t, archsimd.Int16x16.And, andSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.And, andSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.And, andSlice[int32])
	testInt32x8Binary(t, archsimd.Int32x8.And, andSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.And, andSlice[int64])
	testInt64x4Binary(t, archsimd.Int64x4.And, andSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.And, andSlice[int8])
	testInt8x32Binary(t, archsimd.Int8x32.And, andSlice[int8])

	testUint16x16Binary(t, archsimd.Uint16x16.And, andSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.And, andSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.And, andSlice[uint32])
	testUint32x8Binary(t, archsimd.Uint32x8.And, andSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.And, andSlice[uint64])
	testUint64x4Binary(t, archsimd.Uint64x4.And, andSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.And, andSlice[uint8])
	testUint8x32Binary(t, archsimd.Uint8x32.And, andSlice[uint8])

	if archsimd.X86.AVX512() {
		//	testInt8x64Binary(t, archsimd.Int8x64.And, andISlice[int8]) // missing
		//	testInt16x32Binary(t, archsimd.Int16x32.And, andISlice[int16]) // missing
		testInt32x16Binary(t, archsimd.Int32x16.And, andSlice[int32])
		testInt64x8Binary(t, archsimd.Int64x8.And, andSlice[int64])
		//	testUint8x64Binary(t, archsimd.Uint8x64.And, andISlice[uint8]) // missing
		//	testUint16x32Binary(t, archsimd.Uint16x32.And, andISlice[uint16]) // missing
		testUint32x16Binary(t, archsimd.Uint32x16.And, andSlice[uint32])
		testUint64x8Binary(t, archsimd.Uint64x8.And, andSlice[uint64])
	}
}

func TestAndNot(t *testing.T) {
	testInt16x16Binary(t, archsimd.Int16x16.AndNot, andNotSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.AndNot, andNotSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.AndNot, andNotSlice[int32])
	testInt32x8Binary(t, archsimd.Int32x8.AndNot, andNotSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.AndNot, andNotSlice[int64])
	testInt64x4Binary(t, archsimd.Int64x4.AndNot, andNotSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.AndNot, andNotSlice[int8])
	testInt8x32Binary(t, archsimd.Int8x32.AndNot, andNotSlice[int8])

	testUint16x16Binary(t, archsimd.Uint16x16.AndNot, andNotSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.AndNot, andNotSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.AndNot, andNotSlice[uint32])
	testUint32x8Binary(t, archsimd.Uint32x8.AndNot, andNotSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.AndNot, andNotSlice[uint64])
	testUint64x4Binary(t, archsimd.Uint64x4.AndNot, andNotSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.AndNot, andNotSlice[uint8])
	testUint8x32Binary(t, archsimd.Uint8x32.AndNot, andNotSlice[uint8])

	if archsimd.X86.AVX512() {
		testInt8x64Binary(t, archsimd.Int8x64.AndNot, andNotSlice[int8])
		testInt16x32Binary(t, archsimd.Int16x32.AndNot, andNotSlice[int16])
		testInt32x16Binary(t, archsimd.Int32x16.AndNot, andNotSlice[int32])
		testInt64x8Binary(t, archsimd.Int64x8.AndNot, andNotSlice[int64])
		testUint8x64Binary(t, archsimd.Uint8x64.AndNot, andNotSlice[uint8])
		testUint16x32Binary(t, archsimd.Uint16x32.AndNot, andNotSlice[uint16])
		testUint32x16Binary(t, archsimd.Uint32x16.AndNot, andNotSlice[uint32])
		testUint64x8Binary(t, archsimd.Uint64x8.AndNot, andNotSlice[uint64])
	}
}

func TestXor(t *testing.T) {
	testInt16x16Binary(t, archsimd.Int16x16.Xor, xorSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.Xor, xorSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Xor, xorSlice[int32])
	testInt32x8Binary(t, archsimd.Int32x8.Xor, xorSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Xor, xorSlice[int64])
	testInt64x4Binary(t, archsimd.Int64x4.Xor, xorSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Xor, xorSlice[int8])
	testInt8x32Binary(t, archsimd.Int8x32.Xor, xorSlice[int8])

	testUint16x16Binary(t, archsimd.Uint16x16.Xor, xorSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.Xor, xorSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Xor, xorSlice[uint32])
	testUint32x8Binary(t, archsimd.Uint32x8.Xor, xorSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Xor, xorSlice[uint64])
	testUint64x4Binary(t, archsimd.Uint64x4.Xor, xorSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.Xor, xorSlice[uint8])
	testUint8x32Binary(t, archsimd.Uint8x32.Xor, xorSlice[uint8])

	if archsimd.X86.AVX512() {
		//	testInt8x64Binary(t, archsimd.Int8x64.Xor, andISlice[int8]) // missing
		//	testInt16x32Binary(t, archsimd.Int16x32.Xor, andISlice[int16]) // missing
		testInt32x16Binary(t, archsimd.Int32x16.Xor, xorSlice[int32])
		testInt64x8Binary(t, archsimd.Int64x8.Xor, xorSlice[int64])
		//	testUint8x64Binary(t, archsimd.Uint8x64.Xor, andISlice[uint8]) // missing
		//	testUint16x32Binary(t, archsimd.Uint16x32.Xor, andISlice[uint16]) // missing
		testUint32x16Binary(t, archsimd.Uint32x16.Xor, xorSlice[uint32])
		testUint64x8Binary(t, archsimd.Uint64x8.Xor, xorSlice[uint64])
	}
}

func TestOr(t *testing.T) {
	testInt16x16Binary(t, archsimd.Int16x16.Or, orSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.Or, orSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Or, orSlice[int32])
	testInt32x8Binary(t, archsimd.Int32x8.Or, orSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Or, orSlice[int64])
	testInt64x4Binary(t, archsimd.Int64x4.Or, orSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Or, orSlice[int8])
	testInt8x32Binary(t, archsimd.Int8x32.Or, orSlice[int8])

	testUint16x16Binary(t, archsimd.Uint16x16.Or, orSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.Or, orSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Or, orSlice[uint32])
	testUint32x8Binary(t, archsimd.Uint32x8.Or, orSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Or, orSlice[uint64])
	testUint64x4Binary(t, archsimd.Uint64x4.Or, orSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.Or, orSlice[uint8])
	testUint8x32Binary(t, archsimd.Uint8x32.Or, orSlice[uint8])

	if archsimd.X86.AVX512() {
		//	testInt8x64Binary(t, archsimd.Int8x64.Or, andISlice[int8]) // missing
		//	testInt16x32Binary(t, archsimd.Int16x32.Or, andISlice[int16]) // missing
		testInt32x16Binary(t, archsimd.Int32x16.Or, orSlice[int32])
		testInt64x8Binary(t, archsimd.Int64x8.Or, orSlice[int64])
		//	testUint8x64Binary(t, archsimd.Uint8x64.Or, andISlice[uint8]) // missing
		//	testUint16x32Binary(t, archsimd.Uint16x32.Or, andISlice[uint16]) // missing
		testUint32x16Binary(t, archsimd.Uint32x16.Or, orSlice[uint32])
		testUint64x8Binary(t, archsimd.Uint64x8.Or, orSlice[uint64])
	}
}

func TestMul(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Mul, mulSlice[float32])
	testFloat32x8Binary(t, archsimd.Float32x8.Mul, mulSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Mul, mulSlice[float64])
	testFloat64x4Binary(t, archsimd.Float64x4.Mul, mulSlice[float64])

	testInt16x16Binary(t, archsimd.Int16x16.Mul, mulSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.Mul, mulSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Mul, mulSlice[int32])
	testInt32x8Binary(t, archsimd.Int32x8.Mul, mulSlice[int32])

	// testInt8x16Binary(t, archsimd.Int8x16.Mul, mulSlice[int8]) // nope
	// testInt8x32Binary(t, archsimd.Int8x32.Mul, mulSlice[int8])

	// TODO we should be able to do these, there's no difference between signed/unsigned Mul
	// testUint16x16Binary(t, archsimd.Uint16x16.Mul, mulSlice[uint16])
	// testUint16x8Binary(t, archsimd.Uint16x8.Mul, mulSlice[uint16])
	// testUint32x4Binary(t, archsimd.Uint32x4.Mul, mulSlice[uint32])
	// testUint32x8Binary(t, archsimd.Uint32x8.Mul, mulSlice[uint32])
	// testUint64x2Binary(t, archsimd.Uint64x2.Mul, mulSlice[uint64])
	// testUint64x4Binary(t, archsimd.Uint64x4.Mul, mulSlice[uint64])

	// testUint8x16Binary(t, archsimd.Uint8x16.Mul, mulSlice[uint8]) // nope
	// testUint8x32Binary(t, archsimd.Uint8x32.Mul, mulSlice[uint8])

	if archsimd.X86.AVX512() {
		testInt64x2Binary(t, archsimd.Int64x2.Mul, mulSlice[int64]) // avx512 only
		testInt64x4Binary(t, archsimd.Int64x4.Mul, mulSlice[int64])

		testFloat32x16Binary(t, archsimd.Float32x16.Mul, mulSlice[float32])
		testFloat64x8Binary(t, archsimd.Float64x8.Mul, mulSlice[float64])

		// testInt8x64Binary(t, archsimd.Int8x64.Mul, mulSlice[int8]) // nope
		testInt16x32Binary(t, archsimd.Int16x32.Mul, mulSlice[int16])
		testInt32x16Binary(t, archsimd.Int32x16.Mul, mulSlice[int32])
		testInt64x8Binary(t, archsimd.Int64x8.Mul, mulSlice[int64])
		// testUint8x64Binary(t, archsimd.Uint8x64.Mul, mulSlice[uint8]) // nope

		// TODO signed should do the job
		// testUint16x32Binary(t, archsimd.Uint16x32.Mul, mulSlice[uint16])
		// testUint32x16Binary(t, archsimd.Uint32x16.Mul, mulSlice[uint32])
		// testUint64x8Binary(t, archsimd.Uint64x8.Mul, mulSlice[uint64])
	}
}

func TestDiv(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Div, divSlice[float32])
	testFloat32x8Binary(t, archsimd.Float32x8.Div, divSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Div, divSlice[float64])
	testFloat64x4Binary(t, archsimd.Float64x4.Div, divSlice[float64])

	if archsimd.X86.AVX512() {
		testFloat32x16Binary(t, archsimd.Float32x16.Div, divSlice[float32])
		testFloat64x8Binary(t, archsimd.Float64x8.Div, divSlice[float64])
	}
}

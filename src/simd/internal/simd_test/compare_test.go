// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd"
	"testing"
)

// AVX 2 lacks most comparisons, but they can be synthesized
// from > and =
var comparisonFixed bool = simd.X86.AVX512()

func TestLess(t *testing.T) {
	testFloat32x4Compare(t, simd.Float32x4.Less, lessSlice[float32])
	testFloat32x8Compare(t, simd.Float32x8.Less, lessSlice[float32])
	testFloat64x2Compare(t, simd.Float64x2.Less, lessSlice[float64])
	testFloat64x4Compare(t, simd.Float64x4.Less, lessSlice[float64])

	testInt16x16Compare(t, simd.Int16x16.Less, lessSlice[int16])
	testInt16x8Compare(t, simd.Int16x8.Less, lessSlice[int16])
	testInt32x4Compare(t, simd.Int32x4.Less, lessSlice[int32])
	testInt32x8Compare(t, simd.Int32x8.Less, lessSlice[int32])
	testInt64x2Compare(t, simd.Int64x2.Less, lessSlice[int64])
	testInt64x4Compare(t, simd.Int64x4.Less, lessSlice[int64])
	testInt8x16Compare(t, simd.Int8x16.Less, lessSlice[int8])
	testInt8x32Compare(t, simd.Int8x32.Less, lessSlice[int8])

	testInt16x16Compare(t, simd.Int16x16.Less, lessSlice[int16])
	testInt16x8Compare(t, simd.Int16x8.Less, lessSlice[int16])
	testInt32x4Compare(t, simd.Int32x4.Less, lessSlice[int32])
	testInt32x8Compare(t, simd.Int32x8.Less, lessSlice[int32])
	testInt64x2Compare(t, simd.Int64x2.Less, lessSlice[int64])
	testInt64x4Compare(t, simd.Int64x4.Less, lessSlice[int64])
	testInt8x16Compare(t, simd.Int8x16.Less, lessSlice[int8])
	testInt8x32Compare(t, simd.Int8x32.Less, lessSlice[int8])

	testUint16x16Compare(t, simd.Uint16x16.Less, lessSlice[uint16])
	testUint16x8Compare(t, simd.Uint16x8.Less, lessSlice[uint16])
	testUint32x4Compare(t, simd.Uint32x4.Less, lessSlice[uint32])
	testUint32x8Compare(t, simd.Uint32x8.Less, lessSlice[uint32])
	testUint64x2Compare(t, simd.Uint64x2.Less, lessSlice[uint64])
	testUint64x4Compare(t, simd.Uint64x4.Less, lessSlice[uint64])
	testUint8x16Compare(t, simd.Uint8x16.Less, lessSlice[uint8])
	testUint8x32Compare(t, simd.Uint8x32.Less, lessSlice[uint8])

	if simd.X86.AVX512() {
		testUint16x16Compare(t, simd.Uint16x16.Less, lessSlice[uint16])
		testUint16x8Compare(t, simd.Uint16x8.Less, lessSlice[uint16])
		testUint32x4Compare(t, simd.Uint32x4.Less, lessSlice[uint32])
		testUint32x8Compare(t, simd.Uint32x8.Less, lessSlice[uint32])
		testUint64x2Compare(t, simd.Uint64x2.Less, lessSlice[uint64])
		testUint64x4Compare(t, simd.Uint64x4.Less, lessSlice[uint64])
		testUint8x16Compare(t, simd.Uint8x16.Less, lessSlice[uint8])
		testUint8x32Compare(t, simd.Uint8x32.Less, lessSlice[uint8])

		testFloat32x16Compare(t, simd.Float32x16.Less, lessSlice[float32])
		testFloat64x8Compare(t, simd.Float64x8.Less, lessSlice[float64])
		testInt8x64Compare(t, simd.Int8x64.Less, lessSlice[int8])
		testInt16x32Compare(t, simd.Int16x32.Less, lessSlice[int16])
		testInt32x16Compare(t, simd.Int32x16.Less, lessSlice[int32])
		testInt64x8Compare(t, simd.Int64x8.Less, lessSlice[int64])
		testUint8x64Compare(t, simd.Uint8x64.Less, lessSlice[uint8])
		testUint16x32Compare(t, simd.Uint16x32.Less, lessSlice[uint16])
		testUint32x16Compare(t, simd.Uint32x16.Less, lessSlice[uint32])
		testUint64x8Compare(t, simd.Uint64x8.Less, lessSlice[uint64])
	}
}

func TestLessEqual(t *testing.T) {
	testFloat32x4Compare(t, simd.Float32x4.LessEqual, lessEqualSlice[float32])
	testFloat32x8Compare(t, simd.Float32x8.LessEqual, lessEqualSlice[float32])
	testFloat64x2Compare(t, simd.Float64x2.LessEqual, lessEqualSlice[float64])
	testFloat64x4Compare(t, simd.Float64x4.LessEqual, lessEqualSlice[float64])

	testInt16x16Compare(t, simd.Int16x16.LessEqual, lessEqualSlice[int16])
	testInt16x8Compare(t, simd.Int16x8.LessEqual, lessEqualSlice[int16])
	testInt32x4Compare(t, simd.Int32x4.LessEqual, lessEqualSlice[int32])
	testInt32x8Compare(t, simd.Int32x8.LessEqual, lessEqualSlice[int32])
	testInt64x2Compare(t, simd.Int64x2.LessEqual, lessEqualSlice[int64])
	testInt64x4Compare(t, simd.Int64x4.LessEqual, lessEqualSlice[int64])
	testInt8x16Compare(t, simd.Int8x16.LessEqual, lessEqualSlice[int8])
	testInt8x32Compare(t, simd.Int8x32.LessEqual, lessEqualSlice[int8])

	testUint16x16Compare(t, simd.Uint16x16.LessEqual, lessEqualSlice[uint16])
	testUint16x8Compare(t, simd.Uint16x8.LessEqual, lessEqualSlice[uint16])
	testUint32x4Compare(t, simd.Uint32x4.LessEqual, lessEqualSlice[uint32])
	testUint32x8Compare(t, simd.Uint32x8.LessEqual, lessEqualSlice[uint32])
	testUint64x2Compare(t, simd.Uint64x2.LessEqual, lessEqualSlice[uint64])
	testUint64x4Compare(t, simd.Uint64x4.LessEqual, lessEqualSlice[uint64])
	testUint8x16Compare(t, simd.Uint8x16.LessEqual, lessEqualSlice[uint8])
	testUint8x32Compare(t, simd.Uint8x32.LessEqual, lessEqualSlice[uint8])

	if simd.X86.AVX512() {
		testFloat32x16Compare(t, simd.Float32x16.LessEqual, lessEqualSlice[float32])
		testFloat64x8Compare(t, simd.Float64x8.LessEqual, lessEqualSlice[float64])
		testInt8x64Compare(t, simd.Int8x64.LessEqual, lessEqualSlice[int8])
		testInt16x32Compare(t, simd.Int16x32.LessEqual, lessEqualSlice[int16])
		testInt32x16Compare(t, simd.Int32x16.LessEqual, lessEqualSlice[int32])
		testInt64x8Compare(t, simd.Int64x8.LessEqual, lessEqualSlice[int64])
		testUint8x64Compare(t, simd.Uint8x64.LessEqual, lessEqualSlice[uint8])
		testUint16x32Compare(t, simd.Uint16x32.LessEqual, lessEqualSlice[uint16])
		testUint32x16Compare(t, simd.Uint32x16.LessEqual, lessEqualSlice[uint32])
		testUint64x8Compare(t, simd.Uint64x8.LessEqual, lessEqualSlice[uint64])
	}
}

func TestGreater(t *testing.T) {
	testFloat32x4Compare(t, simd.Float32x4.Greater, greaterSlice[float32])
	testFloat32x8Compare(t, simd.Float32x8.Greater, greaterSlice[float32])
	testFloat64x2Compare(t, simd.Float64x2.Greater, greaterSlice[float64])
	testFloat64x4Compare(t, simd.Float64x4.Greater, greaterSlice[float64])

	testInt16x16Compare(t, simd.Int16x16.Greater, greaterSlice[int16])
	testInt16x8Compare(t, simd.Int16x8.Greater, greaterSlice[int16])
	testInt32x4Compare(t, simd.Int32x4.Greater, greaterSlice[int32])
	testInt32x8Compare(t, simd.Int32x8.Greater, greaterSlice[int32])

	testInt64x2Compare(t, simd.Int64x2.Greater, greaterSlice[int64])
	testInt64x4Compare(t, simd.Int64x4.Greater, greaterSlice[int64])
	testInt8x16Compare(t, simd.Int8x16.Greater, greaterSlice[int8])
	testInt8x32Compare(t, simd.Int8x32.Greater, greaterSlice[int8])

	testUint16x16Compare(t, simd.Uint16x16.Greater, greaterSlice[uint16])
	testUint16x8Compare(t, simd.Uint16x8.Greater, greaterSlice[uint16])
	testUint32x4Compare(t, simd.Uint32x4.Greater, greaterSlice[uint32])
	testUint32x8Compare(t, simd.Uint32x8.Greater, greaterSlice[uint32])

	testUint64x2Compare(t, simd.Uint64x2.Greater, greaterSlice[uint64])
	testUint64x4Compare(t, simd.Uint64x4.Greater, greaterSlice[uint64])
	testUint8x16Compare(t, simd.Uint8x16.Greater, greaterSlice[uint8])
	testUint8x32Compare(t, simd.Uint8x32.Greater, greaterSlice[uint8])

	if simd.X86.AVX512() {

		testFloat32x16Compare(t, simd.Float32x16.Greater, greaterSlice[float32])
		testFloat64x8Compare(t, simd.Float64x8.Greater, greaterSlice[float64])
		testInt8x64Compare(t, simd.Int8x64.Greater, greaterSlice[int8])
		testInt16x32Compare(t, simd.Int16x32.Greater, greaterSlice[int16])
		testInt32x16Compare(t, simd.Int32x16.Greater, greaterSlice[int32])
		testInt64x8Compare(t, simd.Int64x8.Greater, greaterSlice[int64])
		testUint8x64Compare(t, simd.Uint8x64.Greater, greaterSlice[uint8])
		testUint16x32Compare(t, simd.Uint16x32.Greater, greaterSlice[uint16])
		testUint32x16Compare(t, simd.Uint32x16.Greater, greaterSlice[uint32])
		testUint64x8Compare(t, simd.Uint64x8.Greater, greaterSlice[uint64])
	}
}

func TestGreaterEqual(t *testing.T) {
	testFloat32x4Compare(t, simd.Float32x4.GreaterEqual, greaterEqualSlice[float32])
	testFloat32x8Compare(t, simd.Float32x8.GreaterEqual, greaterEqualSlice[float32])
	testFloat64x2Compare(t, simd.Float64x2.GreaterEqual, greaterEqualSlice[float64])
	testFloat64x4Compare(t, simd.Float64x4.GreaterEqual, greaterEqualSlice[float64])

	testInt16x16Compare(t, simd.Int16x16.GreaterEqual, greaterEqualSlice[int16])
	testInt16x8Compare(t, simd.Int16x8.GreaterEqual, greaterEqualSlice[int16])
	testInt32x4Compare(t, simd.Int32x4.GreaterEqual, greaterEqualSlice[int32])
	testInt32x8Compare(t, simd.Int32x8.GreaterEqual, greaterEqualSlice[int32])
	testInt64x2Compare(t, simd.Int64x2.GreaterEqual, greaterEqualSlice[int64])
	testInt64x4Compare(t, simd.Int64x4.GreaterEqual, greaterEqualSlice[int64])
	testInt8x16Compare(t, simd.Int8x16.GreaterEqual, greaterEqualSlice[int8])
	testInt8x32Compare(t, simd.Int8x32.GreaterEqual, greaterEqualSlice[int8])

	testUint16x16Compare(t, simd.Uint16x16.GreaterEqual, greaterEqualSlice[uint16])
	testUint16x8Compare(t, simd.Uint16x8.GreaterEqual, greaterEqualSlice[uint16])
	testUint32x4Compare(t, simd.Uint32x4.GreaterEqual, greaterEqualSlice[uint32])
	testUint32x8Compare(t, simd.Uint32x8.GreaterEqual, greaterEqualSlice[uint32])
	testUint64x2Compare(t, simd.Uint64x2.GreaterEqual, greaterEqualSlice[uint64])
	testUint64x4Compare(t, simd.Uint64x4.GreaterEqual, greaterEqualSlice[uint64])
	testUint8x16Compare(t, simd.Uint8x16.GreaterEqual, greaterEqualSlice[uint8])
	testUint8x32Compare(t, simd.Uint8x32.GreaterEqual, greaterEqualSlice[uint8])

	if simd.X86.AVX512() {
		testFloat32x16Compare(t, simd.Float32x16.GreaterEqual, greaterEqualSlice[float32])
		testFloat64x8Compare(t, simd.Float64x8.GreaterEqual, greaterEqualSlice[float64])
		testInt8x64Compare(t, simd.Int8x64.GreaterEqual, greaterEqualSlice[int8])
		testInt16x32Compare(t, simd.Int16x32.GreaterEqual, greaterEqualSlice[int16])
		testInt32x16Compare(t, simd.Int32x16.GreaterEqual, greaterEqualSlice[int32])
		testInt64x8Compare(t, simd.Int64x8.GreaterEqual, greaterEqualSlice[int64])
		testUint8x64Compare(t, simd.Uint8x64.GreaterEqual, greaterEqualSlice[uint8])
		testUint16x32Compare(t, simd.Uint16x32.GreaterEqual, greaterEqualSlice[uint16])
		testUint32x16Compare(t, simd.Uint32x16.GreaterEqual, greaterEqualSlice[uint32])
		testUint64x8Compare(t, simd.Uint64x8.GreaterEqual, greaterEqualSlice[uint64])
	}
}

func TestEqual(t *testing.T) {
	testFloat32x4Compare(t, simd.Float32x4.Equal, equalSlice[float32])
	testFloat32x8Compare(t, simd.Float32x8.Equal, equalSlice[float32])
	testFloat64x2Compare(t, simd.Float64x2.Equal, equalSlice[float64])
	testFloat64x4Compare(t, simd.Float64x4.Equal, equalSlice[float64])

	testInt16x16Compare(t, simd.Int16x16.Equal, equalSlice[int16])
	testInt16x8Compare(t, simd.Int16x8.Equal, equalSlice[int16])
	testInt32x4Compare(t, simd.Int32x4.Equal, equalSlice[int32])
	testInt32x8Compare(t, simd.Int32x8.Equal, equalSlice[int32])
	testInt64x2Compare(t, simd.Int64x2.Equal, equalSlice[int64])
	testInt64x4Compare(t, simd.Int64x4.Equal, equalSlice[int64])
	testInt8x16Compare(t, simd.Int8x16.Equal, equalSlice[int8])
	testInt8x32Compare(t, simd.Int8x32.Equal, equalSlice[int8])

	testUint16x16Compare(t, simd.Uint16x16.Equal, equalSlice[uint16])
	testUint16x8Compare(t, simd.Uint16x8.Equal, equalSlice[uint16])
	testUint32x4Compare(t, simd.Uint32x4.Equal, equalSlice[uint32])
	testUint32x8Compare(t, simd.Uint32x8.Equal, equalSlice[uint32])
	testUint64x2Compare(t, simd.Uint64x2.Equal, equalSlice[uint64])
	testUint64x4Compare(t, simd.Uint64x4.Equal, equalSlice[uint64])
	testUint8x16Compare(t, simd.Uint8x16.Equal, equalSlice[uint8])
	testUint8x32Compare(t, simd.Uint8x32.Equal, equalSlice[uint8])

	if simd.X86.AVX512() {
		testFloat32x16Compare(t, simd.Float32x16.Equal, equalSlice[float32])
		testFloat64x8Compare(t, simd.Float64x8.Equal, equalSlice[float64])
		testInt8x64Compare(t, simd.Int8x64.Equal, equalSlice[int8])
		testInt16x32Compare(t, simd.Int16x32.Equal, equalSlice[int16])
		testInt32x16Compare(t, simd.Int32x16.Equal, equalSlice[int32])
		testInt64x8Compare(t, simd.Int64x8.Equal, equalSlice[int64])
		testUint8x64Compare(t, simd.Uint8x64.Equal, equalSlice[uint8])
		testUint16x32Compare(t, simd.Uint16x32.Equal, equalSlice[uint16])
		testUint32x16Compare(t, simd.Uint32x16.Equal, equalSlice[uint32])
		testUint64x8Compare(t, simd.Uint64x8.Equal, equalSlice[uint64])
	}
}

func TestNotEqual(t *testing.T) {
	testFloat32x4Compare(t, simd.Float32x4.NotEqual, notEqualSlice[float32])
	testFloat32x8Compare(t, simd.Float32x8.NotEqual, notEqualSlice[float32])
	testFloat64x2Compare(t, simd.Float64x2.NotEqual, notEqualSlice[float64])
	testFloat64x4Compare(t, simd.Float64x4.NotEqual, notEqualSlice[float64])

	testInt16x16Compare(t, simd.Int16x16.NotEqual, notEqualSlice[int16])
	testInt16x8Compare(t, simd.Int16x8.NotEqual, notEqualSlice[int16])
	testInt32x4Compare(t, simd.Int32x4.NotEqual, notEqualSlice[int32])
	testInt32x8Compare(t, simd.Int32x8.NotEqual, notEqualSlice[int32])
	testInt64x2Compare(t, simd.Int64x2.NotEqual, notEqualSlice[int64])
	testInt64x4Compare(t, simd.Int64x4.NotEqual, notEqualSlice[int64])
	testInt8x16Compare(t, simd.Int8x16.NotEqual, notEqualSlice[int8])
	testInt8x32Compare(t, simd.Int8x32.NotEqual, notEqualSlice[int8])

	testUint16x16Compare(t, simd.Uint16x16.NotEqual, notEqualSlice[uint16])
	testUint16x8Compare(t, simd.Uint16x8.NotEqual, notEqualSlice[uint16])
	testUint32x4Compare(t, simd.Uint32x4.NotEqual, notEqualSlice[uint32])
	testUint32x8Compare(t, simd.Uint32x8.NotEqual, notEqualSlice[uint32])
	testUint64x2Compare(t, simd.Uint64x2.NotEqual, notEqualSlice[uint64])
	testUint64x4Compare(t, simd.Uint64x4.NotEqual, notEqualSlice[uint64])
	testUint8x16Compare(t, simd.Uint8x16.NotEqual, notEqualSlice[uint8])
	testUint8x32Compare(t, simd.Uint8x32.NotEqual, notEqualSlice[uint8])

	if simd.X86.AVX512() {
		testFloat32x16Compare(t, simd.Float32x16.NotEqual, notEqualSlice[float32])
		testFloat64x8Compare(t, simd.Float64x8.NotEqual, notEqualSlice[float64])
		testInt8x64Compare(t, simd.Int8x64.NotEqual, notEqualSlice[int8])
		testInt16x32Compare(t, simd.Int16x32.NotEqual, notEqualSlice[int16])
		testInt32x16Compare(t, simd.Int32x16.NotEqual, notEqualSlice[int32])
		testInt64x8Compare(t, simd.Int64x8.NotEqual, notEqualSlice[int64])
		testUint8x64Compare(t, simd.Uint8x64.NotEqual, notEqualSlice[uint8])
		testUint16x32Compare(t, simd.Uint16x32.NotEqual, notEqualSlice[uint16])
		testUint32x16Compare(t, simd.Uint32x16.NotEqual, notEqualSlice[uint32])
		testUint64x8Compare(t, simd.Uint64x8.NotEqual, notEqualSlice[uint64])
	}
}

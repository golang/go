// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestLess(t *testing.T) {
	testFloat32x4Compare(t, archsimd.Float32x4.Less, lessSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.Less, lessSlice[float64])

	testInt16x8Compare(t, archsimd.Int16x8.Less, lessSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.Less, lessSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.Less, lessSlice[int64])
	testInt8x16Compare(t, archsimd.Int8x16.Less, lessSlice[int8])

	testUint16x8Compare(t, archsimd.Uint16x8.Less, lessSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.Less, lessSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.Less, lessSlice[uint64])
	testUint8x16Compare(t, archsimd.Uint8x16.Less, lessSlice[uint8])
}

func TestLessEqual(t *testing.T) {
	testFloat32x4Compare(t, archsimd.Float32x4.LessEqual, lessEqualSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.LessEqual, lessEqualSlice[float64])

	testInt16x8Compare(t, archsimd.Int16x8.LessEqual, lessEqualSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.LessEqual, lessEqualSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.LessEqual, lessEqualSlice[int64])
	testInt8x16Compare(t, archsimd.Int8x16.LessEqual, lessEqualSlice[int8])

	testUint16x8Compare(t, archsimd.Uint16x8.LessEqual, lessEqualSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.LessEqual, lessEqualSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.LessEqual, lessEqualSlice[uint64])
	testUint8x16Compare(t, archsimd.Uint8x16.LessEqual, lessEqualSlice[uint8])
}

func TestGreater(t *testing.T) {
	testFloat32x4Compare(t, archsimd.Float32x4.Greater, greaterSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.Greater, greaterSlice[float64])

	testInt16x8Compare(t, archsimd.Int16x8.Greater, greaterSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.Greater, greaterSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.Greater, greaterSlice[int64])
	testInt8x16Compare(t, archsimd.Int8x16.Greater, greaterSlice[int8])

	testUint16x8Compare(t, archsimd.Uint16x8.Greater, greaterSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.Greater, greaterSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.Greater, greaterSlice[uint64])
	testUint8x16Compare(t, archsimd.Uint8x16.Greater, greaterSlice[uint8])
}

func TestGreaterEqual(t *testing.T) {
	testFloat32x4Compare(t, archsimd.Float32x4.GreaterEqual, greaterEqualSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.GreaterEqual, greaterEqualSlice[float64])

	testInt16x8Compare(t, archsimd.Int16x8.GreaterEqual, greaterEqualSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.GreaterEqual, greaterEqualSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.GreaterEqual, greaterEqualSlice[int64])
	testInt8x16Compare(t, archsimd.Int8x16.GreaterEqual, greaterEqualSlice[int8])

	testUint16x8Compare(t, archsimd.Uint16x8.GreaterEqual, greaterEqualSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.GreaterEqual, greaterEqualSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.GreaterEqual, greaterEqualSlice[uint64])
	testUint8x16Compare(t, archsimd.Uint8x16.GreaterEqual, greaterEqualSlice[uint8])
}

func TestEqual(t *testing.T) {
	testFloat32x4Compare(t, archsimd.Float32x4.Equal, equalSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.Equal, equalSlice[float64])

	testInt16x8Compare(t, archsimd.Int16x8.Equal, equalSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.Equal, equalSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.Equal, equalSlice[int64])
	testInt8x16Compare(t, archsimd.Int8x16.Equal, equalSlice[int8])

	testUint16x8Compare(t, archsimd.Uint16x8.Equal, equalSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.Equal, equalSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.Equal, equalSlice[uint64])
	testUint8x16Compare(t, archsimd.Uint8x16.Equal, equalSlice[uint8])
}

func TestNotEqual(t *testing.T) {
	testFloat32x4Compare(t, archsimd.Float32x4.NotEqual, notEqualSlice[float32])
	testFloat64x2Compare(t, archsimd.Float64x2.NotEqual, notEqualSlice[float64])

	testInt16x8Compare(t, archsimd.Int16x8.NotEqual, notEqualSlice[int16])
	testInt32x4Compare(t, archsimd.Int32x4.NotEqual, notEqualSlice[int32])
	testInt64x2Compare(t, archsimd.Int64x2.NotEqual, notEqualSlice[int64])
	testInt8x16Compare(t, archsimd.Int8x16.NotEqual, notEqualSlice[int8])

	testUint16x8Compare(t, archsimd.Uint16x8.NotEqual, notEqualSlice[uint16])
	testUint32x4Compare(t, archsimd.Uint32x4.NotEqual, notEqualSlice[uint32])
	testUint64x2Compare(t, archsimd.Uint64x2.NotEqual, notEqualSlice[uint64])
	testUint8x16Compare(t, archsimd.Uint8x16.NotEqual, notEqualSlice[uint8])
}

// Not yet implemented for WASM
// func TestIsNaN(t *testing.T) {
// 	testFloat32x4UnaryCompare(t, archsimd.Float32x4.IsNaN, isNaNSlice[float32])
// 	testFloat64x2UnaryCompare(t, archsimd.Float64x2.IsNaN, isNaNSlice[float64])

// 	// Test x.IsNaN().Or(y.IsNaN()), which is optimized to VCMPP(S|D) $3, x, y.
// 	want32 := mapCompare(func(x, y float32) bool { return x != x || y != y })
// 	want64 := mapCompare(func(x, y float64) bool { return x != x || y != y })
// 	testFloat32x4Compare(t,
// 		func(x, y archsimd.Float32x4) archsimd.Mask32x4 {
// 			return x.IsNaN().Or(y.IsNaN())
// 		}, want32)
// 	testFloat64x2Compare(t,
// 		func(x, y archsimd.Float64x2) archsimd.Mask64x2 {
// 			return x.IsNaN().Or(y.IsNaN())
// 		}, want64)
// }

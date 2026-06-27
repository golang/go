// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestConcatEven(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.ConcatEven, deinterleaveSlice[int8](128, false))
	testInt16x8Binary(t, archsimd.Int16x8.ConcatEven, deinterleaveSlice[int16](128, false))
	testInt32x4Binary(t, archsimd.Int32x4.ConcatEven, deinterleaveSlice[int32](128, false))
	testInt64x2Binary(t, archsimd.Int64x2.ConcatEven, deinterleaveSlice[int64](128, false))
	testUint8x16Binary(t, archsimd.Uint8x16.ConcatEven, deinterleaveSlice[uint8](128, false))
	testUint16x8Binary(t, archsimd.Uint16x8.ConcatEven, deinterleaveSlice[uint16](128, false))
	testUint32x4Binary(t, archsimd.Uint32x4.ConcatEven, deinterleaveSlice[uint32](128, false))
	testUint64x2Binary(t, archsimd.Uint64x2.ConcatEven, deinterleaveSlice[uint64](128, false))
}

func TestConcatOdd(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.ConcatOdd, deinterleaveSlice[int8](128, true))
	testInt16x8Binary(t, archsimd.Int16x8.ConcatOdd, deinterleaveSlice[int16](128, true))
	testInt32x4Binary(t, archsimd.Int32x4.ConcatOdd, deinterleaveSlice[int32](128, true))
	testInt64x2Binary(t, archsimd.Int64x2.ConcatOdd, deinterleaveSlice[int64](128, true))
	testUint8x16Binary(t, archsimd.Uint8x16.ConcatOdd, deinterleaveSlice[uint8](128, true))
	testUint16x8Binary(t, archsimd.Uint16x8.ConcatOdd, deinterleaveSlice[uint16](128, true))
	testUint32x4Binary(t, archsimd.Uint32x4.ConcatOdd, deinterleaveSlice[uint32](128, true))
	testUint64x2Binary(t, archsimd.Uint64x2.ConcatOdd, deinterleaveSlice[uint64](128, true))
}

func TestInterleaveEven(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.InterleaveEven, transposeSlice[int8](128, false))
	testInt16x8Binary(t, archsimd.Int16x8.InterleaveEven, transposeSlice[int16](128, false))
	testInt32x4Binary(t, archsimd.Int32x4.InterleaveEven, transposeSlice[int32](128, false))
	testInt64x2Binary(t, archsimd.Int64x2.InterleaveEven, transposeSlice[int64](128, false))
	testUint8x16Binary(t, archsimd.Uint8x16.InterleaveEven, transposeSlice[uint8](128, false))
	testUint16x8Binary(t, archsimd.Uint16x8.InterleaveEven, transposeSlice[uint16](128, false))
	testUint32x4Binary(t, archsimd.Uint32x4.InterleaveEven, transposeSlice[uint32](128, false))
	testUint64x2Binary(t, archsimd.Uint64x2.InterleaveEven, transposeSlice[uint64](128, false))
}

func TestInterleaveOdd(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.InterleaveOdd, transposeSlice[int8](128, true))
	testInt16x8Binary(t, archsimd.Int16x8.InterleaveOdd, transposeSlice[int16](128, true))
	testInt32x4Binary(t, archsimd.Int32x4.InterleaveOdd, transposeSlice[int32](128, true))
	testInt64x2Binary(t, archsimd.Int64x2.InterleaveOdd, transposeSlice[int64](128, true))
	testUint8x16Binary(t, archsimd.Uint8x16.InterleaveOdd, transposeSlice[uint8](128, true))
	testUint16x8Binary(t, archsimd.Uint16x8.InterleaveOdd, transposeSlice[uint16](128, true))
	testUint32x4Binary(t, archsimd.Uint32x4.InterleaveOdd, transposeSlice[uint32](128, true))
	testUint64x2Binary(t, archsimd.Uint64x2.InterleaveOdd, transposeSlice[uint64](128, true))
}

func TestInterleaveLoARM64(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.InterleaveLo, interleaveSlice[int8](128, false))
	testInt16x8Binary(t, archsimd.Int16x8.InterleaveLo, interleaveSlice[int16](128, false))
	testInt32x4Binary(t, archsimd.Int32x4.InterleaveLo, interleaveSlice[int32](128, false))
	testInt64x2Binary(t, archsimd.Int64x2.InterleaveLo, interleaveSlice[int64](128, false))
	testUint8x16Binary(t, archsimd.Uint8x16.InterleaveLo, interleaveSlice[uint8](128, false))
	testUint16x8Binary(t, archsimd.Uint16x8.InterleaveLo, interleaveSlice[uint16](128, false))
	testUint32x4Binary(t, archsimd.Uint32x4.InterleaveLo, interleaveSlice[uint32](128, false))
	testUint64x2Binary(t, archsimd.Uint64x2.InterleaveLo, interleaveSlice[uint64](128, false))
}

func TestInterleaveHiARM64(t *testing.T) {
	testInt8x16Binary(t, archsimd.Int8x16.InterleaveHi, interleaveSlice[int8](128, true))
	testInt16x8Binary(t, archsimd.Int16x8.InterleaveHi, interleaveSlice[int16](128, true))
	testInt32x4Binary(t, archsimd.Int32x4.InterleaveHi, interleaveSlice[int32](128, true))
	testInt64x2Binary(t, archsimd.Int64x2.InterleaveHi, interleaveSlice[int64](128, true))
	testUint8x16Binary(t, archsimd.Uint8x16.InterleaveHi, interleaveSlice[uint8](128, true))
	testUint16x8Binary(t, archsimd.Uint16x8.InterleaveHi, interleaveSlice[uint16](128, true))
	testUint32x4Binary(t, archsimd.Uint32x4.InterleaveHi, interleaveSlice[uint32](128, true))
	testUint64x2Binary(t, archsimd.Uint64x2.InterleaveHi, interleaveSlice[uint64](128, true))
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

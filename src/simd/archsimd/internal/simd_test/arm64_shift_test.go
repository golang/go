// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestShift(t *testing.T) {
	// Signed — reuse binary helpers, same-type operand pairs
	testInt8x16Binary(t, archsimd.Int8x16.Shift, shiftSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.Shift, shiftSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Shift, shiftSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Shift, shiftSlice[int64])

	// Unsigned — mixed-type operand pairs
	testUint8x16Shift(t, archsimd.Uint8x16.Shift, shiftMixedSlice[uint8, int8])
	testUint16x8Shift(t, archsimd.Uint16x8.Shift, shiftMixedSlice[uint16, int16])
	testUint32x4Shift(t, archsimd.Uint32x4.Shift, shiftMixedSlice[uint32, int32])
	testUint64x2Shift(t, archsimd.Uint64x2.Shift, shiftMixedSlice[uint64, int64])
}

func TestShiftSaturated(t *testing.T) {
	// Signed — reuse binary helpers
	testInt8x16Binary(t, archsimd.Int8x16.ShiftSaturated, shiftSaturatingSignedSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.ShiftSaturated, shiftSaturatingSignedSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.ShiftSaturated, shiftSaturatingSignedSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.ShiftSaturated, shiftSaturatingSignedSlice[int64])

	// Unsigned — mixed-type
	testUint8x16Shift(t, archsimd.Uint8x16.ShiftSaturated, shiftSaturatingUnsignedSlice[uint8, int8])
	testUint16x8Shift(t, archsimd.Uint16x8.ShiftSaturated, shiftSaturatingUnsignedSlice[uint16, int16])
	testUint32x4Shift(t, archsimd.Uint32x4.ShiftSaturated, shiftSaturatingUnsignedSlice[uint32, int32])
	testUint64x2Shift(t, archsimd.Uint64x2.ShiftSaturated, shiftSaturatingUnsignedSlice[uint64, int64])
}

func TestConcatShiftBytesRight(t *testing.T) {
	hide := hideConst[uint64]

	csbr := func(shift uint64) func(x, y []uint8) []uint8 {
		return func(x, y []uint8) []uint8 {
			z := make([]uint8, len(x))
			for i := range z {
				target := i + int(shift)
				if target < 16 {
					z[i] = y[target]
				} else if target < 32 {
					z[i] = x[(target - 16)]
				}
			}
			return z
		}
	}

	t.Run("Uint8x16", func(t *testing.T) {
		for _, shift := range []uint64{0, 2, 8, 15} {
			t.Log("shift", shift)
			testUint8x16Binary(t,
				func(x, y archsimd.Uint8x16) archsimd.Uint8x16 { return x.ConcatShiftBytesRight(y, shift) },
				csbr(shift))
			testUint8x16Binary(t,
				func(x, y archsimd.Uint8x16) archsimd.Uint8x16 { return x.ConcatShiftBytesRight(y, hide(shift)) },
				csbr(hide(shift)))
		}
	})
}

func TestShiftAll8(t *testing.T) {
	testInt8x16ShiftAll(t, archsimd.Int8x16.ShiftAllLeft, shiftAllLeftSlice[int8])
	testUint8x16ShiftAll(t, archsimd.Uint8x16.ShiftAllLeft, shiftAllLeftSlice[uint8])
	testInt8x16ShiftAll(t, archsimd.Int8x16.ShiftAllRight, shiftAllRightSlice[int8])
	testUint8x16ShiftAll(t, archsimd.Uint8x16.ShiftAllRight, shiftAllRightSlice[uint8])
}

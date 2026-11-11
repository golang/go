// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestScale(t *testing.T) {
	// Signed — reuse binary helpers, same-type operand pairs
	testInt8x16Binary(t, archsimd.Int8x16.Scale, shiftSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.Scale, shiftSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Scale, shiftSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Scale, shiftSlice[int64])

	// Unsigned — mixed-type operand pairs
	testUint8x16Shift(t, archsimd.Uint8x16.Scale, shiftMixedSlice[uint8, int8])
	testUint16x8Shift(t, archsimd.Uint16x8.Scale, shiftMixedSlice[uint16, int16])
	testUint32x4Shift(t, archsimd.Uint32x4.Scale, shiftMixedSlice[uint32, int32])
	testUint64x2Shift(t, archsimd.Uint64x2.Scale, shiftMixedSlice[uint64, int64])
}

func TestScaleSaturated(t *testing.T) {
	// ScaleSaturated takes an Int8 scale vector and uses the element at the
	// low byte of each element of x. These tests pass the scale as a vector
	// of x's shape and reinterpret it, so the reference uses the low byte of
	// each scale element.
	s16 := func(s archsimd.Int16x8) archsimd.Int8x16 { return s.ToBits().ReshapeToUint8s().BitsToInt8() }
	s32 := func(s archsimd.Int32x4) archsimd.Int8x16 { return s.ToBits().ReshapeToUint8s().BitsToInt8() }
	s64 := func(s archsimd.Int64x2) archsimd.Int8x16 { return s.ToBits().ReshapeToUint8s().BitsToInt8() }

	// Signed — reuse binary helpers
	testInt8x16Binary(t, archsimd.Int8x16.ScaleSaturated, scaleSaturatingSignedSlice[int8])
	testInt16x8Binary(t, func(x, s archsimd.Int16x8) archsimd.Int16x8 { return x.ScaleSaturated(s16(s)) }, scaleSaturatingSignedSlice[int16])
	testInt32x4Binary(t, func(x, s archsimd.Int32x4) archsimd.Int32x4 { return x.ScaleSaturated(s32(s)) }, scaleSaturatingSignedSlice[int32])
	testInt64x2Binary(t, func(x, s archsimd.Int64x2) archsimd.Int64x2 { return x.ScaleSaturated(s64(s)) }, scaleSaturatingSignedSlice[int64])

	// Unsigned — mixed-type
	testUint8x16Shift(t, archsimd.Uint8x16.ScaleSaturated, scaleSaturatingUnsignedSlice[uint8, int8])
	testUint16x8Shift(t, func(x archsimd.Uint16x8, s archsimd.Int16x8) archsimd.Uint16x8 { return x.ScaleSaturated(s16(s)) }, scaleSaturatingUnsignedSlice[uint16, int16])
	testUint32x4Shift(t, func(x archsimd.Uint32x4, s archsimd.Int32x4) archsimd.Uint32x4 { return x.ScaleSaturated(s32(s)) }, scaleSaturatingUnsignedSlice[uint32, int32])
	testUint64x2Shift(t, func(x archsimd.Uint64x2, s archsimd.Int64x2) archsimd.Uint64x2 { return x.ScaleSaturated(s64(s)) }, scaleSaturatingUnsignedSlice[uint64, int64])
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

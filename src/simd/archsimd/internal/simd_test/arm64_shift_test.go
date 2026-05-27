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

var testShiftConstAmt uint64 = 3

func TestShiftLeftConst(t *testing.T) {
	// Signed
	testInt8x16ShiftConst(t, archsimd.Int8x16.ShiftLeftConst, shiftLeftByConstSlice[int8])
	testInt16x8ShiftConst(t, archsimd.Int16x8.ShiftLeftConst, shiftLeftByConstSlice[int16])
	testInt32x4ShiftConst(t, archsimd.Int32x4.ShiftLeftConst, shiftLeftByConstSlice[int32])
	testInt64x2ShiftConst(t, archsimd.Int64x2.ShiftLeftConst, shiftLeftByConstSlice[int64])
	// Unsigned
	testUint8x16ShiftConst(t, archsimd.Uint8x16.ShiftLeftConst, shiftLeftByConstSlice[uint8])
	testUint16x8ShiftConst(t, archsimd.Uint16x8.ShiftLeftConst, shiftLeftByConstSlice[uint16])
	testUint32x4ShiftConst(t, archsimd.Uint32x4.ShiftLeftConst, shiftLeftByConstSlice[uint32])
	testUint64x2ShiftConst(t, archsimd.Uint64x2.ShiftLeftConst, shiftLeftByConstSlice[uint64])

	// Variable shift amount to prevent constant folding
	forSlice(t, int32s, 4, func(x []int32) bool {
		a := archsimd.LoadInt32x4(x)
		g := make([]int32, 4)
		a.ShiftLeftConst(testShiftConstAmt).Store(g)
		w := shiftLeftByConstSlice(x, testShiftConstAmt)
		return checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, amt=%d", x, testShiftConstAmt) })
	})
}

func TestShiftRightConst(t *testing.T) {
	// Signed (arithmetic right shift)
	testInt8x16ShiftConst(t, archsimd.Int8x16.ShiftRightConst, shiftRightByConstSlice[int8])
	testInt16x8ShiftConst(t, archsimd.Int16x8.ShiftRightConst, shiftRightByConstSlice[int16])
	testInt32x4ShiftConst(t, archsimd.Int32x4.ShiftRightConst, shiftRightByConstSlice[int32])
	testInt64x2ShiftConst(t, archsimd.Int64x2.ShiftRightConst, shiftRightByConstSlice[int64])
	// Unsigned (logical right shift)
	testUint8x16ShiftConst(t, archsimd.Uint8x16.ShiftRightConst, shiftRightByConstSlice[uint8])
	testUint16x8ShiftConst(t, archsimd.Uint16x8.ShiftRightConst, shiftRightByConstSlice[uint16])
	testUint32x4ShiftConst(t, archsimd.Uint32x4.ShiftRightConst, shiftRightByConstSlice[uint32])
	testUint64x2ShiftConst(t, archsimd.Uint64x2.ShiftRightConst, shiftRightByConstSlice[uint64])

	// Variable shift amount to prevent constant folding
	forSlice(t, int32s, 4, func(x []int32) bool {
		a := archsimd.LoadInt32x4(x)
		g := make([]int32, 4)
		a.ShiftRightConst(testShiftConstAmt).Store(g)
		w := shiftRightByConstSlice(x, testShiftConstAmt)
		return checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, amt=%d", x, testShiftConstAmt) })
	})
}

func TestShiftLeftSaturatedConst(t *testing.T) {
	// Signed (saturating to signed range)
	testInt8x16ShiftConst(t, archsimd.Int8x16.ShiftLeftSaturatedConst, shiftLeftSaturatingByConstSlice[int8])
	testInt16x8ShiftConst(t, archsimd.Int16x8.ShiftLeftSaturatedConst, shiftLeftSaturatingByConstSlice[int16])
	testInt32x4ShiftConst(t, archsimd.Int32x4.ShiftLeftSaturatedConst, shiftLeftSaturatingByConstSlice[int32])
	testInt64x2ShiftConst(t, archsimd.Int64x2.ShiftLeftSaturatedConst, shiftLeftSaturatingByConstSlice[int64])
	// Unsigned (saturating to unsigned range)
	testUint8x16ShiftConst(t, archsimd.Uint8x16.ShiftLeftSaturatedConst, shiftLeftSaturatingUByConstSlice[uint8])
	testUint16x8ShiftConst(t, archsimd.Uint16x8.ShiftLeftSaturatedConst, shiftLeftSaturatingUByConstSlice[uint16])
	testUint32x4ShiftConst(t, archsimd.Uint32x4.ShiftLeftSaturatedConst, shiftLeftSaturatingUByConstSlice[uint32])
	testUint64x2ShiftConst(t, archsimd.Uint64x2.ShiftLeftSaturatedConst, shiftLeftSaturatingUByConstSlice[uint64])

	// Variable shift amount to prevent constant folding
	forSlice(t, int32s, 4, func(x []int32) bool {
		a := archsimd.LoadInt32x4(x)
		g := make([]int32, 4)
		a.ShiftLeftSaturatedConst(testShiftConstAmt).Store(g)
		w := shiftLeftSaturatingByConstSlice(x, testShiftConstAmt)
		return checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, amt=%d", x, testShiftConstAmt) })
	})
}

// testShiftAllAmts contains shift amounts for ShiftAll tests, including
// in-range amounts for all element sizes and out-of-range amounts to
// verify CSEL/CMPconst clamping logic in the lowering rules.
var testShiftAllAmts = []uint64{0, 1, 3, 7, 15, 31, 63, 128, 1024}

// testShiftAllVarAmt is a non-constant shift amount to prevent constant folding.
var testShiftAllVarAmt uint64 = 3

func TestShiftAllLeft(t *testing.T) {
	// Signed
	testInt8x16ShiftAll(t, archsimd.Int8x16.ShiftAllLeft, shiftAllLeftSlice[int8])
	testInt16x8ShiftAll(t, archsimd.Int16x8.ShiftAllLeft, shiftAllLeftSlice[int16])
	testInt32x4ShiftAll(t, archsimd.Int32x4.ShiftAllLeft, shiftAllLeftSlice[int32])
	testInt64x2ShiftAll(t, archsimd.Int64x2.ShiftAllLeft, shiftAllLeftSlice[int64])
	// Unsigned
	testUint8x16ShiftAll(t, archsimd.Uint8x16.ShiftAllLeft, shiftAllLeftSlice[uint8])
	testUint16x8ShiftAll(t, archsimd.Uint16x8.ShiftAllLeft, shiftAllLeftSlice[uint16])
	testUint32x4ShiftAll(t, archsimd.Uint32x4.ShiftAllLeft, shiftAllLeftSlice[uint32])
	testUint64x2ShiftAll(t, archsimd.Uint64x2.ShiftAllLeft, shiftAllLeftSlice[uint64])

	// Variable shift amount to prevent constant folding
	forSlice(t, int32s, 4, func(x []int32) bool {
		a := archsimd.LoadInt32x4(x)
		g := make([]int32, 4)
		a.ShiftAllLeft(testShiftAllVarAmt).Store(g)
		w := shiftAllLeftSlice(x, testShiftAllVarAmt)
		return checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, amt=%d", x, testShiftAllVarAmt) })
	})
}

func TestShiftAllRight(t *testing.T) {
	// Signed (arithmetic right shift)
	testInt8x16ShiftAll(t, archsimd.Int8x16.ShiftAllRight, shiftAllRightSlice[int8])
	testInt16x8ShiftAll(t, archsimd.Int16x8.ShiftAllRight, shiftAllRightSlice[int16])
	testInt32x4ShiftAll(t, archsimd.Int32x4.ShiftAllRight, shiftAllRightSlice[int32])
	testInt64x2ShiftAll(t, archsimd.Int64x2.ShiftAllRight, shiftAllRightSlice[int64])
	// Unsigned (logical right shift)
	testUint8x16ShiftAll(t, archsimd.Uint8x16.ShiftAllRight, shiftAllRightSlice[uint8])
	testUint16x8ShiftAll(t, archsimd.Uint16x8.ShiftAllRight, shiftAllRightSlice[uint16])
	testUint32x4ShiftAll(t, archsimd.Uint32x4.ShiftAllRight, shiftAllRightSlice[uint32])
	testUint64x2ShiftAll(t, archsimd.Uint64x2.ShiftAllRight, shiftAllRightSlice[uint64])

	// Variable shift amount to prevent constant folding
	forSlice(t, int32s, 4, func(x []int32) bool {
		a := archsimd.LoadInt32x4(x)
		g := make([]int32, 4)
		a.ShiftAllRight(testShiftAllVarAmt).Store(g)
		w := shiftAllRightSlice(x, testShiftAllVarAmt)
		return checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, amt=%d", x, testShiftAllVarAmt) })
	})
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

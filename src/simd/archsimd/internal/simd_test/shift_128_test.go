// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || wasm || arm64)

package simd_test

import (
	"runtime"
	"simd/archsimd"
	"testing"
)

func TestRotateAllLeft(t *testing.T) {
	x := uint8(0x81)
	if y := rotl(x, 1); y != 3 {
		t.Errorf("Expected 3, got 0x%x", y)
	}
	if y := rotl(x, 7); y != 0xc0 {
		t.Errorf("Expected 0xc0, got 0x%x", y)
	}
	if y := rotr(x, 4); y != 0x18 {
		t.Errorf("Expected 0x18, got 0x%x", y)
	}

	for i := uint64(0); i < 65; i++ {
		testUint64x2Unary(t, curry2(archsimd.Uint64x2.RotateAllLeft, i), rotlOfSlice[uint64](i))
		testUint32x4Unary(t, curry2(archsimd.Uint32x4.RotateAllLeft, i), rotlOfSlice[uint32](i))
		//		testUint16x8Unary(t, curry2(archsimd.Uint16x8.RotateAllLeft, i), rotlOfSlice[uint16](i))
		//		testUint8x16Unary(t, curry2(archsimd.Uint8x16.RotateAllLeft, i), rotlOfSlice[uint8](i))
	}
}

func TestRotateAllRight(t *testing.T) {
	x := uint8(0x81)
	if y := rotr(x, 1); y != 0xc0 {
		t.Errorf("Expected 0xc0, got 0x%x", y)
	}
	if y := rotr(x, 7); y != 3 {
		t.Errorf("Expected 3, got 0x%x", y)
	}
	if y := rotr(x, 4); y != 0x18 {
		t.Errorf("Expected 0x18, got 0x%x", y)
	}

	for i := uint64(0); i < 65; i++ {
		testUint64x2Unary(t, curry2(archsimd.Uint64x2.RotateAllRight, i), rotrOfSlice[uint64](i))
		testUint32x4Unary(t, curry2(archsimd.Uint32x4.RotateAllRight, i), rotrOfSlice[uint32](i))
		//		testUint16x8Unary(t, curry2(archsimd.Uint16x8.RotateAllLeft, i), rotlOfSlice[uint16](i))
		//		testUint8x16Unary(t, curry2(archsimd.Uint8x16.RotateAllLeft, i), rotlOfSlice[uint8](i))
	}
}

func TestShiftAll(t *testing.T) {
	// Test both const and non-const shifts.
	// Test both regular and over-shifts.

	hide := hideConst[uint64]

	// ShiftAllLeft

	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeft(2) },
		map1(func(x int32) int32 { return x << 2 }))
	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeft(hide(2)) },
		map1(func(x int32) int32 { return x << hide(2) }))

	// Ironically, we have to hide the constant in the want function so the
	// compiler doesn't complain about a silly shift.
	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeft(0x1000) },
		map1(func(x int32) int32 { return x << hide(0x1000) }))
	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllLeft(hide(0x1000)) },
		map1(func(x int32) int32 { return x << hide(0x1000) }))

	testInt16x8ShiftAll(t, archsimd.Int16x8.ShiftAllLeft, shiftAllLeftSlice[int16])
	testInt32x4ShiftAll(t, archsimd.Int32x4.ShiftAllLeft, shiftAllLeftSlice[int32])
	testInt64x2ShiftAll(t, archsimd.Int64x2.ShiftAllLeft, shiftAllLeftSlice[int64])
	testUint16x8ShiftAll(t, archsimd.Uint16x8.ShiftAllLeft, shiftAllLeftSlice[uint16])
	testUint32x4ShiftAll(t, archsimd.Uint32x4.ShiftAllLeft, shiftAllLeftSlice[uint32])
	testUint64x2ShiftAll(t, archsimd.Uint64x2.ShiftAllLeft, shiftAllLeftSlice[uint64])

	// Signed ShiftAllRight

	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRight(2) },
		map1(func(x int32) int32 { return x >> 2 }))
	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRight(hide(2)) },
		map1(func(x int32) int32 { return x >> hide(2) }))

	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRight(0x1000) },
		map1(func(x int32) int32 { return x >> hide(0x1000) }))
	testInt32x4Unary(t,
		func(x archsimd.Int32x4) archsimd.Int32x4 { return x.ShiftAllRight(hide(0x1000)) },
		map1(func(x int32) int32 { return x >> hide(0x1000) }))

	testInt16x8ShiftAll(t, archsimd.Int16x8.ShiftAllRight, shiftAllRightSlice[int16])
	testInt32x4ShiftAll(t, archsimd.Int32x4.ShiftAllRight, shiftAllRightSlice[int32])
	if runtime.GOARCH != "amd64" || archsimd.X86.AVX512() {
		testInt64x2ShiftAll(t, archsimd.Int64x2.ShiftAllRight, shiftAllRightSlice[int64])
	}

	// Unsigned ShiftAllRight

	testUint32x4Unary(t,
		func(x archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRight(2) },
		map1(func(x uint32) uint32 { return x >> 2 }))
	testUint32x4Unary(t,
		func(x archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRight(hide(2)) },
		map1(func(x uint32) uint32 { return x >> hide(2) }))

	testUint32x4Unary(t,
		func(x archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRight(0x1000) },
		map1(func(x uint32) uint32 { return x >> hide(0x1000) }))
	testUint32x4Unary(t,
		func(x archsimd.Uint32x4) archsimd.Uint32x4 { return x.ShiftAllRight(hide(0x1000)) },
		map1(func(x uint32) uint32 { return x >> hide(0x1000) }))

	testUint16x8ShiftAll(t, archsimd.Uint16x8.ShiftAllRight, shiftAllRightSlice[uint16])
	testUint32x4ShiftAll(t, archsimd.Uint32x4.ShiftAllRight, shiftAllRightSlice[uint32])
	testUint64x2ShiftAll(t, archsimd.Uint64x2.ShiftAllRight, shiftAllRightSlice[uint64])
}

var testShiftAllAmts = []uint64{0, 1, 3, 7, 15, 31, 63, 128, 1024}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64)

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// TestShiftRightNarrowConst checks the values of shift-right-by-constant followed by truncation.
func TestShiftRightNarrowConst(t *testing.T) {
	testUint16x8ConvertToUint8(t, func(x archsimd.Uint16x8) archsimd.Uint8x16 { return x.ShiftAllRight(0).TruncToUint8() },
		map1(func(x uint16) uint8 { return uint8(x) }))
	testUint16x8ConvertToUint8(t, func(x archsimd.Uint16x8) archsimd.Uint8x16 { return x.ShiftAllRight(1).TruncToUint8() },
		map1(func(x uint16) uint8 { return uint8(x >> 1) }))
	testUint16x8ConvertToUint8(t, func(x archsimd.Uint16x8) archsimd.Uint8x16 { return x.ShiftAllRight(4).TruncToUint8() },
		map1(func(x uint16) uint8 { return uint8(x >> 4) }))
	testUint16x8ConvertToUint8(t, func(x archsimd.Uint16x8) archsimd.Uint8x16 { return x.ShiftAllRight(8).TruncToUint8() }, // max fusable
		map1(func(x uint16) uint8 { return uint8(x >> 8) }))
	testUint16x8ConvertToUint8(t, func(x archsimd.Uint16x8) archsimd.Uint8x16 { return x.ShiftAllRight(9).TruncToUint8() }, // imm only, no fusion
		map1(func(x uint16) uint8 { return uint8(x >> 9) }))
	testUint16x8ConvertToUint8(t, func(x archsimd.Uint16x8) archsimd.Uint8x16 { return x.ShiftAllRight(15).TruncToUint8() },
		map1(func(x uint16) uint8 { return uint8(x >> 15) }))
	testUint16x8ConvertToUint8(t, func(x archsimd.Uint16x8) archsimd.Uint8x16 { return x.ShiftAllRight(16).TruncToUint8() }, // == width
		map1(func(x uint16) uint8 { return uint8(x >> hideConst(uint(16))) }))
	testUint16x8ConvertToUint8(t, func(x archsimd.Uint16x8) archsimd.Uint8x16 { return x.ShiftAllRight(17).TruncToUint8() }, // > width
		map1(func(x uint16) uint8 { return uint8(x >> hideConst(uint(17))) }))

	testInt16x8ConvertToInt8(t, func(x archsimd.Int16x8) archsimd.Int8x16 { return x.ShiftAllRight(4).TruncToInt8() },
		map1(func(x int16) int8 { return int8(x >> 4) }))
	testInt16x8ConvertToInt8(t, func(x archsimd.Int16x8) archsimd.Int8x16 { return x.ShiftAllRight(8).TruncToInt8() },
		map1(func(x int16) int8 { return int8(x >> 8) }))
	testInt16x8ConvertToInt8(t, func(x archsimd.Int16x8) archsimd.Int8x16 { return x.ShiftAllRight(0).TruncToInt8() },
		map1(func(x int16) int8 { return int8(x) }))
	testInt16x8ConvertToInt8(t, func(x archsimd.Int16x8) archsimd.Int8x16 { return x.ShiftAllRight(16).TruncToInt8() },
		map1(func(x int16) int8 { return int8(x >> hideConst(uint(16))) }))

	testUint32x4ConvertToUint16(t, func(x archsimd.Uint32x4) archsimd.Uint16x8 { return x.ShiftAllRight(8).TruncToUint16() },
		map1(func(x uint32) uint16 { return uint16(x >> 8) }))
	testUint32x4ConvertToUint16(t, func(x archsimd.Uint32x4) archsimd.Uint16x8 { return x.ShiftAllRight(16).TruncToUint16() },
		map1(func(x uint32) uint16 { return uint16(x >> 16) }))
	testUint32x4ConvertToUint16(t, func(x archsimd.Uint32x4) archsimd.Uint16x8 { return x.ShiftAllRight(0).TruncToUint16() },
		map1(func(x uint32) uint16 { return uint16(x) }))
	testUint32x4ConvertToUint16(t, func(x archsimd.Uint32x4) archsimd.Uint16x8 { return x.ShiftAllRight(32).TruncToUint16() },
		map1(func(x uint32) uint16 { return uint16(x >> hideConst(uint(32))) }))

	testInt32x4ConvertToInt16(t, func(x archsimd.Int32x4) archsimd.Int16x8 { return x.ShiftAllRight(8).TruncToInt16() },
		map1(func(x int32) int16 { return int16(x >> 8) }))
	testInt32x4ConvertToInt16(t, func(x archsimd.Int32x4) archsimd.Int16x8 { return x.ShiftAllRight(16).TruncToInt16() },
		map1(func(x int32) int16 { return int16(x >> 16) }))

	testUint64x2ConvertToUint32(t, func(x archsimd.Uint64x2) archsimd.Uint32x4 { return x.ShiftAllRight(32).TruncToUint32() },
		map1(func(x uint64) uint32 { return uint32(x >> 32) }))
	testUint64x2ConvertToUint32(t, func(x archsimd.Uint64x2) archsimd.Uint32x4 { return x.ShiftAllRight(0).TruncToUint32() },
		map1(func(x uint64) uint32 { return uint32(x) }))
	testUint64x2ConvertToUint32(t, func(x archsimd.Uint64x2) archsimd.Uint32x4 { return x.ShiftAllRight(64).TruncToUint32() },
		map1(func(x uint64) uint32 { return uint32(x >> hideConst(uint(64))) }))

	testInt64x2ConvertToInt32(t, func(x archsimd.Int64x2) archsimd.Int32x4 { return x.ShiftAllRight(32).TruncToInt32() },
		map1(func(x int64) int32 { return int32(x >> 32) }))
}

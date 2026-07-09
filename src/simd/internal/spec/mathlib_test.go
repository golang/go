// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

import (
	"math"
	"testing"
)

func assertEq[T comparable](t *testing.T, got, want T, msg string) {
	t.Helper()
	if got != want {
		t.Errorf("%s: got %v, want %v", msg, got, want)
	}
}

func TestIsSigned(t *testing.T) {
	assertEq(t, isSigned[int8](), true, "isSigned[int8]()")
	assertEq(t, isSigned[uint8](), false, "isSigned[uint8]()")
	assertEq(t, isSigned[int16](), true, "isSigned[int16]()")
	assertEq(t, isSigned[uint16](), false, "isSigned[uint16]()")
	assertEq(t, isSigned[int32](), true, "isSigned[int32]()")
	assertEq(t, isSigned[uint32](), false, "isSigned[uint32]()")
	assertEq(t, isSigned[int64](), true, "isSigned[int64]()")
	assertEq(t, isSigned[uint64](), false, "isSigned[uint64]()")
}

func TestMaxVal(t *testing.T) {
	assertEq(t, maxVal[int8](), int8(math.MaxInt8), "maxVal[int8]")
	assertEq(t, maxVal[uint8](), uint8(math.MaxUint8), "maxVal[uint8]")
	assertEq(t, maxVal[int16](), int16(math.MaxInt16), "maxVal[int16]")
	assertEq(t, maxVal[uint16](), uint16(math.MaxUint16), "maxVal[uint16]")
	assertEq(t, maxVal[int32](), int32(math.MaxInt32), "maxVal[int32]")
	assertEq(t, maxVal[uint32](), uint32(math.MaxUint32), "maxVal[uint32]")
	assertEq(t, maxVal[int64](), int64(math.MaxInt64), "maxVal[int64]")
	assertEq(t, maxVal[uint64](), uint64(math.MaxUint64), "maxVal[uint64]")
}

func TestMinVal(t *testing.T) {
	assertEq(t, minVal[int8](), int8(math.MinInt8), "minVal[int8]")
	assertEq(t, minVal[uint8](), uint8(0), "minVal[uint8]")
	assertEq(t, minVal[int16](), int16(math.MinInt16), "minVal[int16]")
	assertEq(t, minVal[uint16](), uint16(0), "minVal[uint16]")
	assertEq(t, minVal[int32](), int32(math.MinInt32), "minVal[int32]")
	assertEq(t, minVal[uint32](), uint32(0), "minVal[uint32]")
	assertEq(t, minVal[int64](), int64(math.MinInt64), "minVal[int64]")
	assertEq(t, minVal[uint64](), uint64(0), "minVal[uint64]")
}

func TestSaturate(t *testing.T) {
	// saturate[T Ints | Uints, U Ints | Uints](x T) U

	// Signed to signed: int16 -> int8
	assertEq(t, saturate[int16, int8](126), int8(126), "saturate[int16, int8](126)")
	assertEq(t, saturate[int16, int8](-127), int8(-127), "saturate[int16, int8](-127)")
	assertEq(t, saturate[int16, int8](128), int8(math.MaxInt8), "saturate[int16, int8](128) overflow")
	assertEq(t, saturate[int16, int8](-129), int8(math.MinInt8), "saturate[int16, int8](-129) underflow")

	// Unsigned to unsigned: uint16 -> uint8
	assertEq(t, saturate[uint16, uint8](254), uint8(254), "saturate[uint16, uint8](254)")
	assertEq(t, saturate[uint16, uint8](256), uint8(math.MaxUint8), "saturate[uint16, uint8](256) overflow")

	// Signed to unsigned: int16 -> uint8
	assertEq(t, saturate[int16, uint8](1), uint8(1), "saturate[int16, uint8](1)")
	assertEq(t, saturate[int16, uint8](-1), uint8(0), "saturate[int16, uint8](-1) underflow")
	assertEq(t, saturate[int16, uint8](254), uint8(254), "saturate[int16, uint8](254)")
	assertEq(t, saturate[int16, uint8](256), uint8(math.MaxUint8), "saturate[int16, uint8](256) overflow")

	// Unsigned to signed: uint16 -> int8
	assertEq(t, saturate[uint16, int8](126), int8(126), "saturate[uint16, int8](126)")
	assertEq(t, saturate[uint16, int8](128), int8(math.MaxInt8), "saturate[uint16, int8](128) overflow")

	// Wider bounds: int64 to int32
	assertEq(t, saturate[int64, int32](math.MaxInt32-1), int32(math.MaxInt32-1), "saturate[int64, int32](MaxInt32-1)")
	assertEq(t, saturate[int64, int32](math.MaxInt32+1), int32(math.MaxInt32), "saturate[int64, int32](MaxInt32+1) overflow")
	assertEq(t, saturate[int64, int32](math.MinInt32+1), int32(math.MinInt32+1), "saturate[int64, int32](MinInt32+1)")
	assertEq(t, saturate[int64, int32](math.MinInt32-1), int32(math.MinInt32), "saturate[int64, int32](MinInt32-1) underflow")
}

func TestAddSaturated(t *testing.T) {
	// Signed int8
	assertEq(t, addSaturated[int8](125, 1), int8(126), "addSaturated[int8](125, 1)")
	assertEq(t, addSaturated[int8](120, 10), int8(math.MaxInt8), "addSaturated[int8](120, 10) overflow")
	assertEq(t, addSaturated[int8](-126, -1), int8(-127), "addSaturated[int8](-126, -1)")
	assertEq(t, addSaturated[int8](-120, -10), int8(math.MinInt8), "addSaturated[int8](-120, -10) underflow")

	// Unsigned uint8
	assertEq(t, addSaturated[uint8](253, 1), uint8(254), "addSaturated[uint8](253, 1)")
	assertEq(t, addSaturated[uint8](250, 10), uint8(math.MaxUint8), "addSaturated[uint8](250, 10) overflow")
	assertEq(t, addSaturated[uint8](254, 0), uint8(254), "addSaturated[uint8](254, 0)")

	// Signed int64
	assertEq(t, addSaturated[int64](math.MaxInt64-2, 1), int64(math.MaxInt64-1), "addSaturated[int64](MaxInt64-2, 1)")
	assertEq(t, addSaturated[int64](math.MaxInt64-5, 10), int64(math.MaxInt64), "addSaturated[int64](max-5, 10) overflow")
	assertEq(t, addSaturated[int64](math.MinInt64+2, -1), int64(math.MinInt64+1), "addSaturated[int64](MinInt64+2, -1)")
	assertEq(t, addSaturated[int64](math.MinInt64+5, -10), int64(math.MinInt64), "addSaturated[int64](min+5, -10) underflow")

	// Unsigned uint64
	assertEq(t, addSaturated[uint64](math.MaxUint64-2, 1), uint64(math.MaxUint64-1), "addSaturated[uint64](MaxUint64-2, 1)")
	assertEq(t, addSaturated[uint64](math.MaxUint64-5, 10), uint64(math.MaxUint64), "addSaturated[uint64](max-5, 10) overflow")
}

func TestMulSaturatedUSS(t *testing.T) {
	// mulSaturatedUSS[X Uints, Y Ints](x X, y Y) Y

	// uint8, int8
	assertEq(t, mulSaturatedUSS[uint8, int8](0, 10), int8(0), "mulSaturatedUSS[uint8, int8](0, 10)")
	assertEq(t, mulSaturatedUSS[uint8, int8](10, 0), int8(0), "mulSaturatedUSS[uint8, int8](10, 0)")
	assertEq(t, mulSaturatedUSS[uint8, int8](2, 63), int8(126), "mulSaturatedUSS[uint8, int8](2, 63)")
	assertEq(t, mulSaturatedUSS[uint8, int8](1, 126), int8(126), "mulSaturatedUSS[uint8, int8](1, 126)")
	assertEq(t, mulSaturatedUSS[uint8, int8](1, -127), int8(-127), "mulSaturatedUSS[uint8, int8](1, -127)")
	assertEq(t, mulSaturatedUSS[uint8, int8](10, 20), int8(math.MaxInt8), "mulSaturatedUSS[uint8, int8](10, 20) positive overflow")
	assertEq(t, mulSaturatedUSS[uint8, int8](10, -20), int8(math.MinInt8), "mulSaturatedUSS[uint8, int8](10, -20) negative overflow")

	// uint64, int64
	assertEq(t, mulSaturatedUSS[uint64, int64](2, (math.MaxInt64-1)/2), int64(math.MaxInt64-1), "mulSaturatedUSS[uint64, int64](2, (MaxInt64-1)/2)")
	assertEq(t, mulSaturatedUSS[uint64, int64](2, math.MaxInt64), int64(math.MaxInt64), "mulSaturatedUSS[uint64, int64](2, MaxInt64) overflow")
	assertEq(t, mulSaturatedUSS[uint64, int64](2, (math.MinInt64/2)+1), int64(math.MinInt64+2), "mulSaturatedUSS[uint64, int64](2, (MinInt64/2)+1)")
	assertEq(t, mulSaturatedUSS[uint64, int64](2, math.MinInt64), int64(math.MinInt64), "mulSaturatedUSS[uint64, int64](2, MinInt64) underflow")
	assertEq(t, mulSaturatedUSS[uint64, int64](math.MaxUint64, 1), int64(math.MaxInt64), "mulSaturatedUSS[uint64, int64](MaxUint64, 1) overflow")
	assertEq(t, mulSaturatedUSS[uint64, int64](math.MaxUint64, -1), int64(math.MinInt64), "mulSaturatedUSS[uint64, int64](MaxUint64, -1) underflow")
}

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

type assertFn1Tab[X, Want any] struct {
	x    X
	want Want
}

func assertFn1[X, Z comparable](t *testing.T, fn func(X) Z, label string, tab ...assertFn1Tab[X, Z]) {
	t.Helper()
	for _, entry := range tab {
		got := fn(entry.x)
		if got != entry.want {
			t.Errorf("%s(%v) = %v; want %v", label, entry.x, got, entry.want)
		}
	}
}

type assertFn2Tab[X, Y, Want any] struct {
	x    X
	y    Y
	want Want
}

func assertFn2[X, Y, Z comparable](t *testing.T, fn func(X, Y) Z, label string, tab ...assertFn2Tab[X, Y, Z]) {
	t.Helper()
	for _, entry := range tab {
		got := fn(entry.x, entry.y)
		if got != entry.want {
			t.Errorf("%s(%v, %v) = %v; want %v", label, entry.x, entry.y, got, entry.want)
		}
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
	assertFn1[int16, int8](t, saturate, "saturate[int16, int8]",
		{126, 126}, {-127, -127}, {128, math.MaxInt8}, {-129, math.MinInt8})

	// Unsigned to unsigned: uint16 -> uint8
	assertFn1[uint16, uint8](t, saturate, "saturate[uint16, uint8]",
		{254, 254}, {256, math.MaxUint8})

	// Signed to unsigned: int16 -> uint8
	assertFn1[int16, uint8](t, saturate, "saturate[int16, uint8]",
		{1, 1}, {-1, 0}, {254, 254}, {256, math.MaxUint8})

	// Unsigned to signed: uint16 -> int8
	assertFn1[uint16, int8](t, saturate, "saturate[uint16, int8]",
		{126, 126}, {128, math.MaxInt8})

	// Wider bounds: int64 to int32
	assertFn1[int64, int32](t, saturate, "saturate[int64, int32]",
		{math.MaxInt32 - 1, math.MaxInt32 - 1},
		{math.MaxInt32 + 1, math.MaxInt32},
		{math.MinInt32 + 1, math.MinInt32 + 1},
		{math.MinInt32 - 1, math.MinInt32})
}

func TestAddSaturated(t *testing.T) {
	// Signed int8
	assertFn2[int8, int8, int8](t, addSaturated, "addSaturated[int8]",
		{125, 1, 126}, {120, 10, math.MaxInt8}, {-126, -1, -127}, {-120, -10, math.MinInt8})

	// Unsigned uint8
	assertFn2[uint8, uint8, uint8](t, addSaturated, "addSaturated[uint8]",
		{253, 1, 254}, {250, 10, math.MaxUint8}, {254, 0, 254})

	// Signed int64
	assertFn2[int64, int64, int64](t, addSaturated, "addSaturated[int64]",
		{math.MaxInt64 - 2, 1, math.MaxInt64 - 1},
		{math.MaxInt64 - 5, 10, math.MaxInt64},
		{math.MinInt64 + 2, -1, math.MinInt64 + 1},
		{math.MinInt64 + 5, -10, math.MinInt64})

	// Unsigned uint64
	assertFn2[uint64, uint64, uint64](t, addSaturated, "addSaturated[uint64]",
		{math.MaxUint64 - 2, 1, math.MaxUint64 - 1},
		{math.MaxUint64 - 5, 10, math.MaxUint64})
}

func TestMulSaturatedUSS(t *testing.T) {
	// mulSaturatedUSS[X Uints, Y Ints](x X, y Y) Y

	// uint8, int8
	assertFn2[uint8, int8, int8](t, mulSaturatedUSS, "mulSaturatedUSS[uint8, int8]",
		{0, 10, 0}, {10, 0, 0}, {2, 63, 126}, {1, 126, 126}, {1, -127, -127},
		{10, 20, math.MaxInt8}, {10, -20, math.MinInt8})

	// uint64, int64
	assertFn2[uint64, int64, int64](t, mulSaturatedUSS, "mulSaturatedUSS[uint64, int64]",
		{2, (math.MaxInt64 - 1) / 2, math.MaxInt64 - 1},
		{2, math.MaxInt64, math.MaxInt64},
		{2, (math.MinInt64 / 2) + 1, math.MinInt64 + 2},
		{2, math.MinInt64, math.MinInt64},
		{math.MaxUint64, 1, math.MaxInt64},
		{math.MaxUint64, -1, math.MinInt64})
}

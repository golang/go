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

func TestSubSaturated(t *testing.T) {
	// Signed int8
	assertFn2[int8, int8, int8](t, subSaturated, "subSaturated[int8]",
		{125, 1, 124}, {120, -10, math.MaxInt8}, {-126, 2, -128}, {-120, 10, math.MinInt8})

	// Unsigned uint8
	assertFn2[uint8, uint8, uint8](t, subSaturated, "subSaturated[uint8]",
		{253, 1, 252}, {5, 10, 0}, {254, 0, 254})

	// Signed int64
	assertFn2[int64, int64, int64](t, subSaturated, "subSaturated[int64]",
		{math.MaxInt64 - 2, 1, math.MaxInt64 - 3},
		{math.MaxInt64 - 5, -10, math.MaxInt64},
		{math.MinInt64 + 2, 1, math.MinInt64 + 1},
		{math.MinInt64 + 5, 10, math.MinInt64},
		{0, math.MinInt64, math.MaxInt64})

	// Unsigned uint64
	assertFn2[uint64, uint64, uint64](t, subSaturated, "subSaturated[uint64]",
		{math.MaxUint64 - 2, 1, math.MaxUint64 - 3},
		{5, 10, 0})
}

func TestScaleSaturated(t *testing.T) {
	// Signed int8
	assertFn2[int8, int, int8](t, scaleSaturated, "scaleSaturated[int8]",
		{10, 0, 10}, {10, 1, 20}, {10, -1, 5}, {-10, -1, -5}, {-60, 1, -120},
		{0, 10, 0}, {10, -8, 0}, {-10, -8, -1}, {10, -100, 0}, {-10, -100, -1},
		// Overflow
		{100, 1, math.MaxInt8}, {1, 10, math.MaxInt8},
		// Underflow
		{-70, 1, math.MinInt8}, {-1, 10, math.MinInt8})

	// Unsigned uint8
	assertFn2[uint8, int, uint8](t, scaleSaturated, "scaleSaturated[uint8]",
		{10, 0, 10}, {10, 1, 20}, {20, -1, 10}, {0, 10, 0},
		{10, -8, 0}, {10, -100, 0},
		// Overflow
		{200, 1, math.MaxUint8}, {1, 10, math.MaxUint8})

	// Signed int64
	assertFn2[int64, int, int64](t, scaleSaturated, "scaleSaturated[int64]",
		{1, 62, 1 << 62}, {-1, 63, math.MinInt64}, {0, 100, 0},
		{10, -64, 0}, {-10, -64, -1}, {10, -100, 0}, {-10, -100, -1},
		{10, math.MinInt, 0}, {-10, math.MinInt, -1},
		// Overflow
		{1, 63, math.MaxInt64}, {10, 100, math.MaxInt64},
		// Underflow
		{-2, 63, math.MinInt64})

	// Unsigned uint64
	assertFn2[uint64, int, uint64](t, scaleSaturated, "scaleSaturated[uint64]",
		{1, 63, 1 << 63}, {0, 100, 0},
		{10, -64, 0}, {10, -100, 0}, {10, math.MinInt, 0},
		// Overflow
		{2, 63, math.MaxUint64}, {5, 100, math.MaxUint64})
}

func TestClmul(t *testing.T) {
	// uint8 tests
	tab8 := []struct {
		a, b   uint8
		hi, lo uint8
	}{
		{0, 0, 0, 0},
		{0, 0xff, 0, 0},
		{1, 0xab, 0, 0xab},
		{0b1110, 0b1011, 0, 0b1100010}, // Example from bitmath.go doc
		{0x80, 0x80, 0x40, 0},
		{0xff, 0xff, 0x55, 0x55},
	}
	for _, tc := range tab8 {
		hi, lo := clmul(tc.a, tc.b)
		if hi != tc.hi || lo != tc.lo {
			t.Errorf("clmul[uint8](%#x, %#x) = (%#x, %#x); want (%#x, %#x)", tc.a, tc.b, hi, lo, tc.hi, tc.lo)
		}
		// Commutative
		rhi, rlo := clmul(tc.b, tc.a)
		if rhi != hi || rlo != lo {
			t.Errorf("clmul[uint8](%#x, %#x) != clmul(%#x, %#x)", tc.b, tc.a, tc.a, tc.b)
		}
	}

	// uint16 tests
	tab16 := []struct {
		a, b   uint16
		hi, lo uint16
	}{
		{0, 0, 0, 0},
		{1, 0xabcd, 0, 0xabcd},
		{0x8000, 0x8000, 0x4000, 0},
		{0xffff, 0xffff, 0x5555, 0x5555},
	}
	for _, tc := range tab16 {
		hi, lo := clmul(tc.a, tc.b)
		if hi != tc.hi || lo != tc.lo {
			t.Errorf("clmul[uint16](%#x, %#x) = (%#x, %#x); want (%#x, %#x)", tc.a, tc.b, hi, lo, tc.hi, tc.lo)
		}
		rhi, rlo := clmul(tc.b, tc.a)
		if rhi != hi || rlo != lo {
			t.Errorf("clmul[uint16](%#x, %#x) != clmul(%#x, %#x)", tc.b, tc.a, tc.a, tc.b)
		}
	}

	// uint32 tests
	tab32 := []struct {
		a, b   uint32
		hi, lo uint32
	}{
		{0, 0, 0, 0},
		{1, 0x12345678, 0, 0x12345678},
		{0x80000000, 0x80000000, 0x40000000, 0},
		{0xffffffff, 0xffffffff, 0x55555555, 0x55555555},
	}
	for _, tc := range tab32 {
		hi, lo := clmul(tc.a, tc.b)
		if hi != tc.hi || lo != tc.lo {
			t.Errorf("clmul[uint32](%#x, %#x) = (%#x, %#x); want (%#x, %#x)", tc.a, tc.b, hi, lo, tc.hi, tc.lo)
		}
		rhi, rlo := clmul(tc.b, tc.a)
		if rhi != hi || rlo != lo {
			t.Errorf("clmul[uint32](%#x, %#x) != clmul(%#x, %#x)", tc.b, tc.a, tc.a, tc.b)
		}
	}

	// uint64 tests
	tab64 := []struct {
		a, b   uint64
		hi, lo uint64
	}{
		{0, 0, 0, 0},
		{1, 0x123456789abcdef0, 0, 0x123456789abcdef0},
		{1, 3, 0, 3},
		{1, 9, 0, 9},
		{5, 3, 0, 15},
		{5, 9, 0, 45},
		{3, 3, 0, 5},
		{0x8000000000000000, 0x8000000000000000, 0x4000000000000000, 0},
		{0xffffffffffffffff, 0xffffffffffffffff, 0x5555555555555555, 0x5555555555555555},
	}
	for _, tc := range tab64 {
		hi, lo := clmul(tc.a, tc.b)
		if hi != tc.hi || lo != tc.lo {
			t.Errorf("clmul[uint64](%#x, %#x) = (%#x, %#x); want (%#x, %#x)", tc.a, tc.b, hi, lo, tc.hi, tc.lo)
		}
		rhi, rlo := clmul(tc.b, tc.a)
		if rhi != hi || rlo != lo {
			t.Errorf("clmul[uint64](%#x, %#x) != clmul(%#x, %#x)", tc.b, tc.a, tc.a, tc.b)
		}
	}
}

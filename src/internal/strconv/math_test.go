// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	. "internal/strconv"
	"math"
	"testing"
)

var pow10Tests = []struct {
	exp10 int
	mant  uint128
	exp2  int
	ok    bool
}{
	{-349, uint128{0, 0}, 0, false},
	{-348, uint128{0xFA8FD5A0081C0288, 0x1732C869CD60E453}, -1156, true},
	{0, uint128{0x8000000000000000, 0x0000000000000000}, 1, true},
	{347, uint128{0xD13EB46469447567, 0x4B7195F2D2D1A9FB}, 1153, true},
	{348, uint128{0, 0}, 0, false},
}

func TestPow10(t *testing.T) {
	for _, tt := range pow10Tests {
		mant, exp2, ok := pow10(tt.exp10)
		if mant != tt.mant || exp2 != tt.exp2 {
			t.Errorf("pow10(%d) = %#016x, %#016x, %d, %v want %#016x,%#016x, %d, %v",
				tt.exp10, mant.Hi, mant.Lo, exp2, ok,
				tt.mant.Hi, tt.mant.Lo, tt.exp2, tt.ok)
		}
	}

	for p := pow10Min; p <= pow10Max; p++ {
		mant, exp2, ok := pow10(p)
		if !ok {
			t.Errorf("pow10(%d) not ok", p)
			continue
		}
		// Note: -64 instead of -128 because we only used mant.Hi, not all of mant.
		have := math.Ldexp(float64(mant.Hi), exp2-64)
		want := math.Pow(10, float64(p))
		if math.Abs(have-want)/want > 0.00001 {
			t.Errorf("pow10(%d) = %#016x%016x/2^128 * 2^%d = %g want ~%g", p, mant.Hi, mant.Lo, exp2, have, want)
		}
	}

}

func u128(hi, lo uint64) uint128 {
	return uint128{Hi: hi, Lo: lo}
}

var umul192Tests = []struct {
	x   uint64
	y   uint128
	hi  uint64
	mid uint64
	lo  uint64
}{
	{0, u128(0, 0), 0, 0, 0},
	{^uint64(0), u128(^uint64(0), ^uint64(0)), ^uint64(1), ^uint64(0), 1},
}

func TestUmul192(t *testing.T) {
	for _, tt := range umul192Tests {
		hi, mid, lo := Umul192(tt.x, tt.y)
		if hi != tt.hi || mid != tt.mid || lo != tt.lo {
			t.Errorf("umul192(%#x, {%#x,%#x}) = %#x, %#x, %#x, want %#x, %#x, %#x",
				tt.x, tt.y.Hi, tt.y.Lo, hi, mid, lo, tt.hi, tt.mid, tt.lo)
		}
	}
}

func TestMulLog10_2(t *testing.T) {
	for x := -1600; x <= +1600; x++ {
		iMath := mulLog10_2(x)
		fMath := int(math.Floor(float64(x) * math.Ln2 / math.Ln10))
		if iMath != fMath {
			t.Errorf("mulLog10_2(%d) failed: %d vs %d\n", x, iMath, fMath)
		}
	}
}

func TestMulLog2_10(t *testing.T) {
	for x := -500; x <= +500; x++ {
		iMath := mulLog2_10(x)
		fMath := int(math.Floor(float64(x) * math.Ln10 / math.Ln2))
		if iMath != fMath {
			t.Errorf("mulLog2_10(%d) failed: %d vs %d\n", x, iMath, fMath)
		}
	}
}

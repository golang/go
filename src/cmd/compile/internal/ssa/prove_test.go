// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"math"
	"math/bits"
	"testing"
)

func testLimitUnaryOpSigned8(t *testing.T, opName string, initLimit Limit, op func(l Limit, bitsize uint) Limit, opImpl func(int8) int8) {
	for min := math.MinInt8; min <= math.MaxInt8; min++ {
		for max := min; max <= math.MaxInt8; max++ {
			realSmallest, realBiggest := int8(math.MaxInt8), int8(math.MinInt8)
			for i := min; i <= max; i++ {
				result := opImpl(int8(i))
				if result < realSmallest {
					realSmallest = result
				}
				if result > realBiggest {
					realBiggest = result
				}
			}

			l := Limit{int64(min), int64(max), 0, math.MaxUint64}
			l = op(l, 8)
			l = l.Intersect(initLimit) // We assume this is gonna be used by newLimit which is seeded by the op size already.

			if l.Min != int64(realSmallest) || l.Max != int64(realBiggest) {
				t.Errorf("%s(%d..%d) = %d..%d; want %d..%d", opName, min, max, l.Min, l.Max, realSmallest, realBiggest)
			}
		}
	}
}

func testLimitUnaryOpUnsigned8(t *testing.T, opName string, initLimit Limit, op func(l Limit, bitsize uint) Limit, opImpl func(uint8) uint8) {
	for min := 0; min <= math.MaxUint8; min++ {
		for max := min; max <= math.MaxUint8; max++ {
			realSmallest, realBiggest := uint8(math.MaxUint8), uint8(0)
			for i := min; i <= max; i++ {
				result := opImpl(uint8(i))
				if result < realSmallest {
					realSmallest = result
				}
				if result > realBiggest {
					realBiggest = result
				}
			}

			l := Limit{math.MinInt64, math.MaxInt64, uint64(min), uint64(max)}
			l = op(l, 8)
			l = l.Intersect(initLimit) // We assume this is gonna be used by newLimit which is seeded by the op size already.

			if l.Umin != uint64(realSmallest) || l.Umax != uint64(realBiggest) {
				t.Errorf("%s(%d..%d) = %d..%d; want %d..%d", opName, min, max, l.Umin, l.Umax, realSmallest, realBiggest)
			}
		}
	}
}

func TestLimitNegSigned(t *testing.T) {
	testLimitUnaryOpSigned8(t, "neg", NoLimitForBitsize(8), Limit.Neg, func(x int8) int8 { return -x })
}
func TestLimitNegUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "neg", NoLimitForBitsize(8), Limit.Neg, func(x uint8) uint8 { return -x })
}

func TestLimitComSigned(t *testing.T) {
	testLimitUnaryOpSigned8(t, "com", NoLimitForBitsize(8), Limit.Com, func(x int8) int8 { return ^x })
}
func TestLimitComUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "com", NoLimitForBitsize(8), Limit.Com, func(x uint8) uint8 { return ^x })
}

func TestLimitCtzUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "ctz", Limit{-128, 127, 0, 8}, Limit.Ctz, func(x uint8) uint8 { return uint8(bits.TrailingZeros8(x)) })
}

func TestLimitBitlenUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "bitlen", Limit{-128, 127, 0, 8}, Limit.Bitlen, func(x uint8) uint8 { return uint8(bits.Len8(x)) })
}

func TestLimitPopcountUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "popcount", Limit{-128, 127, 0, 8}, Limit.Popcount, func(x uint8) uint8 { return uint8(bits.OnesCount8(x)) })
}

func TestConvertIntWithBitsize(t *testing.T) {
	if got := ConvertIntWithBitsize[int64, uint64](255, 8); got != -1 {
		t.Errorf("convertIntWithBitsize(255, 8) = %d; want -1", got)
	}
	if got := ConvertIntWithBitsize[uint64, int64](-1, 8); got != 255 {
		t.Errorf("convertIntWithBitsize(-1, 8) = %d; want 255", got)
	}

	if got := ConvertIntWithBitsize[int64, uint64](127, 8); got != 127 {
		t.Errorf("convertIntWithBitsize(127, 8) = %d; want 127", got)
	}
	if got := ConvertIntWithBitsize[uint64, int64](127, 8); got != 127 {
		t.Errorf("convertIntWithBitsize(127, 8) = %d; want 127", got)
	}
}

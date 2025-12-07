// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"math"
	"math/bits"
	"testing"
)

func testLimitUnaryOpSigned8(t *testing.T, opName string, initLimit limit, op func(l limit, bitsize uint) limit, opImpl func(int8) int8) {
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

			l := limit{int64(min), int64(max), 0, math.MaxUint64}
			l = op(l, 8)
			l = l.intersect(initLimit) // We assume this is gonna be used by newLimit which is seeded by the op size already.

			if l.min != int64(realSmallest) || l.max != int64(realBiggest) {
				t.Errorf("%s(%d..%d) = %d..%d; want %d..%d", opName, min, max, l.min, l.max, realSmallest, realBiggest)
			}
		}
	}
}

func testLimitUnaryOpUnsigned8(t *testing.T, opName string, initLimit limit, op func(l limit, bitsize uint) limit, opImpl func(uint8) uint8) {
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

			l := limit{math.MinInt64, math.MaxInt64, uint64(min), uint64(max)}
			l = op(l, 8)
			l = l.intersect(initLimit) // We assume this is gonna be used by newLimit which is seeded by the op size already.

			if l.umin != uint64(realSmallest) || l.umax != uint64(realBiggest) {
				t.Errorf("%s(%d..%d) = %d..%d; want %d..%d", opName, min, max, l.umin, l.umax, realSmallest, realBiggest)
			}
		}
	}
}

func TestLimitNegSigned(t *testing.T) {
	testLimitUnaryOpSigned8(t, "neg", noLimitForBitsize(8), limit.neg, func(x int8) int8 { return -x })
}
func TestLimitNegUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "neg", noLimitForBitsize(8), limit.neg, func(x uint8) uint8 { return -x })
}

func TestLimitComSigned(t *testing.T) {
	testLimitUnaryOpSigned8(t, "com", noLimitForBitsize(8), limit.com, func(x int8) int8 { return ^x })
}
func TestLimitComUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "com", noLimitForBitsize(8), limit.com, func(x uint8) uint8 { return ^x })
}

func TestLimitCtzUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "ctz", limit{-128, 127, 0, 8}, limit.ctz, func(x uint8) uint8 { return uint8(bits.TrailingZeros8(x)) })
}

func TestLimitBitlenUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "bitlen", limit{-128, 127, 0, 8}, limit.bitlen, func(x uint8) uint8 { return uint8(bits.Len8(x)) })
}

func TestLimitPopcountUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "popcount", limit{-128, 127, 0, 8}, limit.popcount, func(x uint8) uint8 { return uint8(bits.OnesCount8(x)) })
}

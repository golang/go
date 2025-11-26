// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"math"
	"testing"
)

func testLimitUnaryOpSigned8(t *testing.T, opName string, op func(l limit, bitsize uint) limit, opImpl func(int8) int8) {
	sizeLimit := noLimitForBitsize(8)
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
			l = l.intersect(sizeLimit) // We assume this is gonna be used by newLimit which is seeded by the op size already.

			if l.min != int64(realSmallest) || l.max != int64(realBiggest) {
				t.Errorf("%s(%d..%d) = %d..%d; want %d..%d", opName, min, max, l.min, l.max, realSmallest, realBiggest)
			}
		}
	}
}

func testLimitUnaryOpUnsigned8(t *testing.T, opName string, op func(l limit, bitsize uint) limit, opImpl func(uint8) uint8) {
	sizeLimit := noLimitForBitsize(8)
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
			l = l.intersect(sizeLimit) // We assume this is gonna be used by newLimit which is seeded by the op size already.

			if l.umin != uint64(realSmallest) || l.umax != uint64(realBiggest) {
				t.Errorf("%s(%d..%d) = %d..%d; want %d..%d", opName, min, max, l.umin, l.umax, realSmallest, realBiggest)
			}
		}
	}
}

func TestLimitNegSigned(t *testing.T) {
	testLimitUnaryOpSigned8(t, "neg", limit.neg, func(x int8) int8 { return -x })
}
func TestLimitNegUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "neg", limit.neg, func(x uint8) uint8 { return -x })
}

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"fmt"
	"math"
	"math/bits"
	"testing"

	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/block"
)

func TestGetBranchJumpTable(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry", ctrl{kind: block.BlockJumpTable, succs: []string{"first", "shared", "shared", "last"}}),
		Bloc("first", Goto("exit")),
		Bloc("shared", Goto("exit")),
		Bloc("last", Goto("exit")),
		Bloc("exit", Exit("")),
	)
	sdom := f.f.Sdom()
	for _, tt := range []struct {
		name string
		want branch
	}{
		{"first", jumpTable0},
		{"shared", unknown},
		{"last", jumpTable0 + 3},
		{"exit", unknown},
	} {
		if got := getBranch(sdom, f.blocks["entry"], f.blocks[tt.name]); got != tt.want {
			t.Errorf("getBranch(entry, %s) = %v; want %v", tt.name, got, tt.want)
		}
	}
}

func BenchmarkGetBranchJumpTable(b *testing.B) {
	for _, n := range []int{8, 1000, 2000, 4000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			c := testConfig(b)
			f := c.config.NewFunc(c.Frontend(), new(ssa.Cache))
			p := f.NewBlock(block.BlockJumpTable)
			f.Entry = p
			exit := f.NewBlock(block.BlockExit)
			for range n {
				child := f.NewBlock(block.BlockPlain)
				p.AddEdgeTo(child)
				child.AddEdgeTo(exit)
			}
			sdom := f.Sdom()
			for b.Loop() {
				for _, e := range p.Succs {
					getBranch(sdom, p, e.B)
				}
			}
		})
	}
}

func testLimitUnaryOpSigned8(t *testing.T, opName string, initLimit ssa.Limit, op func(l ssa.Limit, bitsize uint) ssa.Limit, opImpl func(int8) int8) {
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

			l := ssa.Limit{Min: int64(min), Max: int64(max), Umin: 0, Umax: math.MaxUint64}
			l = op(l, 8)
			l = l.Intersect(initLimit) // We assume this is gonna be used by newLimit which is seeded by the op size already.

			if l.Min != int64(realSmallest) || l.Max != int64(realBiggest) {
				t.Errorf("%s(%d..%d) = %d..%d; want %d..%d", opName, min, max, l.Min, l.Max, realSmallest, realBiggest)
			}
		}
	}
}

func testLimitUnaryOpUnsigned8(t *testing.T, opName string, initLimit ssa.Limit, op func(l ssa.Limit, bitsize uint) ssa.Limit, opImpl func(uint8) uint8) {
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

			l := ssa.Limit{Min: math.MinInt64, Max: math.MaxInt64, Umin: uint64(min), Umax: uint64(max)}
			l = op(l, 8)
			l = l.Intersect(initLimit) // We assume this is gonna be used by newLimit which is seeded by the op size already.

			if l.Umin != uint64(realSmallest) || l.Umax != uint64(realBiggest) {
				t.Errorf("%s(%d..%d) = %d..%d; want %d..%d", opName, min, max, l.Umin, l.Umax, realSmallest, realBiggest)
			}
		}
	}
}

func TestLimitNegSigned(t *testing.T) {
	testLimitUnaryOpSigned8(t, "neg", ssa.NoLimitForBitsize(8), ssa.Limit.Neg, func(x int8) int8 { return -x })
}
func TestLimitNegUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "neg", ssa.NoLimitForBitsize(8), ssa.Limit.Neg, func(x uint8) uint8 { return -x })
}

func TestLimitComSigned(t *testing.T) {
	testLimitUnaryOpSigned8(t, "com", ssa.NoLimitForBitsize(8), ssa.Limit.Com, func(x int8) int8 { return ^x })
}
func TestLimitComUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "com", ssa.NoLimitForBitsize(8), ssa.Limit.Com, func(x uint8) uint8 { return ^x })
}

func TestLimitCtzUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "ctz", ssa.Limit{Min: -128, Max: 127, Umin: 0, Umax: 8}, ssa.Limit.Ctz, func(x uint8) uint8 { return uint8(bits.TrailingZeros8(x)) })
}

func TestLimitBitlenUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "bitlen", ssa.Limit{Min: -128, Max: 127, Umin: 0, Umax: 8}, ssa.Limit.Bitlen, func(x uint8) uint8 { return uint8(bits.Len8(x)) })
}

func TestLimitPopcountUnsigned(t *testing.T) {
	testLimitUnaryOpUnsigned8(t, "popcount", ssa.Limit{Min: -128, Max: 127, Umin: 0, Umax: 8}, ssa.Limit.Popcount, func(x uint8) uint8 { return uint8(bits.OnesCount8(x)) })
}

func TestConvertIntWithBitsize(t *testing.T) {
	if got := ssa.ConvertIntWithBitsize[int64, uint64](255, 8); got != -1 {
		t.Errorf("convertIntWithBitsize(255, 8) = %d; want -1", got)
	}
	if got := ssa.ConvertIntWithBitsize[uint64, int64](-1, 8); got != 255 {
		t.Errorf("convertIntWithBitsize(-1, 8) = %d; want 255", got)
	}

	if got := ssa.ConvertIntWithBitsize[int64, uint64](127, 8); got != 127 {
		t.Errorf("convertIntWithBitsize(127, 8) = %d; want 127", got)
	}
	if got := ssa.ConvertIntWithBitsize[uint64, int64](127, 8); got != 127 {
		t.Errorf("convertIntWithBitsize(127, 8) = %d; want 127", got)
	}
}

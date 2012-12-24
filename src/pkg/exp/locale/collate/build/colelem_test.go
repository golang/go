// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"exp/locale/collate"
	"testing"
)

type ceTest struct {
	f   func(in []int) (uint32, error)
	arg []int
	val uint32
}

func normalCE(in []int) (ce uint32, err error) {
	return makeCE(rawCE{w: in[:3], ccc: uint8(in[3])})
}

func expandCE(in []int) (ce uint32, err error) {
	return makeExpandIndex(in[0])
}

func contractCE(in []int) (ce uint32, err error) {
	return makeContractIndex(ctHandle{in[0], in[1]}, in[2])
}

func decompCE(in []int) (ce uint32, err error) {
	return makeDecompose(in[0], in[1])
}

var ceTests = []ceTest{
	{normalCE, []int{0, 0, 0, 0}, 0xA0000000},
	{normalCE, []int{0, 0x28, 3, 0}, 0xA0002803},
	{normalCE, []int{0, 0x28, 3, 0xFF}, 0xAFF02803},
	{normalCE, []int{100, defaultSecondary, 3, 0}, 0x0000C883},
	// non-ignorable primary with non-default secondary
	{normalCE, []int{100, 0x28, defaultTertiary, 0}, 0x4000C828},
	{normalCE, []int{100, defaultSecondary + 8, 3, 0}, 0x0000C983},
	{normalCE, []int{100, 0, 3, 0}, 0xFFFF}, // non-ignorable primary with non-supported secondary
	{normalCE, []int{100, 1, 3, 0}, 0xFFFF},
	{normalCE, []int{1 << maxPrimaryBits, defaultSecondary, 0, 0}, 0xFFFF},
	{normalCE, []int{0, 1 << maxSecondaryBits, 0, 0}, 0xFFFF},
	{normalCE, []int{100, defaultSecondary, 1 << maxTertiaryBits, 0}, 0xFFFF},
	{normalCE, []int{0x123, defaultSecondary, 8, 0xFF}, 0x88FF0123},
	{normalCE, []int{0x123, defaultSecondary + 1, 8, 0xFF}, 0xFFFF},

	{contractCE, []int{0, 0, 0}, 0xC0000000},
	{contractCE, []int{1, 1, 1}, 0xC0010011},
	{contractCE, []int{1, (1 << maxNBits) - 1, 1}, 0xC001001F},
	{contractCE, []int{(1 << maxTrieIndexBits) - 1, 1, 1}, 0xC001FFF1},
	{contractCE, []int{1, 1, (1 << maxContractOffsetBits) - 1}, 0xDFFF0011},
	{contractCE, []int{1, (1 << maxNBits), 1}, 0xFFFF},
	{contractCE, []int{(1 << maxTrieIndexBits), 1, 1}, 0xFFFF},
	{contractCE, []int{1, (1 << maxContractOffsetBits), 1}, 0xFFFF},

	{expandCE, []int{0}, 0xE0000000},
	{expandCE, []int{5}, 0xE0000005},
	{expandCE, []int{(1 << maxExpandIndexBits) - 1}, 0xE000FFFF},
	{expandCE, []int{1 << maxExpandIndexBits}, 0xFFFF},

	{decompCE, []int{0, 0}, 0xF0000000},
	{decompCE, []int{1, 1}, 0xF0000101},
	{decompCE, []int{0x1F, 0x1F}, 0xF0001F1F},
	{decompCE, []int{256, 0x1F}, 0xFFFF},
	{decompCE, []int{0x1F, 256}, 0xFFFF},
}

func TestColElem(t *testing.T) {
	for i, tt := range ceTests {
		in := make([]int, len(tt.arg))
		copy(in, tt.arg)
		ce, err := tt.f(in)
		if tt.val == 0xFFFF {
			if err == nil {
				t.Errorf("%d: expected error for args %x", i, tt.arg)
			}
			continue
		}
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err.Error())
		}
		if ce != tt.val {
			t.Errorf("%d: colElem=%X; want %X", i, ce, tt.val)
		}
	}
}

func mkRawCES(in [][]int) []rawCE {
	out := []rawCE{}
	for _, w := range in {
		out = append(out, rawCE{w: w})
	}
	return out
}

type weightsTest struct {
	a, b   [][]int
	level  collate.Level
	result int
}

var nextWeightTests = []weightsTest{
	{
		a:     [][]int{{100, 20, 5, 0}},
		b:     [][]int{{101, defaultSecondary, defaultTertiary, 0}},
		level: collate.Primary,
	},
	{
		a:     [][]int{{100, 20, 5, 0}},
		b:     [][]int{{100, 21, defaultTertiary, 0}},
		level: collate.Secondary,
	},
	{
		a:     [][]int{{100, 20, 5, 0}},
		b:     [][]int{{100, 20, 6, 0}},
		level: collate.Tertiary,
	},
	{
		a:     [][]int{{100, 20, 5, 0}},
		b:     [][]int{{100, 20, 5, 0}},
		level: collate.Identity,
	},
}

var extra = [][]int{{200, 32, 8, 0}, {0, 32, 8, 0}, {0, 0, 8, 0}, {0, 0, 0, 0}}

func TestNextWeight(t *testing.T) {
	for i, tt := range nextWeightTests {
		test := func(l collate.Level, tt weightsTest, a, gold [][]int) {
			res := nextWeight(tt.level, mkRawCES(a))
			if !equalCEArrays(mkRawCES(gold), res) {
				t.Errorf("%d:%d: expected weights %d; found %d", i, l, gold, res)
			}
		}
		test(-1, tt, tt.a, tt.b)
		for l := collate.Primary; l <= collate.Tertiary; l++ {
			if tt.level <= l {
				test(l, tt, append(tt.a, extra[l]), tt.b)
			} else {
				test(l, tt, append(tt.a, extra[l]), append(tt.b, extra[l]))
			}
		}
	}
}

var compareTests = []weightsTest{
	{
		[][]int{{100, 20, 5, 0}},
		[][]int{{100, 20, 5, 0}},
		collate.Identity,
		0,
	},
	{
		[][]int{{100, 20, 5, 0}, extra[0]},
		[][]int{{100, 20, 5, 1}},
		collate.Primary,
		1,
	},
	{
		[][]int{{100, 20, 5, 0}},
		[][]int{{101, 20, 5, 0}},
		collate.Primary,
		-1,
	},
	{
		[][]int{{101, 20, 5, 0}},
		[][]int{{100, 20, 5, 0}},
		collate.Primary,
		1,
	},
	{
		[][]int{{100, 0, 0, 0}, {0, 20, 5, 0}},
		[][]int{{0, 20, 5, 0}, {100, 0, 0, 0}},
		collate.Identity,
		0,
	},
	{
		[][]int{{100, 20, 5, 0}},
		[][]int{{100, 21, 5, 0}},
		collate.Secondary,
		-1,
	},
	{
		[][]int{{100, 20, 5, 0}},
		[][]int{{100, 20, 2, 0}},
		collate.Tertiary,
		1,
	},
	{
		[][]int{{100, 20, 5, 1}},
		[][]int{{100, 20, 5, 2}},
		collate.Quaternary,
		-1,
	},
}

func TestCompareWeights(t *testing.T) {
	for i, tt := range compareTests {
		test := func(tt weightsTest, a, b [][]int) {
			res, level := compareWeights(mkRawCES(a), mkRawCES(b))
			if res != tt.result {
				t.Errorf("%d: expected comparisson result %d; found %d", i, tt.result, res)
			}
			if level != tt.level {
				t.Errorf("%d: expected level %d; found %d", i, tt.level, level)
			}
		}
		test(tt, tt.a, tt.b)
		test(tt, append(tt.a, extra[0]), append(tt.b, extra[0]))
	}
}

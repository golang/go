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
	return makeCE(in)
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
	{normalCE, []int{0, 0, 0}, 0x80000000},
	{normalCE, []int{0, 0x28, 3}, 0x80002803},
	{normalCE, []int{100, defaultSecondary, 3}, 0x0000C803},
	// non-ignorable primary with non-default secondary
	{normalCE, []int{100, 0x28, defaultTertiary}, 0x40006428},
	{normalCE, []int{100, defaultSecondary + 8, 3}, 0x0000C903},
	{normalCE, []int{100, 0, 3}, 0xFFFF}, // non-ignorable primary with non-supported secondary
	{normalCE, []int{100, 1, 3}, 0xFFFF},
	{normalCE, []int{1 << maxPrimaryBits, defaultSecondary, 0}, 0xFFFF},
	{normalCE, []int{0, 1 << maxSecondaryBits, 0}, 0xFFFF},
	{normalCE, []int{100, defaultSecondary, 1 << maxTertiaryBits}, 0xFFFF},

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

var extra = []int{200, 32, 8, 0}

func TestNextWeight(t *testing.T) {
	for i, tt := range nextWeightTests {
		test := func(tt weightsTest, a, gold [][]int) {
			res := nextWeight(tt.level, a)
			if !equalCEArrays(gold, res) {
				t.Errorf("%d: expected weights %d; found %d", i, tt.b, res)
			}
		}
		test(tt, tt.a, tt.b)
		test(tt, append(tt.a, extra), append(tt.b, extra))
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
		[][]int{{100, 20, 5, 0}, extra},
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
			res, level := compareWeights(a, b)
			if res != tt.result {
				t.Errorf("%d: expected comparisson result %d; found %d", i, tt.result, res)
			}
			if level != tt.level {
				t.Errorf("%d: expected level %d; found %d", i, tt.level, level)
			}
		}
		test(tt, tt.a, tt.b)
		test(tt, append(tt.a, extra), append(tt.b, extra))
	}
}

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

import (
	"testing"
	"unicode"
)

type ceTest struct {
	f   func(inout []int) (colElem, ceType)
	arg []int
}

// The make* funcs are simplified versions of the functions in build/colelem.go
func makeCE(weights []int) colElem {
	const (
		maxPrimaryBits   = 21
		maxSecondaryBits = 16
		maxTertiaryBits  = 8
		isSecondary      = 0x40000000
	)
	var ce colElem
	if weights[0] != 0 {
		ce = colElem(weights[0]<<maxTertiaryBits + weights[2])
	} else {
		ce = colElem(weights[1]<<maxTertiaryBits + weights[2])
		ce |= isSecondary
	}
	return ce
}

func makeContractIndex(index, n, offset int) colElem {
	const (
		contractID       = 0x80000000
		maxNBits         = 5
		maxTrieIndexBits = 11
	)
	ce := colElem(contractID)
	ce += colElem(offset << (maxTrieIndexBits + maxNBits))
	ce += colElem(index << maxNBits)
	ce += colElem(n)
	return ce
}

func makeExpandIndex(index int) colElem {
	const expandID = 0xC0000000
	return expandID + colElem(index)
}

func makeDecompose(t1, t2 int) colElem {
	const decompID = 0xE0000000
	return colElem(t2<<8+t1) + decompID
}

func normalCE(inout []int) (ce colElem, t ceType) {
	w := splitCE(makeCE(inout))
	inout[0] = int(w.primary)
	inout[1] = int(w.secondary)
	inout[2] = int(w.tertiary)
	return ce, ceNormal
}

func expandCE(inout []int) (ce colElem, t ceType) {
	ce = makeExpandIndex(inout[0])
	inout[0] = splitExpandIndex(ce)
	return ce, ceExpansionIndex
}

func contractCE(inout []int) (ce colElem, t ceType) {
	ce = makeContractIndex(inout[0], inout[1], inout[2])
	i, n, o := splitContractIndex(ce)
	inout[0], inout[1], inout[2] = i, n, o
	return ce, ceContractionIndex
}

func decompCE(inout []int) (ce colElem, t ceType) {
	ce = makeDecompose(inout[0], inout[1])
	t1, t2 := splitDecompose(ce)
	inout[0], inout[1] = int(t1), int(t2)
	return ce, ceDecompose
}

const (
	maxPrimaryBits   = 21
	maxSecondaryBits = 16
	maxTertiaryBits  = 8
)

var ceTests = []ceTest{
	{normalCE, []int{0, 0, 0}},
	{normalCE, []int{0, 30, 3}},
	{normalCE, []int{100, defaultSecondary, 3}},

	{contractCE, []int{0, 0, 0}},
	{contractCE, []int{1, 1, 1}},
	{contractCE, []int{1, (1 << maxNBits) - 1, 1}},
	{contractCE, []int{(1 << maxTrieIndexBits) - 1, 1, 1}},
	{contractCE, []int{1, 1, (1 << maxContractOffsetBits) - 1}},

	{expandCE, []int{0}},
	{expandCE, []int{5}},
	{expandCE, []int{(1 << maxExpandIndexBits) - 1}},

	{decompCE, []int{0, 0}},
	{decompCE, []int{1, 1}},
	{decompCE, []int{0x1F, 0x1F}},
}

func TestColElem(t *testing.T) {
	for i, tt := range ceTests {
		inout := make([]int, len(tt.arg))
		copy(inout, tt.arg)
		ce, typ := tt.f(inout)
		if ce.ctype() != typ {
			t.Errorf("%d: type is %d; want %d", i, ce.ctype(), typ)
		}
		for j, a := range tt.arg {
			if inout[j] != a {
				t.Errorf("%d: argument %d is %d; want %d", i, j, inout[j], a)
			}
		}
	}
}

type implicitTest struct {
	r rune
	p int
}

var implicitTests = []implicitTest{
	{0x33FF, 0x52F3F},
	{0x3400, 0x22F40},
	{0x4DC0, 0x54900},
	{0x4DFF, 0x5493F},
	{0x4E00, 0x14940},
	{0x9FCB, 0x19B0B},
	{0xA000, 0x59B40},
	{0xF8FF, 0x5F43F},
	{0xF900, 0x1F440},
	{0xFA23, 0x1F563},
	{0xFAFF, 0x1F63F},
	{0xFB00, 0x5F640},
	{0x20000, 0x3FB40},
	{0x2B81C, 0x4B35C},
	{unicode.MaxRune, 0x15FB3F}, // maximum primary value
}

func TestImplicit(t *testing.T) {
	for _, tt := range implicitTests {
		if p := implicitPrimary(tt.r); p != tt.p {
			t.Errorf("%U: was %X; want %X", tt.r, p, tt.p)
		}
	}
}

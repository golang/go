// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colltab

import (
	"testing"
	"unicode"
)

type ceTest struct {
	f   func(inout []int) (Elem, ceType)
	arg []int
}

func makeCE(weights []int) Elem {
	ce, _ := MakeElem(weights[0], weights[1], weights[2], uint8(weights[3]))
	return ce
}

func makeContractIndex(index, n, offset int) Elem {
	const (
		contractID            = 0xC0000000
		maxNBits              = 4
		maxTrieIndexBits      = 12
		maxContractOffsetBits = 13
	)
	ce := Elem(contractID)
	ce += Elem(offset << (maxNBits + maxTrieIndexBits))
	ce += Elem(index << maxNBits)
	ce += Elem(n)
	return ce
}

func makeExpandIndex(index int) Elem {
	const expandID = 0xE0000000
	return expandID + Elem(index)
}

func makeDecompose(t1, t2 int) Elem {
	const decompID = 0xF0000000
	return Elem(t2<<8+t1) + decompID
}

func normalCE(inout []int) (ce Elem, t ceType) {
	ce = makeCE(inout)
	inout[0] = ce.Primary()
	inout[1] = ce.Secondary()
	inout[2] = int(ce.Tertiary())
	inout[3] = int(ce.CCC())
	return ce, ceNormal
}

func expandCE(inout []int) (ce Elem, t ceType) {
	ce = makeExpandIndex(inout[0])
	inout[0] = splitExpandIndex(ce)
	return ce, ceExpansionIndex
}

func contractCE(inout []int) (ce Elem, t ceType) {
	ce = makeContractIndex(inout[0], inout[1], inout[2])
	i, n, o := splitContractIndex(ce)
	inout[0], inout[1], inout[2] = i, n, o
	return ce, ceContractionIndex
}

func decompCE(inout []int) (ce Elem, t ceType) {
	ce = makeDecompose(inout[0], inout[1])
	t1, t2 := splitDecompose(ce)
	inout[0], inout[1] = int(t1), int(t2)
	return ce, ceDecompose
}

var ceTests = []ceTest{
	{normalCE, []int{0, 0, 0, 0}},
	{normalCE, []int{0, 30, 3, 0}},
	{normalCE, []int{0, 30, 3, 0xFF}},
	{normalCE, []int{100, defaultSecondary, defaultTertiary, 0}},
	{normalCE, []int{100, defaultSecondary, defaultTertiary, 0xFF}},
	{normalCE, []int{100, defaultSecondary, 3, 0}},
	{normalCE, []int{0x123, defaultSecondary, 8, 0xFF}},

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
			t.Errorf("%d: type is %d; want %d (ColElem: %X)", i, ce.ctype(), typ, ce)
		}
		for j, a := range tt.arg {
			if inout[j] != a {
				t.Errorf("%d: argument %d is %X; want %X (ColElem: %X)", i, j, inout[j], a, ce)
			}
		}
	}
}

type implicitTest struct {
	r rune
	p int
}

var implicitTests = []implicitTest{
	{0x33FF, 0x533FF},
	{0x3400, 0x23400},
	{0x4DC0, 0x54DC0},
	{0x4DFF, 0x54DFF},
	{0x4E00, 0x14E00},
	{0x9FCB, 0x19FCB},
	{0xA000, 0x5A000},
	{0xF8FF, 0x5F8FF},
	{0xF900, 0x1F900},
	{0xFA23, 0x1FA23},
	{0xFAD9, 0x1FAD9},
	{0xFB00, 0x5FB00},
	{0x20000, 0x40000},
	{0x2B81C, 0x4B81C},
	{unicode.MaxRune, 0x15FFFF}, // maximum primary value
}

func TestImplicit(t *testing.T) {
	for _, tt := range implicitTests {
		if p := implicitPrimary(tt.r); p != tt.p {
			t.Errorf("%U: was %X; want %X", tt.r, p, tt.p)
		}
	}
}

func TestUpdateTertiary(t *testing.T) {
	tests := []struct {
		in, out Elem
		t       uint8
	}{
		{0x4000FE20, 0x0000FE8A, 0x0A},
		{0x4000FE21, 0x0000FEAA, 0x0A},
		{0x0000FE8B, 0x0000FE83, 0x03},
		{0x82FF0188, 0x9BFF0188, 0x1B},
		{0xAFF0CC02, 0xAFF0CC1B, 0x1B},
	}
	for i, tt := range tests {
		if out := tt.in.updateTertiary(tt.t); out != tt.out {
			t.Errorf("%d: was %X; want %X", i, out, tt.out)
		}
	}
}

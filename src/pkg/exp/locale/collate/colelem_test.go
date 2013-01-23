// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

import (
	"testing"
	"unicode"
)

type ceTest struct {
	f   func(inout []int) (Elem, ceType)
	arg []int
}

// The make* funcs are simplified versions of the functions in build/colelem.go
func makeCE(weights []int) Elem {
	const (
		maxPrimaryBits          = 21
		maxSecondaryBits        = 12
		maxSecondaryCompactBits = 8
		maxSecondaryDiffBits    = 4
		maxTertiaryBits         = 8
		maxTertiaryCompactBits  = 5
		isPrimary               = 0x40000000
		isPrimaryCCC            = 0x80000000
		isSecondary             = 0xA0000000
	)
	var ce Elem
	ccc := weights[3]
	if weights[0] != 0 {
		if ccc != 0 {
			ce = Elem(weights[2] << 24)
			ce |= Elem(ccc) << 16
			ce |= Elem(weights[0])
			ce |= isPrimaryCCC
		} else if weights[2] == defaultTertiary {
			ce = Elem(weights[0]<<(maxSecondaryCompactBits+1) + weights[1])
			ce |= isPrimary
		} else {
			d := weights[1] - defaultSecondary + 4
			ce = Elem(weights[0]<<maxSecondaryDiffBits + d)
			ce = ce<<maxTertiaryCompactBits + Elem(weights[2])
		}
	} else {
		ce = Elem(weights[1]<<maxTertiaryBits + weights[2])
		ce += Elem(ccc) << 20
		ce |= isSecondary
	}
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

const (
	maxPrimaryBits   = 21
	maxSecondaryBits = 16
	maxTertiaryBits  = 8
)

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

func TestDoNorm(t *testing.T) {
	const div = -1 // The insertion point of the next block.
	tests := []struct {
		in, out []int
	}{
		{in: []int{4, div, 3},
			out: []int{3, 4},
		},
		{in: []int{4, div, 3, 3, 3},
			out: []int{3, 3, 3, 4},
		},
		{in: []int{0, 4, div, 3},
			out: []int{0, 3, 4},
		},
		{in: []int{0, 0, 4, 5, div, 3, 3},
			out: []int{0, 0, 3, 3, 4, 5},
		},
		{in: []int{0, 0, 1, 4, 5, div, 3, 3},
			out: []int{0, 0, 1, 3, 3, 4, 5},
		},
		{in: []int{0, 0, 1, 4, 5, div, 4, 4},
			out: []int{0, 0, 1, 4, 4, 4, 5},
		},
	}
	for j, tt := range tests {
		i := iter{}
		var w, p, s int
		for k, cc := range tt.in {
			if cc == 0 {
				s = 0
			}
			if cc == div {
				w = 100
				p = k
				i.pStarter = s
				continue
			}
			i.ce = append(i.ce, makeCE([]int{w, 20, 2, cc}))
		}
		i.prevCCC = i.ce[p-1].CCC()
		i.doNorm(p, i.ce[p].CCC())
		if len(i.ce) != len(tt.out) {
			t.Errorf("%d: length was %d; want %d", j, len(i.ce), len(tt.out))
		}
		prevCCC := uint8(0)
		for k, ce := range i.ce {
			if int(ce.CCC()) != tt.out[k] {
				t.Errorf("%d:%d: unexpected CCC. Was %d; want %d", j, k, ce.CCC(), tt.out[k])
			}
			if k > 0 && ce.CCC() == prevCCC && i.ce[k-1].Primary() > ce.Primary() {
				t.Errorf("%d:%d: normalization crossed across CCC boundary.", j, k)
			}
		}
	}
	// test cutoff of large sequence of combining characters.
	result := []uint8{8, 8, 8, 5, 5}
	for o := -2; o <= 2; o++ {
		i := iter{pStarter: 2, prevCCC: 8}
		n := maxCombiningCharacters + 1 + o
		for j := 1; j < n+i.pStarter; j++ {
			i.ce = append(i.ce, makeCE([]int{100, 20, 2, 8}))
		}
		p := len(i.ce)
		i.ce = append(i.ce, makeCE([]int{0, 20, 2, 5}))
		i.doNorm(p, 5)
		if i.prevCCC != result[o+2] {
			t.Errorf("%d: i.prevCCC was %d; want %d", n, i.prevCCC, result[o+2])
		}
		if result[o+2] == 5 && i.pStarter != p {
			t.Errorf("%d: i.pStarter was %d; want %d", n, i.pStarter, p)
		}
	}
}

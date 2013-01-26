// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import "testing"

// cjk returns an implicit collation element for a CJK rune.
func cjk(r rune) []rawCE {
	// A CJK character C is represented in the DUCET as
	//   [.AAAA.0020.0002.C][.BBBB.0000.0000.C]
	// Where AAAA is the most significant 15 bits plus a base value.
	// Any base value will work for the test, so we pick the common value of FB40.
	const base = 0xFB40
	return []rawCE{
		{w: []int{base + int(r>>15), defaultSecondary, defaultTertiary, int(r)}},
		{w: []int{int(r&0x7FFF) | 0x8000, 0, 0, int(r)}},
	}
}

func pCE(p int) []rawCE {
	return mkCE([]int{p, defaultSecondary, defaultTertiary, 0}, 0)
}

func pqCE(p, q int) []rawCE {
	return mkCE([]int{p, defaultSecondary, defaultTertiary, q}, 0)
}

func ptCE(p, t int) []rawCE {
	return mkCE([]int{p, defaultSecondary, t, 0}, 0)
}

func ptcCE(p, t int, ccc uint8) []rawCE {
	return mkCE([]int{p, defaultSecondary, t, 0}, ccc)
}

func sCE(s int) []rawCE {
	return mkCE([]int{0, s, defaultTertiary, 0}, 0)
}

func stCE(s, t int) []rawCE {
	return mkCE([]int{0, s, t, 0}, 0)
}

func scCE(s int, ccc uint8) []rawCE {
	return mkCE([]int{0, s, defaultTertiary, 0}, ccc)
}

func mkCE(w []int, ccc uint8) []rawCE {
	return []rawCE{rawCE{w, ccc}}
}

// ducetElem is used to define test data that is used to generate a table.
type ducetElem struct {
	str string
	ces []rawCE
}

func newBuilder(t *testing.T, ducet []ducetElem) *Builder {
	b := NewBuilder()
	for _, e := range ducet {
		ces := [][]int{}
		for _, ce := range e.ces {
			ces = append(ces, ce.w)
		}
		if err := b.Add([]rune(e.str), ces, nil); err != nil {
			t.Errorf(err.Error())
		}
	}
	b.t = &table{}
	b.root.sort()
	return b
}

type convertTest struct {
	in, out []rawCE
	err     bool
}

var convLargeTests = []convertTest{
	{pCE(0xFB39), pCE(0xFB39), false},
	{cjk(0x2F9B2), pqCE(0x3F9B2, 0x2F9B2), false},
	{pCE(0xFB40), pCE(0), true},
	{append(pCE(0xFB40), pCE(0)[0]), pCE(0), true},
	{pCE(0xFFFE), pCE(illegalOffset), false},
	{pCE(0xFFFF), pCE(illegalOffset + 1), false},
}

func TestConvertLarge(t *testing.T) {
	for i, tt := range convLargeTests {
		e := new(entry)
		for _, ce := range tt.in {
			e.elems = append(e.elems, makeRawCE(ce.w, ce.ccc))
		}
		elems, err := convertLargeWeights(e.elems)
		if tt.err {
			if err == nil {
				t.Errorf("%d: expected error; none found", i)
			}
			continue
		} else if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		}
		if !equalCEArrays(elems, tt.out) {
			t.Errorf("%d: conversion was %x; want %x", i, elems, tt.out)
		}
	}
}

// Collation element table for simplify tests.
var simplifyTest = []ducetElem{
	{"\u0300", sCE(30)}, // grave
	{"\u030C", sCE(40)}, // caron
	{"A", ptCE(100, 8)},
	{"D", ptCE(104, 8)},
	{"E", ptCE(105, 8)},
	{"I", ptCE(110, 8)},
	{"z", ptCE(130, 8)},
	{"\u05F2", append(ptCE(200, 4), ptCE(200, 4)[0])},
	{"\u05B7", sCE(80)},
	{"\u00C0", append(ptCE(100, 8), sCE(30)...)},                                // A with grave, can be removed
	{"\u00C8", append(ptCE(105, 8), sCE(30)...)},                                // E with grave
	{"\uFB1F", append(ptCE(200, 4), ptCE(200, 4)[0], sCE(80)[0])},               // eliminated by NFD
	{"\u00C8\u0302", ptCE(106, 8)},                                              // block previous from simplifying
	{"\u01C5", append(ptCE(104, 9), ptCE(130, 4)[0], stCE(40, maxTertiary)[0])}, // eliminated by NFKD
	// no removal: tertiary value of third element is not maxTertiary
	{"\u2162", append(ptCE(110, 9), ptCE(110, 4)[0], ptCE(110, 8)[0])},
}

var genColTests = []ducetElem{
	{"\uFA70", pqCE(0x1FA70, 0xFA70)},
	{"A\u0300", append(ptCE(100, 8), sCE(30)...)},
	{"A\u0300\uFA70", append(ptCE(100, 8), sCE(30)[0], pqCE(0x1FA70, 0xFA70)[0])},
	{"A\u0300A\u0300", append(ptCE(100, 8), sCE(30)[0], ptCE(100, 8)[0], sCE(30)[0])},
}

func TestGenColElems(t *testing.T) {
	b := newBuilder(t, simplifyTest[:5])

	for i, tt := range genColTests {
		res := b.root.genColElems(tt.str)
		if !equalCEArrays(tt.ces, res) {
			t.Errorf("%d: result %X; want %X", i, res, tt.ces)
		}
	}
}

type strArray []string

func (sa strArray) contains(s string) bool {
	for _, e := range sa {
		if e == s {
			return true
		}
	}
	return false
}

var simplifyRemoved = strArray{"\u00C0", "\uFB1F"}
var simplifyMarked = strArray{"\u01C5"}

func TestSimplify(t *testing.T) {
	b := newBuilder(t, simplifyTest)
	o := &b.root
	simplify(o)

	for i, tt := range simplifyTest {
		if simplifyRemoved.contains(tt.str) {
			continue
		}
		e := o.find(tt.str)
		if e.str != tt.str || !equalCEArrays(e.elems, tt.ces) {
			t.Errorf("%d: found element %s -> %X; want %s -> %X", i, e.str, e.elems, tt.str, tt.ces)
			break
		}
	}
	var i, k int
	for e := o.front(); e != nil; e, _ = e.nextIndexed() {
		gold := simplifyMarked.contains(e.str)
		if gold {
			k++
		}
		if gold != e.decompose {
			t.Errorf("%d: %s has decompose %v; want %v", i, e.str, e.decompose, gold)
		}
		i++
	}
	if k != len(simplifyMarked) {
		t.Errorf(" an entry that should be marked as decompose was deleted")
	}
}

var expandTest = []ducetElem{
	{"\u0300", append(scCE(29, 230), scCE(30, 230)...)},
	{"\u00C0", append(ptCE(100, 8), scCE(30, 230)...)},
	{"\u00C8", append(ptCE(105, 8), scCE(30, 230)...)},
	{"\u00C9", append(ptCE(105, 8), scCE(30, 230)...)}, // identical expansion
	{"\u05F2", append(ptCE(200, 4), ptCE(200, 4)[0], ptCE(200, 4)[0])},
	{"\u01FF", append(ptCE(200, 4), ptcCE(201, 4, 0)[0], scCE(30, 230)[0])},
}

func TestExpand(t *testing.T) {
	const (
		totalExpansions = 5
		totalElements   = 2 + 2 + 2 + 3 + 3 + totalExpansions
	)
	b := newBuilder(t, expandTest)
	o := &b.root
	b.processExpansions(o)

	e := o.front()
	for _, tt := range expandTest {
		exp := b.t.expandElem[e.expansionIndex:]
		if int(exp[0]) != len(tt.ces) {
			t.Errorf("%U: len(expansion)==%d; want %d", []rune(tt.str)[0], exp[0], len(tt.ces))
		}
		exp = exp[1:]
		for j, w := range tt.ces {
			if ce, _ := makeCE(w); exp[j] != ce {
				t.Errorf("%U: element %d is %X; want %X", []rune(tt.str)[0], j, exp[j], ce)
			}
		}
		e, _ = e.nextIndexed()
	}
	// Verify uniquing.
	if len(b.t.expandElem) != totalElements {
		t.Errorf("len(expandElem)==%d; want %d", len(b.t.expandElem), totalElements)
	}
}

var contractTest = []ducetElem{
	{"abc", pCE(102)},
	{"abd", pCE(103)},
	{"a", pCE(100)},
	{"ab", pCE(101)},
	{"ac", pCE(104)},
	{"bcd", pCE(202)},
	{"b", pCE(200)},
	{"bc", pCE(201)},
	{"bd", pCE(203)},
	// shares suffixes with a*
	{"Ab", pCE(301)},
	{"A", pCE(300)},
	{"Ac", pCE(304)},
	{"Abc", pCE(302)},
	{"Abd", pCE(303)},
	// starter to be ignored
	{"z", pCE(1000)},
}

func TestContract(t *testing.T) {
	const (
		totalElements = 5 + 5 + 4
	)
	b := newBuilder(t, contractTest)
	o := &b.root
	b.processContractions(o)

	indexMap := make(map[int]bool)
	handleMap := make(map[rune]*entry)
	for e := o.front(); e != nil; e, _ = e.nextIndexed() {
		if e.contractionHandle.n > 0 {
			handleMap[e.runes[0]] = e
			indexMap[e.contractionHandle.index] = true
		}
	}
	// Verify uniquing.
	if len(indexMap) != 2 {
		t.Errorf("number of tries is %d; want %d", len(indexMap), 2)
	}
	for _, tt := range contractTest {
		e, ok := handleMap[[]rune(tt.str)[0]]
		if !ok {
			continue
		}
		str := tt.str[1:]
		offset, n := b.t.contractTries.lookup(e.contractionHandle, []byte(str))
		if len(str) != n {
			t.Errorf("%s: bytes consumed==%d; want %d", tt.str, n, len(str))
		}
		ce := b.t.contractElem[offset+e.contractionIndex]
		if want, _ := makeCE(tt.ces[0]); want != ce {
			t.Errorf("%s: element %X; want %X", tt.str, ce, want)
		}
	}
	if len(b.t.contractElem) != totalElements {
		t.Errorf("len(expandElem)==%d; want %d", len(b.t.contractElem), totalElements)
	}
}

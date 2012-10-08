// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"bytes"
	"sort"
	"testing"
)

var largetosmall = []stridx{
	{"a", 5},
	{"ab", 4},
	{"abc", 3},
	{"abcd", 2},
	{"abcde", 1},
	{"abcdef", 0},
}

var offsetSortTests = [][]stridx{
	{
		{"bcde", 1},
		{"bc", 5},
		{"ab", 4},
		{"bcd", 3},
		{"abcd", 0},
		{"abc", 2},
	},
	largetosmall,
}

func TestOffsetSort(t *testing.T) {
	for i, st := range offsetSortTests {
		sort.Sort(offsetSort(st))
		for j, si := range st {
			if j != si.index {
				t.Errorf("%d: failed: %v", i, st)
			}
		}
	}
	for i, tt := range genStateTests {
		// ensure input is well-formed
		sort.Sort(offsetSort(tt.in))
		for j, si := range tt.in {
			if si.index != j+1 {
				t.Errorf("%dth sort failed: %v", i, tt.in)
			}
		}
	}
}

var genidxtest1 = []stridx{
	{"bcde", 3},
	{"bc", 6},
	{"ab", 2},
	{"bcd", 5},
	{"abcd", 0},
	{"abc", 1},
	{"bcdf", 4},
}

var genidxSortTests = [][]stridx{
	genidxtest1,
	largetosmall,
}

func TestGenIdxSort(t *testing.T) {
	for i, st := range genidxSortTests {
		sort.Sort(genidxSort(st))
		for j, si := range st {
			if j != si.index {
				t.Errorf("%dth sort failed %v", i, st)
				break
			}
		}
	}
}

var entrySortTests = []contractTrieSet{
	{
		{10, 0, 1, 3},
		{99, 0, 1, 0},
		{20, 50, 0, 2},
		{30, 0, 1, 1},
	},
}

func TestEntrySort(t *testing.T) {
	for i, et := range entrySortTests {
		sort.Sort(entrySort(et))
		for j, fe := range et {
			if j != int(fe.i) {
				t.Errorf("%dth sort failed %v", i, et)
				break
			}
		}
	}
}

type GenStateTest struct {
	in            []stridx
	firstBlockLen int
	out           contractTrieSet
}

var genStateTests = []GenStateTest{
	{[]stridx{
		{"abc", 1},
	},
		1,
		contractTrieSet{
			{'a', 0, 1, noIndex},
			{'b', 0, 1, noIndex},
			{'c', 'c', final, 1},
		},
	},
	{[]stridx{
		{"abc", 1},
		{"abd", 2},
		{"abe", 3},
	},
		1,
		contractTrieSet{
			{'a', 0, 1, noIndex},
			{'b', 0, 1, noIndex},
			{'c', 'e', final, 1},
		},
	},
	{[]stridx{
		{"abc", 1},
		{"ab", 2},
		{"a", 3},
	},
		1,
		contractTrieSet{
			{'a', 0, 1, 3},
			{'b', 0, 1, 2},
			{'c', 'c', final, 1},
		},
	},
	{[]stridx{
		{"abc", 1},
		{"abd", 2},
		{"ab", 3},
		{"ac", 4},
		{"a", 5},
		{"b", 6},
	},
		2,
		contractTrieSet{
			{'b', 'b', final, 6},
			{'a', 0, 2, 5},
			{'c', 'c', final, 4},
			{'b', 0, 1, 3},
			{'c', 'd', final, 1},
		},
	},
	{[]stridx{
		{"bcde", 2},
		{"bc", 7},
		{"ab", 6},
		{"bcd", 5},
		{"abcd", 1},
		{"abc", 4},
		{"bcdf", 3},
	},
		2,
		contractTrieSet{
			{'b', 3, 1, noIndex},
			{'a', 0, 1, noIndex},
			{'b', 0, 1, 6},
			{'c', 0, 1, 4},
			{'d', 'd', final, 1},
			{'c', 0, 1, 7},
			{'d', 0, 1, 5},
			{'e', 'f', final, 2},
		},
	},
}

func TestGenStates(t *testing.T) {
	for i, tt := range genStateTests {
		si := []stridx{}
		for _, e := range tt.in {
			si = append(si, e)
		}
		// ensure input is well-formed
		sort.Sort(genidxSort(si))
		ct := contractTrieSet{}
		n, _ := ct.genStates(si)
		if nn := tt.firstBlockLen; nn != n {
			t.Errorf("%d: block len %v; want %v", i, n, nn)
		}
		if lv, lw := len(ct), len(tt.out); lv != lw {
			t.Errorf("%d: len %v; want %v", i, lv, lw)
			continue
		}
		for j, fe := range tt.out {
			const msg = "%d:%d: value %s=%v; want %v"
			if fe.l != ct[j].l {
				t.Errorf(msg, i, j, "l", ct[j].l, fe.l)
			}
			if fe.h != ct[j].h {
				t.Errorf(msg, i, j, "h", ct[j].h, fe.h)
			}
			if fe.n != ct[j].n {
				t.Errorf(msg, i, j, "n", ct[j].n, fe.n)
			}
			if fe.i != ct[j].i {
				t.Errorf(msg, i, j, "i", ct[j].i, fe.i)
			}
		}
	}
}

func TestLookupContraction(t *testing.T) {
	for i, tt := range genStateTests {
		input := []string{}
		for _, e := range tt.in {
			input = append(input, e.str)
		}
		cts := contractTrieSet{}
		h, _ := cts.appendTrie(input)
		for j, si := range tt.in {
			str := si.str
			for _, s := range []string{str, str + "X"} {
				msg := "%d:%d: %s(%s) %v; want %v"
				idx, sn := cts.lookup(h, []byte(s))
				if idx != si.index {
					t.Errorf(msg, i, j, "index", s, idx, si.index)
				}
				if sn != len(str) {
					t.Errorf(msg, i, j, "sn", s, sn, len(str))
				}
			}
		}
	}
}

func TestPrintContractionTrieSet(t *testing.T) {
	testdata := contractTrieSet(genStateTests[4].out)
	buf := &bytes.Buffer{}
	testdata.print(buf, "test")
	if contractTrieOutput != buf.String() {
		t.Errorf("output differs; found\n%s", buf.String())
		println(string(buf.Bytes()))
	}
}

const contractTrieOutput = `// testCTEntries: 8 entries, 32 bytes
var testCTEntries = [8]struct{l,h,n,i uint8}{
	{0x62, 0x3, 1, 255},
	{0x61, 0x0, 1, 255},
	{0x62, 0x0, 1, 6},
	{0x63, 0x0, 1, 4},
	{0x64, 0x64, 0, 1},
	{0x63, 0x0, 1, 7},
	{0x64, 0x0, 1, 5},
	{0x65, 0x66, 0, 2},
}
var testContractTrieSet = contractTrieSet( testCTEntries[:] )
`

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate_test

import (
	"exp/locale/collate"
	"exp/locale/collate/build"
	"exp/norm"
	"testing"
)

type ColElems []collate.Weights

type input struct {
	str string
	ces [][]int
}

type check struct {
	in  string
	n   int
	out ColElems
}

type tableTest struct {
	in  []input
	chk []check
}

func w(ce ...int) collate.Weights {
	return collate.W(ce...)
}

var defaults = w(0)

func pt(p, t int) []int {
	return []int{p, defaults.Secondary, t}
}

func makeTable(in []input) (*collate.Collator, error) {
	b := build.NewBuilder()
	for _, r := range in {
		b.Add([]rune(r.str), r.ces)
	}
	c, err := b.Build("")
	if err == nil {
		collate.InitCollator(c)
	}
	return c, err
}

// modSeq holds a seqeunce of modifiers in increasing order of CCC long enough
// to cause a segment overflow if not handled correctly. The last rune in this
// list has a CCC of 214.
var modSeq = []rune{
	0x05B1, 0x05B2, 0x05B3, 0x05B4, 0x05B5, 0x05B6, 0x05B7, 0x05B8, 0x05B9, 0x05BB,
	0x05BC, 0x05BD, 0x05BF, 0x05C1, 0x05C2, 0xFB1E, 0x064B, 0x064C, 0x064D, 0x064E,
	0x064F, 0x0650, 0x0651, 0x0652, 0x0670, 0x0711, 0x0C55, 0x0C56, 0x0E38, 0x0E48,
	0x0EB8, 0x0EC8, 0x0F71, 0x0F72, 0x0F74, 0x0321, 0x1DCE,
}

var mods []input
var modW = func() ColElems {
	ws := ColElems{}
	for _, r := range modSeq {
		rune := norm.NFC.PropertiesString(string(r))
		ws = append(ws, w(0, int(rune.CCC())))
		mods = append(mods, input{string(r), [][]int{{0, int(rune.CCC())}}})
	}
	return ws
}()

var appendNextTests = []tableTest{
	{ // test getWeights
		[]input{
			{"a", [][]int{{100}}},
			{"b", [][]int{{105}}},
			{"c", [][]int{{110}}},
			{"ß", [][]int{{120}}},
		},
		[]check{
			{"a", 1, ColElems{w(100)}},
			{"b", 1, ColElems{w(105)}},
			{"c", 1, ColElems{w(110)}},
			{"d", 1, ColElems{w(0x4FBA4)}},
			{"ab", 1, ColElems{w(100)}},
			{"bc", 1, ColElems{w(105)}},
			{"dd", 1, ColElems{w(0x4FBA4)}},
			{"ß", 2, ColElems{w(120)}},
		},
	},
	{ // test expansion
		[]input{
			{"u", [][]int{{100}}},
			{"U", [][]int{{100}, {0, 25}}},
			{"w", [][]int{{100}, {100}}},
			{"W", [][]int{{100}, {0, 25}, {100}, {0, 25}}},
		},
		[]check{
			{"u", 1, ColElems{w(100)}},
			{"U", 1, ColElems{w(100), w(0, 25)}},
			{"w", 1, ColElems{w(100), w(100)}},
			{"W", 1, ColElems{w(100), w(0, 25), w(100), w(0, 25)}},
		},
	},
	{ // test decompose
		[]input{
			{"D", [][]int{pt(104, 8)}},
			{"z", [][]int{pt(130, 8)}},
			{"\u030C", [][]int{{0, 40}}},                               // Caron
			{"\u01C5", [][]int{pt(104, 9), pt(130, 4), {0, 40, 0x1F}}}, // ǅ = D+z+caron
		},
		[]check{
			{"\u01C5", 2, ColElems{w(pt(104, 9)...), w(pt(130, 4)...), w(0, 40, 0x1F)}},
		},
	},
	{ // test basic contraction
		[]input{
			{"a", [][]int{{100}}},
			{"ab", [][]int{{101}}},
			{"aab", [][]int{{101}, {101}}},
			{"abc", [][]int{{102}}},
			{"b", [][]int{{200}}},
			{"c", [][]int{{300}}},
			{"d", [][]int{{400}}},
		},
		[]check{
			{"a", 1, ColElems{w(100)}},
			{"aa", 1, ColElems{w(100)}},
			{"aac", 1, ColElems{w(100)}},
			{"d", 1, ColElems{w(400)}},
			{"ab", 2, ColElems{w(101)}},
			{"abb", 2, ColElems{w(101)}},
			{"aab", 3, ColElems{w(101), w(101)}},
			{"aaba", 3, ColElems{w(101), w(101)}},
			{"abc", 3, ColElems{w(102)}},
			{"abcd", 3, ColElems{w(102)}},
		},
	},
	{ // test discontinuous contraction
		append(mods, []input{
			// modifiers; secondary weight equals ccc
			{"\u0316", [][]int{{0, 220}}},
			{"\u0317", [][]int{{0, 220}, {0, 220}}},
			{"\u302D", [][]int{{0, 222}}},
			{"\u302E", [][]int{{0, 224}}}, // used as starter
			{"\u302F", [][]int{{0, 224}}}, // used as starter
			{"\u18A9", [][]int{{0, 228}}},
			{"\u0300", [][]int{{0, 230}}},
			{"\u0301", [][]int{{0, 230}}},
			{"\u0315", [][]int{{0, 232}}},
			{"\u031A", [][]int{{0, 232}}},
			{"\u035C", [][]int{{0, 233}}},
			{"\u035F", [][]int{{0, 233}}},
			{"\u035D", [][]int{{0, 234}}},
			{"\u035E", [][]int{{0, 234}}},
			{"\u0345", [][]int{{0, 240}}},

			// starters
			{"a", [][]int{{100}}},
			{"b", [][]int{{200}}},
			{"c", [][]int{{300}}},
			{"\u03B1", [][]int{{900}}},

			// contractions
			{"a\u0300", [][]int{{101}}},
			{"a\u0301", [][]int{{102}}},
			{"a\u035E", [][]int{{110}}},
			{"a\u035Eb\u035E", [][]int{{115}}},
			{"ac\u035Eaca\u035E", [][]int{{116}}},
			{"a\u035Db\u035D", [][]int{{117}}},
			{"a\u0301\u035Db", [][]int{{120}}},
			{"a\u0301\u035F", [][]int{{121}}},
			{"a\u0301\u035Fb", [][]int{{122}}},
			{"\u03B1\u0345", [][]int{{901}, {902}}},
			{"\u302E\u18A9", [][]int{{0, 131}, {0, 132}}},
			{"\u302F\u18A9", [][]int{{0, 130}}},
		}...),
		[]check{
			{"ab", 1, ColElems{w(100)}},                              // closing segment
			{"a\u0316\u0300b", 5, ColElems{w(101), w(0, 220)}},       // closing segment
			{"a\u0316\u0300", 5, ColElems{w(101), w(0, 220)}},        // no closing segment
			{"a\u0316\u0300\u035Cb", 5, ColElems{w(101), w(0, 220)}}, // completes before segment end
			{"a\u0316\u0300\u035C", 5, ColElems{w(101), w(0, 220)}},  // completes before segment end

			{"a\u0316\u0301b", 5, ColElems{w(102), w(0, 220)}},       // closing segment
			{"a\u0316\u0301", 5, ColElems{w(102), w(0, 220)}},        // no closing segment
			{"a\u0316\u0301\u035Cb", 5, ColElems{w(102), w(0, 220)}}, // completes before segment end
			{"a\u0316\u0301\u035C", 5, ColElems{w(102), w(0, 220)}},  // completes before segment end

			// match blocked by modifier with same ccc
			{"a\u0301\u0315\u031A\u035Fb", 3, ColElems{w(102)}},

			// multiple gaps
			{"a\u0301\u035Db", 6, ColElems{w(120)}},
			{"a\u0301\u035F", 5, ColElems{w(121)}},
			{"a\u0301\u035Fb", 6, ColElems{w(122)}},
			{"a\u0316\u0301\u035F", 7, ColElems{w(121), w(0, 220)}},
			{"a\u0301\u0315\u035Fb", 7, ColElems{w(121), w(0, 232)}},
			{"a\u0316\u0301\u0315\u035Db", 5, ColElems{w(102), w(0, 220)}},
			{"a\u0316\u0301\u0315\u035F", 9, ColElems{w(121), w(0, 220), w(0, 232)}},
			{"a\u0316\u0301\u0315\u035Fb", 9, ColElems{w(121), w(0, 220), w(0, 232)}},
			{"a\u0316\u0301\u0315\u035F\u035D", 9, ColElems{w(121), w(0, 220), w(0, 232)}},
			{"a\u0316\u0301\u0315\u035F\u035Db", 9, ColElems{w(121), w(0, 220), w(0, 232)}},

			// handling of segment overflow
			{ // just fits within segment
				"a" + string(modSeq[:30]) + "\u0301",
				3 + len(string(modSeq[:30])),
				append(ColElems{w(102)}, modW[:30]...),
			},
			{"a" + string(modSeq[:31]) + "\u0301", 1, ColElems{w(100)}}, // overflow
			{"a" + string(modSeq) + "\u0301", 1, ColElems{w(100)}},
			{ // just fits within segment with two interstitial runes
				"a" + string(modSeq[:28]) + "\u0301\u0315\u035F",
				7 + len(string(modSeq[:28])),
				append(append(ColElems{w(121)}, modW[:28]...), w(0, 232)),
			},
			{ // second half does not fit within segment
				"a" + string(modSeq[:29]) + "\u0301\u0315\u035F",
				3 + len(string(modSeq[:29])),
				append(ColElems{w(102)}, modW[:29]...),
			},

			// discontinuity can only occur in last normalization segment
			{"a\u035Eb\u035E", 6, ColElems{w(115)}},
			{"a\u0316\u035Eb\u035E", 5, ColElems{w(110), w(0, 220)}},
			{"a\u035Db\u035D", 6, ColElems{w(117)}},
			{"a\u0316\u035Db\u035D", 1, ColElems{w(100)}},
			{"a\u035Eb\u0316\u035E", 8, ColElems{w(115), w(0, 220)}},
			{"a\u035Db\u0316\u035D", 8, ColElems{w(117), w(0, 220)}},
			{"ac\u035Eaca\u035E", 9, ColElems{w(116)}},
			{"a\u0316c\u035Eaca\u035E", 1, ColElems{w(100)}},
			{"ac\u035Eac\u0316a\u035E", 1, ColElems{w(100)}},

			// expanding contraction
			{"\u03B1\u0345", 4, ColElems{w(901), w(902)}},

			// Theoretical possibilities
			// contraction within a gap
			{"a\u302F\u18A9\u0301", 9, ColElems{w(102), w(0, 130)}},
			// expansion within a gap
			{"a\u0317\u0301", 5, ColElems{w(102), w(0, 220), w(0, 220)}},
			{"a\u302E\u18A9\u0301", 9, ColElems{w(102), w(0, 131), w(0, 132)}},
			{
				"a\u0317\u302E\u18A9\u0301",
				11,
				ColElems{w(102), w(0, 220), w(0, 220), w(0, 131), w(0, 132)},
			},
		},
	},
}

func TestAppendNext(t *testing.T) {
	for i, tt := range appendNextTests {
		c, err := makeTable(tt.in)
		if err != nil {
			t.Errorf("%d: error creating table: %v", i, err)
			continue
		}
		ct := collate.GetTable(c)
		for j, chk := range tt.chk {
			ws, n := ct.AppendNext([]byte(chk.in))
			if n != chk.n {
				t.Errorf("%d:%d: bytes consumed was %d; want %d", i, j, n, chk.n)
			}
			if len(ws) != len(chk.out) {
				t.Errorf("%d:%d: len(ws) was %d; want %d (%v vs %v)\n%X", i, j, len(ws), len(chk.out), ws, chk.out, chk.in)
				continue
			}
			for k, w := range ws {
				if w != chk.out[k] {
					t.Errorf("%d:%d: Weights %d was %v; want %v", i, j, k, w, chk.out[k])
				}
			}
		}
	}
}

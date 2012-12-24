// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import (
	"strings"
	"testing"
)

func doIterNorm(f Form, s string) []byte {
	acc := []byte{}
	i := Iter{}
	i.InitString(f, s)
	for !i.Done() {
		acc = append(acc, i.Next()...)
	}
	return acc
}

func runIterTests(t *testing.T, name string, f Form, tests []AppendTest, norm bool) {
	for i, test := range tests {
		in := test.left + test.right
		gold := test.out
		if norm {
			gold = string(f.AppendString(nil, test.out))
		}
		out := string(doIterNorm(f, in))
		if len(out) != len(gold) {
			const msg = "%s:%d: length is %d; want %d"
			t.Errorf(msg, name, i, len(out), len(gold))
		}
		if out != gold {
			// Find first rune that differs and show context.
			ir := []rune(out)
			ig := []rune(gold)
			t.Errorf("\n%X != \n%X", ir, ig)
			for j := 0; j < len(ir) && j < len(ig); j++ {
				if ir[j] == ig[j] {
					continue
				}
				if j -= 3; j < 0 {
					j = 0
				}
				for e := j + 7; j < e && j < len(ir) && j < len(ig); j++ {
					const msg = "%s:%d: runeAt(%d) = %U; want %U"
					t.Errorf(msg, name, i, j, ir[j], ig[j])
				}
				break
			}
		}
	}
}

func rep(r rune, n int) string {
	return strings.Repeat(string(r), n)
}

const segSize = maxByteBufferSize

var iterTests = []AppendTest{
	{"", ascii, ascii},
	{"", txt_all, txt_all},
	{"", "a" + rep(0x0300, segSize/2), "a" + rep(0x0300, segSize/2)},
}

var iterTestsD = []AppendTest{
	{ // segment overflow on unchanged character
		"",
		"a" + rep(0x0300, segSize/2) + "\u0316",
		"a" + rep(0x0300, segSize/2-1) + "\u0316\u0300",
	},
	{ // segment overflow on unchanged character + start value
		"",
		"a" + rep(0x0300, segSize/2+maxCombiningChars+4) + "\u0316",
		"a" + rep(0x0300, segSize/2+maxCombiningChars) + "\u0316" + rep(0x300, 4),
	},
	{ // segment overflow on decomposition
		"",
		"a" + rep(0x0300, segSize/2-1) + "\u0340",
		"a" + rep(0x0300, segSize/2),
	},
	{ // segment overflow on decomposition + start value
		"",
		"a" + rep(0x0300, segSize/2-1) + "\u0340" + rep(0x300, maxCombiningChars+4) + "\u0320",
		"a" + rep(0x0300, segSize/2-1) + rep(0x300, maxCombiningChars+1) + "\u0320" + rep(0x300, 4),
	},
	{ // start value after ASCII overflow
		"",
		rep('a', segSize) + rep(0x300, maxCombiningChars+2) + "\u0320",
		rep('a', segSize) + rep(0x300, maxCombiningChars) + "\u0320\u0300\u0300",
	},
	{ // start value after Hangul overflow
		"",
		rep(0xAC00, segSize/6) + rep(0x300, maxCombiningChars+2) + "\u0320",
		strings.Repeat("\u1100\u1161", segSize/6) + rep(0x300, maxCombiningChars+1) + "\u0320" + rep(0x300, 1),
	},
	{ // start value after cc=0
		"",
		"您您" + rep(0x300, maxCombiningChars+4) + "\u0320",
		"您您" + rep(0x300, maxCombiningChars) + "\u0320" + rep(0x300, 4),
	},
	{ // start value after normalization
		"",
		"\u0300\u0320a" + rep(0x300, maxCombiningChars+4) + "\u0320",
		"\u0320\u0300a" + rep(0x300, maxCombiningChars) + "\u0320" + rep(0x300, 4),
	},
}

var iterTestsC = []AppendTest{
	{ // ordering of non-composing combining characters
		"",
		"\u0305\u0316",
		"\u0316\u0305",
	},
	{ // segment overflow
		"",
		"a" + rep(0x0305, segSize/2+4) + "\u0316",
		"a" + rep(0x0305, segSize/2-1) + "\u0316" + rep(0x305, 5),
	},
}

func TestIterNextD(t *testing.T) {
	runIterTests(t, "IterNextD1", NFKD, appendTests, true)
	runIterTests(t, "IterNextD2", NFKD, iterTests, true)
	runIterTests(t, "IterNextD3", NFKD, iterTestsD, false)
}

func TestIterNextC(t *testing.T) {
	runIterTests(t, "IterNextC1", NFKC, appendTests, true)
	runIterTests(t, "IterNextC2", NFKC, iterTests, true)
	runIterTests(t, "IterNextC3", NFKC, iterTestsC, false)
}

type SegmentTest struct {
	in  string
	out []string
}

var segmentTests = []SegmentTest{
	{"\u1E0A\u0323a", []string{"\x44\u0323\u0307", "a", ""}},
	{rep('a', segSize), append(strings.Split(rep('a', segSize), ""), "")},
	{rep('a', segSize+2), append(strings.Split(rep('a', segSize+2), ""), "")},
	{rep('a', segSize) + "\u0300aa",
		append(strings.Split(rep('a', segSize-1), ""), "a\u0300", "a", "a", "")},
}

var segmentTestsK = []SegmentTest{
	{"\u3332", []string{"\u30D5", "\u30A1", "\u30E9", "\u30C3", "\u30C8\u3099", ""}},
	// last segment of multi-segment decomposition needs normalization
	{"\u3332\u093C", []string{"\u30D5", "\u30A1", "\u30E9", "\u30C3", "\u30C8\u093C\u3099", ""}},
	// Hangul and Jamo are grouped togeter.
	{"\uAC00", []string{"\u1100\u1161", ""}},
	{"\uAC01", []string{"\u1100\u1161\u11A8", ""}},
	{"\u1100\u1161", []string{"\u1100\u1161", ""}},
}

// Note that, by design, segmentation is equal for composing and decomposing forms.
func TestIterSegmentation(t *testing.T) {
	segmentTest(t, "SegmentTestD", NFD, segmentTests)
	segmentTest(t, "SegmentTestC", NFC, segmentTests)
	segmentTest(t, "SegmentTestD", NFKD, segmentTestsK)
	segmentTest(t, "SegmentTestC", NFKC, segmentTestsK)
}

func segmentTest(t *testing.T, name string, f Form, tests []SegmentTest) {
	iter := Iter{}
	for i, tt := range tests {
		iter.InitString(f, tt.in)
		for j, seg := range tt.out {
			if seg == "" {
				if !iter.Done() {
					res := string(iter.Next())
					t.Errorf(`%s:%d:%d: expected Done()==true, found segment "%s"`, name, i, j, res)
				}
				continue
			}
			if iter.Done() {
				t.Errorf("%s:%d:%d: Done()==true, want false", name, i, j)
			}
			seg = f.String(seg)
			if res := string(iter.Next()); res != seg {
				t.Errorf(`%s:%d:%d" segment was "%s" (%d); want "%s" (%d) %X %X`, name, i, j, res, len(res), seg, len(seg), []rune(res), []rune(seg))
			}
		}
	}
}

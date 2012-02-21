// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import (
	"strings"
	"testing"
)

var iterBufSizes = []int{
	MaxSegmentSize,
	1.5 * MaxSegmentSize,
	2 * MaxSegmentSize,
	3 * MaxSegmentSize,
	100 * MaxSegmentSize,
}

func doIterNorm(f Form, buf []byte, s string) []byte {
	acc := []byte{}
	i := Iter{}
	i.SetInputString(f, s)
	for !i.Done() {
		n := i.Next(buf)
		acc = append(acc, buf[:n]...)
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
		for _, sz := range iterBufSizes {
			buf := make([]byte, sz)
			out := string(doIterNorm(f, buf, in))
			if len(out) != len(gold) {
				const msg = "%s:%d:%d: length is %d; want %d"
				t.Errorf(msg, name, i, sz, len(out), len(gold))
			}
			if out != gold {
				// Find first rune that differs and show context.
				ir := []rune(out)
				ig := []rune(gold)
				for j := 0; j < len(ir) && j < len(ig); j++ {
					if ir[j] == ig[j] {
						continue
					}
					if j -= 3; j < 0 {
						j = 0
					}
					for e := j + 7; j < e && j < len(ir) && j < len(ig); j++ {
						const msg = "%s:%d:%d: runeAt(%d) = %U; want %U"
						t.Errorf(msg, name, i, sz, j, ir[j], ig[j])
					}
					break
				}
			}
		}
	}
}

func rep(r rune, n int) string {
	return strings.Repeat(string(r), n)
}

var iterTests = []AppendTest{
	{"", ascii, ascii},
	{"", txt_all, txt_all},
	{"", "a" + rep(0x0300, MaxSegmentSize/2), "a" + rep(0x0300, MaxSegmentSize/2)},
}

var iterTestsD = []AppendTest{
	{ // segment overflow on unchanged character
		"",
		"a" + rep(0x0300, MaxSegmentSize/2) + "\u0316",
		"a" + rep(0x0300, MaxSegmentSize/2-1) + "\u0316\u0300",
	},
	{ // segment overflow on unchanged character + start value
		"",
		"a" + rep(0x0300, MaxSegmentSize/2+maxCombiningChars+4) + "\u0316",
		"a" + rep(0x0300, MaxSegmentSize/2+maxCombiningChars) + "\u0316" + rep(0x300, 4),
	},
	{ // segment overflow on decomposition
		"",
		"a" + rep(0x0300, MaxSegmentSize/2-1) + "\u0340",
		"a" + rep(0x0300, MaxSegmentSize/2),
	},
	{ // segment overflow on decomposition + start value
		"",
		"a" + rep(0x0300, MaxSegmentSize/2-1) + "\u0340" + rep(0x300, maxCombiningChars+4) + "\u0320",
		"a" + rep(0x0300, MaxSegmentSize/2-1) + rep(0x300, maxCombiningChars+1) + "\u0320" + rep(0x300, 4),
	},
	{ // start value after ASCII overflow
		"",
		rep('a', MaxSegmentSize) + rep(0x300, maxCombiningChars+2) + "\u0320",
		rep('a', MaxSegmentSize) + rep(0x300, maxCombiningChars) + "\u0320\u0300\u0300",
	},
	{ // start value after Hangul overflow
		"",
		rep(0xAC00, MaxSegmentSize/6) + rep(0x300, maxCombiningChars+2) + "\u0320",
		strings.Repeat("\u1100\u1161", MaxSegmentSize/6) + rep(0x300, maxCombiningChars-1) + "\u0320" + rep(0x300, 3),
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
		"a" + rep(0x0305, MaxSegmentSize/2+4) + "\u0316",
		"a" + rep(0x0305, MaxSegmentSize/2-1) + "\u0316" + rep(0x305, 5),
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
	{rep('a', MaxSegmentSize), []string{rep('a', MaxSegmentSize), ""}},
	{rep('a', MaxSegmentSize+2), []string{rep('a', MaxSegmentSize-1), "aaa", ""}},
	{rep('a', MaxSegmentSize) + "\u0300aa", []string{rep('a', MaxSegmentSize-1), "a\u0300", "aa", ""}},
}

// Note that, by design, segmentation is equal for composing and decomposing forms.
func TestIterSegmentation(t *testing.T) {
	segmentTest(t, "SegmentTestD", NFD, segmentTests)
	segmentTest(t, "SegmentTestC", NFC, segmentTests)
}

func segmentTest(t *testing.T, name string, f Form, tests []SegmentTest) {
	iter := Iter{}
	for i, tt := range segmentTests {
		buf := make([]byte, MaxSegmentSize)
		iter.SetInputString(f, tt.in)
		for j, seg := range tt.out {
			if seg == "" {
				if !iter.Done() {
					n := iter.Next(buf)
					res := string(buf[:n])
					t.Errorf(`%s:%d:%d: expected Done()==true, found segment "%s"`, name, i, j, res)
				}
				continue
			}
			if iter.Done() {
				t.Errorf("%s:%d:%d: Done()==true, want false", name, i, j)
			}
			n := iter.Next(buf)
			seg = f.String(seg)
			if res := string(buf[:n]); res != seg {
				t.Errorf(`%s:%d:%d" segment was "%s" (%d); want "%s" (%d)`, name, i, j, res, len(res), seg, len(seg))
			}
		}
	}
}

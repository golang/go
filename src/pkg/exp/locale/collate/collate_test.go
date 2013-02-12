// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

import (
	"bytes"
	"exp/locale/collate/colltab"
	"testing"
)

type weightsTest struct {
	opt     opts
	in, out ColElems
}

type opts struct {
	lev int
	alt AlternateHandling
	top int

	backwards bool
	caseLevel bool
}

func (o opts) level() colltab.Level {
	if o.lev == 0 {
		return colltab.Quaternary
	}
	return colltab.Level(o.lev - 1)
}

func makeCE(w []int) colltab.Elem {
	ce, err := colltab.MakeElem(w[0], w[1], w[2], uint8(w[3]))
	if err != nil {
		panic(err)
	}
	return ce
}

func (o opts) collator() *Collator {
	c := &Collator{
		Strength:    o.level(),
		Alternate:   o.alt,
		Backwards:   o.backwards,
		CaseLevel:   o.caseLevel,
		variableTop: uint32(o.top),
	}
	return c
}

const (
	maxQ = 0x1FFFFF
)

func wpq(p, q int) Weights {
	return W(p, defaults.Secondary, defaults.Tertiary, q)
}

func wsq(s, q int) Weights {
	return W(0, s, defaults.Tertiary, q)
}

func wq(q int) Weights {
	return W(0, 0, 0, q)
}

var zero = W(0, 0, 0, 0)

var processTests = []weightsTest{
	// Shifted
	{ // simple sequence of non-variables
		opt: opts{alt: AltShifted, top: 100},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{wpq(200, maxQ), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // first is a variable
		opt: opts{alt: AltShifted, top: 250},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{wq(200), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // all but first are variable
		opt: opts{alt: AltShifted, top: 999},
		in:  ColElems{W(1000), W(200), W(300), W(400)},
		out: ColElems{wpq(1000, maxQ), wq(200), wq(300), wq(400)},
	},
	{ // first is a modifier
		opt: opts{alt: AltShifted, top: 999},
		in:  ColElems{W(0, 10), W(1000)},
		out: ColElems{wsq(10, maxQ), wpq(1000, maxQ)},
	},
	{ // primary ignorables
		opt: opts{alt: AltShifted, top: 250},
		in:  ColElems{W(200), W(0, 10), W(300), W(0, 15), W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), wsq(15, maxQ), wpq(400, maxQ)},
	},
	{ // secondary ignorables
		opt: opts{alt: AltShifted, top: 250},
		in:  ColElems{W(200), W(0, 0, 10), W(300), W(0, 0, 15), W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), W(0, 0, 15, maxQ), wpq(400, maxQ)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: AltShifted, top: 250},
		in:  ColElems{W(200), zero, W(300), zero, W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), zero, wpq(400, maxQ)},
	},

	// ShiftTrimmed (same as Shifted)
	{ // simple sequence of non-variables
		opt: opts{alt: AltShiftTrimmed, top: 100},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{wpq(200, maxQ), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // first is a variable
		opt: opts{alt: AltShiftTrimmed, top: 250},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{wq(200), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // all but first are variable
		opt: opts{alt: AltShiftTrimmed, top: 999},
		in:  ColElems{W(1000), W(200), W(300), W(400)},
		out: ColElems{wpq(1000, maxQ), wq(200), wq(300), wq(400)},
	},
	{ // first is a modifier
		opt: opts{alt: AltShiftTrimmed, top: 999},
		in:  ColElems{W(0, 10), W(1000)},
		out: ColElems{wsq(10, maxQ), wpq(1000, maxQ)},
	},
	{ // primary ignorables
		opt: opts{alt: AltShiftTrimmed, top: 250},
		in:  ColElems{W(200), W(0, 10), W(300), W(0, 15), W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), wsq(15, maxQ), wpq(400, maxQ)},
	},
	{ // secondary ignorables
		opt: opts{alt: AltShiftTrimmed, top: 250},
		in:  ColElems{W(200), W(0, 0, 10), W(300), W(0, 0, 15), W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), W(0, 0, 15, maxQ), wpq(400, maxQ)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: AltShiftTrimmed, top: 250},
		in:  ColElems{W(200), zero, W(300), zero, W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), zero, wpq(400, maxQ)},
	},

	// Blanked
	{ // simple sequence of non-variables
		opt: opts{alt: AltBlanked, top: 100},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{W(200), W(300), W(400)},
	},
	{ // first is a variable
		opt: opts{alt: AltBlanked, top: 250},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{zero, W(300), W(400)},
	},
	{ // all but first are variable
		opt: opts{alt: AltBlanked, top: 999},
		in:  ColElems{W(1000), W(200), W(300), W(400)},
		out: ColElems{W(1000), zero, zero, zero},
	},
	{ // first is a modifier
		opt: opts{alt: AltBlanked, top: 999},
		in:  ColElems{W(0, 10), W(1000)},
		out: ColElems{W(0, 10), W(1000)},
	},
	{ // primary ignorables
		opt: opts{alt: AltBlanked, top: 250},
		in:  ColElems{W(200), W(0, 10), W(300), W(0, 15), W(400)},
		out: ColElems{zero, zero, W(300), W(0, 15), W(400)},
	},
	{ // secondary ignorables
		opt: opts{alt: AltBlanked, top: 250},
		in:  ColElems{W(200), W(0, 0, 10), W(300), W(0, 0, 15), W(400)},
		out: ColElems{zero, zero, W(300), W(0, 0, 15), W(400)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: AltBlanked, top: 250},
		in:  ColElems{W(200), zero, W(300), zero, W(400)},
		out: ColElems{zero, zero, W(300), zero, W(400)},
	},

	// Non-ignorable: input is always equal to output.
	{ // all but first are variable
		opt: opts{alt: AltNonIgnorable, top: 999},
		in:  ColElems{W(1000), W(200), W(300), W(400)},
		out: ColElems{W(1000), W(200), W(300), W(400)},
	},
	{ // primary ignorables
		opt: opts{alt: AltNonIgnorable, top: 250},
		in:  ColElems{W(200), W(0, 10), W(300), W(0, 15), W(400)},
		out: ColElems{W(200), W(0, 10), W(300), W(0, 15), W(400)},
	},
	{ // secondary ignorables
		opt: opts{alt: AltNonIgnorable, top: 250},
		in:  ColElems{W(200), W(0, 0, 10), W(300), W(0, 0, 15), W(400)},
		out: ColElems{W(200), W(0, 0, 10), W(300), W(0, 0, 15), W(400)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: AltNonIgnorable, top: 250},
		in:  ColElems{W(200), zero, W(300), zero, W(400)},
		out: ColElems{W(200), zero, W(300), zero, W(400)},
	},
}

func TestProcessWeights(t *testing.T) {
	for i, tt := range processTests {
		in := convertFromWeights(tt.in)
		out := convertFromWeights(tt.out)
		processWeights(tt.opt.alt, uint32(tt.opt.top), in)
		for j, w := range in {
			if w != out[j] {
				t.Errorf("%d: Weights %d was %v; want %v %X %X", i, j, w, out[j])
			}
		}
	}
}

type keyFromElemTest struct {
	opt opts
	in  ColElems
	out []byte
}

var defS = byte(defaults.Secondary)
var defT = byte(defaults.Tertiary)

const sep = 0 // separator byte

var keyFromElemTests = []keyFromElemTest{
	{ // simple primary and secondary weights.
		opts{alt: AltShifted},
		ColElems{W(0x200), W(0x7FFF), W(0, 0x30), W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, but with zero element that need to be removed
		opts{alt: AltShifted},
		ColElems{W(0x200), zero, W(0x7FFF), W(0, 0x30), zero, W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, with large primary values
		opts{alt: AltShifted},
		ColElems{W(0x200), W(0x8000), W(0, 0x30), W(0x12345)},
		[]byte{0x2, 0, 0x80, 0x80, 0x00, 0x81, 0x23, 0x45, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, but with the secondary level backwards
		opts{alt: AltShifted, backwards: true},
		ColElems{W(0x200), W(0x7FFF), W(0, 0x30), W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, 0x30, 0, defS, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, ignoring quaternary level
		opts{alt: AltShifted, lev: 3},
		ColElems{W(0x200), zero, W(0x7FFF), W(0, 0x30), zero, W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
		},
	},
	{ // same as first, ignoring tertiary level
		opts{alt: AltShifted, lev: 2},
		ColElems{W(0x200), zero, W(0x7FFF), W(0, 0x30), zero, W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
		},
	},
	{ // same as first, ignoring secondary level
		opts{alt: AltShifted, lev: 1},
		ColElems{W(0x200), zero, W(0x7FFF), W(0, 0x30), zero, W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00},
	},
	{ // simple primary and secondary weights.
		opts{alt: AltShiftTrimmed, top: 0x250},
		ColElems{W(0x300), W(0x200), W(0x7FFF), W(0, 0x30), W(0x800)},
		[]byte{0x3, 0, 0x7F, 0xFF, 0x8, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0x2, 0, // quaternary
		},
	},
	{ // as first, primary with case level enabled
		opts{alt: AltShifted, lev: 1, caseLevel: true},
		ColElems{W(0x200), W(0x7FFF), W(0, 0x30), W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
		},
	},
}

func TestKeyFromElems(t *testing.T) {
	buf := Buffer{}
	for i, tt := range keyFromElemTests {
		buf.Reset()
		in := convertFromWeights(tt.in)
		processWeights(tt.opt.alt, uint32(tt.opt.top), in)
		tt.opt.collator().keyFromElems(&buf, in)
		res := buf.key
		if len(res) != len(tt.out) {
			t.Errorf("%d: len(ws) was %d; want %d (%X should be %X)", i, len(res), len(tt.out), res, tt.out)
		}
		n := len(res)
		if len(tt.out) < n {
			n = len(tt.out)
		}
		for j, c := range res[:n] {
			if c != tt.out[j] {
				t.Errorf("%d: byte %d was %X; want %X", i, j, c, tt.out[j])
			}
		}
	}
}

func TestGetColElems(t *testing.T) {
	for i, tt := range appendNextTests {
		c, err := makeTable(tt.in)
		if err != nil {
			// error is reported in TestAppendNext
			continue
		}
		// Create one large test per table
		str := make([]byte, 0, 4000)
		out := ColElems{}
		for len(str) < 3000 {
			for _, chk := range tt.chk {
				str = append(str, chk.in[:chk.n]...)
				out = append(out, chk.out...)
			}
		}
		for j, chk := range append(tt.chk, check{string(str), len(str), out}) {
			out := convertFromWeights(chk.out)
			ce := c.getColElems([]byte(chk.in)[:chk.n])
			if len(ce) != len(out) {
				t.Errorf("%d:%d: len(ws) was %d; want %d", i, j, len(ce), len(out))
				continue
			}
			cnt := 0
			for k, w := range ce {
				w, _ = colltab.MakeElem(w.Primary(), w.Secondary(), int(w.Tertiary()), 0)
				if w != out[k] {
					t.Errorf("%d:%d: Weights %d was %X; want %X", i, j, k, w, out[k])
					cnt++
				}
				if cnt > 10 {
					break
				}
			}
		}
	}
}

type keyTest struct {
	in  string
	out []byte
}

var keyTests = []keyTest{
	{"abc",
		[]byte{0, 100, 0, 200, 1, 44, 0, 0, 0, 32, 0, 32, 0, 32, 0, 0, 2, 2, 2, 0, 255, 255, 255},
	},
	{"a\u0301",
		[]byte{0, 102, 0, 0, 0, 32, 0, 0, 2, 0, 255},
	},
	{"aaaaa",
		[]byte{0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 0,
			0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 0,
			2, 2, 2, 2, 2, 0,
			255, 255, 255, 255, 255,
		},
	},
}

func TestKey(t *testing.T) {
	c, _ := makeTable(appendNextTests[4].in)
	c.Alternate = AltShifted
	c.Strength = colltab.Quaternary
	buf := Buffer{}
	keys1 := [][]byte{}
	keys2 := [][]byte{}
	for _, tt := range keyTests {
		keys1 = append(keys1, c.Key(&buf, []byte(tt.in)))
		keys2 = append(keys2, c.KeyFromString(&buf, tt.in))
	}
	// Separate generation from testing to ensure buffers are not overwritten.
	for i, tt := range keyTests {
		if !bytes.Equal(keys1[i], tt.out) {
			t.Errorf("%d: Key(%q) = %d; want %d", i, tt.in, keys1[i], tt.out)
		}
		if !bytes.Equal(keys2[i], tt.out) {
			t.Errorf("%d: KeyFromString(%q) = %d; want %d", i, tt.in, keys2[i], tt.out)
		}
	}
}

type compareTest struct {
	a, b string
	res  int // comparison result
}

var compareTests = []compareTest{
	{"a\u0301", "a", 1},
	{"a\u0301b", "ab", 1},
	{"a", "a\u0301", -1},
	{"ab", "a\u0301b", -1},
	{"bc", "a\u0301c", 1},
	{"ab", "aB", -1},
	{"a\u0301", "a\u0301", 0},
	{"a", "a", 0},
	// Only clip prefixes of whole runes.
	{"\u302E", "\u302F", 1},
	// Don't clip prefixes when last rune of prefix may be part of contraction.
	{"a\u035E", "a\u0301\u035F", -1},
	{"a\u0301\u035Fb", "a\u0301\u035F", -1},
}

func TestCompare(t *testing.T) {
	c, _ := makeTable(appendNextTests[4].in)
	for i, tt := range compareTests {
		if res := c.Compare([]byte(tt.a), []byte(tt.b)); res != tt.res {
			t.Errorf("%d: Compare(%q, %q) == %d; want %d", i, tt.a, tt.b, res, tt.res)
		}
		if res := c.CompareString(tt.a, tt.b); res != tt.res {
			t.Errorf("%d: CompareString(%q, %q) == %d; want %d", i, tt.a, tt.b, res, tt.res)
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
			i.ce = append(i.ce, makeCE([]int{w, defaultSecondary, 2, cc}))
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
			i.ce = append(i.ce, makeCE([]int{100, defaultSecondary, 2, 8}))
		}
		p := len(i.ce)
		i.ce = append(i.ce, makeCE([]int{0, defaultSecondary, 2, 5}))
		i.doNorm(p, 5)
		if i.prevCCC != result[o+2] {
			t.Errorf("%d: i.prevCCC was %d; want %d", n, i.prevCCC, result[o+2])
		}
		if result[o+2] == 5 && i.pStarter != p {
			t.Errorf("%d: i.pStarter was %d; want %d", n, i.pStarter, p)
		}
	}
}

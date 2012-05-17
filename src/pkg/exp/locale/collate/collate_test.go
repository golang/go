// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate_test

import (
	"bytes"
	"exp/locale/collate"
	"testing"
)

type weightsTest struct {
	opt     opts
	in, out ColElems
}

type opts struct {
	lev int
	alt collate.AlternateHandling
	top int

	backwards bool
	caseLevel bool
}

func (o opts) level() collate.Level {
	if o.lev == 0 {
		return collate.Quaternary
	}
	return collate.Level(o.lev - 1)
}

func (o opts) collator() *collate.Collator {
	c := &collate.Collator{
		Strength:  o.level(),
		Alternate: o.alt,
		Backwards: o.backwards,
		CaseLevel: o.caseLevel,
	}
	collate.SetTop(c, o.top)
	return c
}

const (
	maxQ = 0x1FFFFF
)

func wpq(p, q int) collate.Weights {
	return collate.W(p, defaults.Secondary, defaults.Tertiary, q)
}

func wsq(s, q int) collate.Weights {
	return collate.W(0, s, defaults.Tertiary, q)
}

func wq(q int) collate.Weights {
	return collate.W(0, 0, 0, q)
}

var zero = w(0, 0, 0, 0)

var processTests = []weightsTest{
	// Shifted
	{ // simple sequence of non-variables
		opt: opts{alt: collate.AltShifted, top: 100},
		in:  ColElems{w(200), w(300), w(400)},
		out: ColElems{wpq(200, maxQ), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // first is a variable
		opt: opts{alt: collate.AltShifted, top: 250},
		in:  ColElems{w(200), w(300), w(400)},
		out: ColElems{wq(200), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // all but first are variable
		opt: opts{alt: collate.AltShifted, top: 999},
		in:  ColElems{w(1000), w(200), w(300), w(400)},
		out: ColElems{wpq(1000, maxQ), wq(200), wq(300), wq(400)},
	},
	{ // first is a modifier
		opt: opts{alt: collate.AltShifted, top: 999},
		in:  ColElems{w(0, 10), w(1000)},
		out: ColElems{wsq(10, maxQ), wpq(1000, maxQ)},
	},
	{ // primary ignorables
		opt: opts{alt: collate.AltShifted, top: 250},
		in:  ColElems{w(200), w(0, 10), w(300), w(0, 15), w(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), wsq(15, maxQ), wpq(400, maxQ)},
	},
	{ // secondary ignorables
		opt: opts{alt: collate.AltShifted, top: 250},
		in:  ColElems{w(200), w(0, 0, 10), w(300), w(0, 0, 15), w(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), w(0, 0, 15, maxQ), wpq(400, maxQ)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: collate.AltShifted, top: 250},
		in:  ColElems{w(200), zero, w(300), zero, w(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), zero, wpq(400, maxQ)},
	},

	// ShiftTrimmed (same as Shifted)
	{ // simple sequence of non-variables
		opt: opts{alt: collate.AltShiftTrimmed, top: 100},
		in:  ColElems{w(200), w(300), w(400)},
		out: ColElems{wpq(200, maxQ), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // first is a variable
		opt: opts{alt: collate.AltShiftTrimmed, top: 250},
		in:  ColElems{w(200), w(300), w(400)},
		out: ColElems{wq(200), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // all but first are variable
		opt: opts{alt: collate.AltShiftTrimmed, top: 999},
		in:  ColElems{w(1000), w(200), w(300), w(400)},
		out: ColElems{wpq(1000, maxQ), wq(200), wq(300), wq(400)},
	},
	{ // first is a modifier
		opt: opts{alt: collate.AltShiftTrimmed, top: 999},
		in:  ColElems{w(0, 10), w(1000)},
		out: ColElems{wsq(10, maxQ), wpq(1000, maxQ)},
	},
	{ // primary ignorables
		opt: opts{alt: collate.AltShiftTrimmed, top: 250},
		in:  ColElems{w(200), w(0, 10), w(300), w(0, 15), w(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), wsq(15, maxQ), wpq(400, maxQ)},
	},
	{ // secondary ignorables
		opt: opts{alt: collate.AltShiftTrimmed, top: 250},
		in:  ColElems{w(200), w(0, 0, 10), w(300), w(0, 0, 15), w(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), w(0, 0, 15, maxQ), wpq(400, maxQ)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: collate.AltShiftTrimmed, top: 250},
		in:  ColElems{w(200), zero, w(300), zero, w(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), zero, wpq(400, maxQ)},
	},

	// Blanked
	{ // simple sequence of non-variables
		opt: opts{alt: collate.AltBlanked, top: 100},
		in:  ColElems{w(200), w(300), w(400)},
		out: ColElems{w(200), w(300), w(400)},
	},
	{ // first is a variable
		opt: opts{alt: collate.AltBlanked, top: 250},
		in:  ColElems{w(200), w(300), w(400)},
		out: ColElems{zero, w(300), w(400)},
	},
	{ // all but first are variable
		opt: opts{alt: collate.AltBlanked, top: 999},
		in:  ColElems{w(1000), w(200), w(300), w(400)},
		out: ColElems{w(1000), zero, zero, zero},
	},
	{ // first is a modifier
		opt: opts{alt: collate.AltBlanked, top: 999},
		in:  ColElems{w(0, 10), w(1000)},
		out: ColElems{w(0, 10), w(1000)},
	},
	{ // primary ignorables
		opt: opts{alt: collate.AltBlanked, top: 250},
		in:  ColElems{w(200), w(0, 10), w(300), w(0, 15), w(400)},
		out: ColElems{zero, zero, w(300), w(0, 15), w(400)},
	},
	{ // secondary ignorables
		opt: opts{alt: collate.AltBlanked, top: 250},
		in:  ColElems{w(200), w(0, 0, 10), w(300), w(0, 0, 15), w(400)},
		out: ColElems{zero, zero, w(300), w(0, 0, 15), w(400)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: collate.AltBlanked, top: 250},
		in:  ColElems{w(200), zero, w(300), zero, w(400)},
		out: ColElems{zero, zero, w(300), zero, w(400)},
	},

	// Non-ignorable: input is always equal to output.
	{ // all but first are variable
		opt: opts{alt: collate.AltNonIgnorable, top: 999},
		in:  ColElems{w(1000), w(200), w(300), w(400)},
		out: ColElems{w(1000), w(200), w(300), w(400)},
	},
	{ // primary ignorables
		opt: opts{alt: collate.AltNonIgnorable, top: 250},
		in:  ColElems{w(200), w(0, 10), w(300), w(0, 15), w(400)},
		out: ColElems{w(200), w(0, 10), w(300), w(0, 15), w(400)},
	},
	{ // secondary ignorables
		opt: opts{alt: collate.AltNonIgnorable, top: 250},
		in:  ColElems{w(200), w(0, 0, 10), w(300), w(0, 0, 15), w(400)},
		out: ColElems{w(200), w(0, 0, 10), w(300), w(0, 0, 15), w(400)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: collate.AltNonIgnorable, top: 250},
		in:  ColElems{w(200), zero, w(300), zero, w(400)},
		out: ColElems{w(200), zero, w(300), zero, w(400)},
	},
}

func TestProcessWeights(t *testing.T) {
	for i, tt := range processTests {
		res := collate.ProcessWeights(tt.opt.alt, tt.opt.top, tt.in)
		if len(res) != len(tt.out) {
			t.Errorf("%d: len(ws) was %d; want %d (%v should be %v)", i, len(res), len(tt.out), res, tt.out)
			continue
		}
		for j, w := range res {
			if w != tt.out[j] {
				t.Errorf("%d: Weights %d was %v; want %v", i, j, w, tt.out[j])
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
		opts{},
		ColElems{w(0x200), w(0x7FFF), w(0, 0x30), w(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, but with zero element that need to be removed
		opts{},
		ColElems{w(0x200), zero, w(0x7FFF), w(0, 0x30), zero, w(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, with large primary values
		opts{},
		ColElems{w(0x200), w(0x8000), w(0, 0x30), w(0x12345)},
		[]byte{0x2, 0, 0x80, 0x80, 0x00, 0x81, 0x23, 0x45, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, but with the secondary level backwards
		opts{backwards: true},
		ColElems{w(0x200), w(0x7FFF), w(0, 0x30), w(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, 0x30, 0, defS, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, ignoring quaternary level
		opts{lev: 3},
		ColElems{w(0x200), zero, w(0x7FFF), w(0, 0x30), zero, w(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
		},
	},
	{ // same as first, ignoring tertiary level
		opts{lev: 2},
		ColElems{w(0x200), zero, w(0x7FFF), w(0, 0x30), zero, w(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
		},
	},
	{ // same as first, ignoring secondary level
		opts{lev: 1},
		ColElems{w(0x200), zero, w(0x7FFF), w(0, 0x30), zero, w(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00},
	},
	{ // simple primary and secondary weights.
		opts{alt: collate.AltShiftTrimmed, top: 0x250},
		ColElems{w(0x300), w(0x200), w(0x7FFF), w(0, 0x30), w(0x800)},
		[]byte{0x3, 0, 0x7F, 0xFF, 0x8, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0x2, 0, // quaternary
		},
	},
	{ // as first, primary with case level enabled
		opts{lev: 1, caseLevel: true},
		ColElems{w(0x200), w(0x7FFF), w(0, 0x30), w(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
		},
	},
}

func TestKeyFromElems(t *testing.T) {
	buf := collate.Buffer{}
	for i, tt := range keyFromElemTests {
		buf.ResetKeys()
		ws := collate.ProcessWeights(tt.opt.alt, tt.opt.top, tt.in)
		res := collate.KeyFromElems(tt.opt.collator(), &buf, ws)
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
		buf := collate.Buffer{}
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
			ws := collate.GetColElems(c, &buf, []byte(chk.in)[:chk.n])
			if len(ws) != len(chk.out) {
				t.Errorf("%d:%d: len(ws) was %d; want %d", i, j, len(ws), len(chk.out))
				continue
			}
			cnt := 0
			for k, w := range ws {
				if w != chk.out[k] {
					t.Errorf("%d:%d: Weights %d was %v; want %v", i, j, k, w, chk.out[k])
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
	buf := collate.Buffer{}
	keys1 := [][]byte{}
	keys2 := [][]byte{}
	for _, tt := range keyTests {
		keys1 = append(keys1, c.Key(&buf, []byte(tt.in)))
		keys2 = append(keys2, c.KeyFromString(&buf, tt.in))
	}
	// Separate generation from testing to ensure buffers are not overwritten.
	for i, tt := range keyTests {
		if bytes.Compare(keys1[i], tt.out) != 0 {
			t.Errorf("%d: Key(%q) = %d; want %d", i, tt.in, keys1[i], tt.out)
		}
		if bytes.Compare(keys2[i], tt.out) != 0 {
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
	{"a", "a\u0301", -1},
	{"a\u0301", "a\u0301", 0},
	{"a", "a", 0},
}

func TestCompare(t *testing.T) {
	c, _ := makeTable(appendNextTests[4].in)
	buf := collate.Buffer{}
	for i, tt := range compareTests {
		if res := c.Compare(&buf, []byte(tt.a), []byte(tt.b)); res != tt.res {
			t.Errorf("%d: Compare(%q, %q) == %d; want %d", i, tt.a, tt.b, res, tt.res)
		}
		if res := c.CompareString(&buf, tt.a, tt.b); res != tt.res {
			t.Errorf("%d: CompareString(%q, %q) == %d; want %d", i, tt.a, tt.b, res, tt.res)
		}
	}
}

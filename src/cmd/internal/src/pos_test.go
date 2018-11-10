// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package src

import (
	"fmt"
	"testing"
)

func TestPos(t *testing.T) {
	f0 := NewFileBase("", "")
	f1 := NewFileBase("f1", "f1")
	f2 := NewLinePragmaBase(Pos{}, "f2", 10)
	f3 := NewLinePragmaBase(MakePos(f1, 10, 1), "f3", 100)
	f4 := NewLinePragmaBase(MakePos(f3, 10, 1), "f4", 100)

	// line directives from issue #19392
	fp := NewFileBase("p.go", "p.go")
	fc := NewLinePragmaBase(MakePos(fp, 3, 0), "c.go", 10)
	ft := NewLinePragmaBase(MakePos(fp, 6, 0), "t.go", 20)
	fv := NewLinePragmaBase(MakePos(fp, 9, 0), "v.go", 30)
	ff := NewLinePragmaBase(MakePos(fp, 12, 0), "f.go", 40)

	for _, test := range []struct {
		pos    Pos
		string string

		// absolute info
		filename  string
		line, col uint

		// relative info
		relFilename string
		relLine     uint
	}{
		{Pos{}, "<unknown line number>", "", 0, 0, "", 0},
		{MakePos(nil, 2, 3), ":2:3", "", 2, 3, "", 2},
		{MakePos(f0, 2, 3), ":2:3", "", 2, 3, "", 2},
		{MakePos(f1, 1, 1), "f1:1:1", "f1", 1, 1, "f1", 1},
		{MakePos(f2, 7, 10), "f2:16[:7:10]", "", 7, 10, "f2", 16},
		{MakePos(f3, 12, 7), "f3:101[f1:12:7]", "f1", 12, 7, "f3", 101},
		{MakePos(f4, 25, 1), "f4:114[f3:25:1]", "f3", 25, 1, "f4", 114},

		// positions from issue #19392
		{MakePos(fc, 4, 0), "c.go:10[p.go:4:0]", "p.go", 4, 0, "c.go", 10},
		{MakePos(ft, 7, 0), "t.go:20[p.go:7:0]", "p.go", 7, 0, "t.go", 20},
		{MakePos(fv, 10, 0), "v.go:30[p.go:10:0]", "p.go", 10, 0, "v.go", 30},
		{MakePos(ff, 13, 0), "f.go:40[p.go:13:0]", "p.go", 13, 0, "f.go", 40},
	} {
		pos := test.pos
		if got := pos.String(); got != test.string {
			t.Errorf("%s: got %q", test.string, got)
		}

		// absolute info
		if got := pos.Filename(); got != test.filename {
			t.Errorf("%s: got filename %q; want %q", test.string, got, test.filename)
		}
		if got := pos.Line(); got != test.line {
			t.Errorf("%s: got line %d; want %d", test.string, got, test.line)
		}
		if got := pos.Col(); got != test.col {
			t.Errorf("%s: got col %d; want %d", test.string, got, test.col)
		}

		// relative info
		if got := pos.RelFilename(); got != test.relFilename {
			t.Errorf("%s: got relFilename %q; want %q", test.string, got, test.relFilename)
		}
		if got := pos.RelLine(); got != test.relLine {
			t.Errorf("%s: got relLine %d; want %d", test.string, got, test.relLine)
		}
	}
}

func TestPredicates(t *testing.T) {
	b1 := NewFileBase("b1", "b1")
	b2 := NewFileBase("b2", "b2")
	for _, test := range []struct {
		p, q                 Pos
		known, before, after bool
	}{
		{NoPos, NoPos, false, false, false},
		{NoPos, MakePos(nil, 1, 0), false, true, false},
		{MakePos(b1, 0, 0), NoPos, true, false, true},
		{MakePos(nil, 1, 0), NoPos, true, false, true},

		{MakePos(nil, 1, 1), MakePos(nil, 1, 1), true, false, false},
		{MakePos(nil, 1, 1), MakePos(nil, 1, 2), true, true, false},
		{MakePos(nil, 1, 2), MakePos(nil, 1, 1), true, false, true},
		{MakePos(nil, 123, 1), MakePos(nil, 1, 123), true, false, true},

		{MakePos(b1, 1, 1), MakePos(b1, 1, 1), true, false, false},
		{MakePos(b1, 1, 1), MakePos(b1, 1, 2), true, true, false},
		{MakePos(b1, 1, 2), MakePos(b1, 1, 1), true, false, true},
		{MakePos(b1, 123, 1), MakePos(b1, 1, 123), true, false, true},

		{MakePos(b1, 1, 1), MakePos(b2, 1, 1), true, true, false},
		{MakePos(b1, 1, 1), MakePos(b2, 1, 2), true, true, false},
		{MakePos(b1, 1, 2), MakePos(b2, 1, 1), true, true, false},
		{MakePos(b1, 123, 1), MakePos(b2, 1, 123), true, true, false},

		// special case: unknown column (column too large to represent)
		{MakePos(nil, 1, colMax+10), MakePos(nil, 1, colMax+20), true, false, false},
	} {
		if got := test.p.IsKnown(); got != test.known {
			t.Errorf("%s known: got %v; want %v", test.p, got, test.known)
		}
		if got := test.p.Before(test.q); got != test.before {
			t.Errorf("%s < %s: got %v; want %v", test.p, test.q, got, test.before)
		}
		if got := test.p.After(test.q); got != test.after {
			t.Errorf("%s > %s: got %v; want %v", test.p, test.q, got, test.after)
		}
	}
}

func TestLico(t *testing.T) {
	for _, test := range []struct {
		x         lico
		string    string
		line, col uint
	}{
		{0, ":0:0", 0, 0},
		{makeLico(0, 0), ":0:0", 0, 0},
		{makeLico(0, 1), ":0:1", 0, 1},
		{makeLico(1, 0), ":1:0", 1, 0},
		{makeLico(1, 1), ":1:1", 1, 1},
		{makeLico(2, 3), ":2:3", 2, 3},
		{makeLico(lineMax, 1), fmt.Sprintf(":%d:1", lineMax), lineMax, 1},
		{makeLico(lineMax+1, 1), fmt.Sprintf(":%d:1", lineMax), lineMax, 1}, // line too large, stick with max. line
		{makeLico(1, colMax), ":1", 1, colMax},
		{makeLico(1, colMax+1), ":1", 1, 0}, // column too large
		{makeLico(lineMax+1, colMax+1), fmt.Sprintf(":%d", lineMax), lineMax, 0},
	} {
		x := test.x
		if got := format("", x.Line(), x.Col(), true); got != test.string {
			t.Errorf("%s: got %q", test.string, got)
		}
	}
}

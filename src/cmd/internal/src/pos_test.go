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
	f2 := NewLinePragmaBase(Pos{}, "f2", "f2", 10, 0)
	f3 := NewLinePragmaBase(MakePos(f1, 10, 1), "f3", "f3", 100, 1)
	f4 := NewLinePragmaBase(MakePos(f3, 10, 1), "f4", "f4", 100, 1)

	// line directives with non-1 columns
	f5 := NewLinePragmaBase(MakePos(f1, 5, 5), "f5", "f5", 10, 1)

	// line directives from issue #19392
	fp := NewFileBase("p.go", "p.go")
	fc := NewLinePragmaBase(MakePos(fp, 4, 1), "c.go", "c.go", 10, 1)
	ft := NewLinePragmaBase(MakePos(fp, 7, 1), "t.go", "t.go", 20, 1)
	fv := NewLinePragmaBase(MakePos(fp, 10, 1), "v.go", "v.go", 30, 1)
	ff := NewLinePragmaBase(MakePos(fp, 13, 1), "f.go", "f.go", 40, 1)

	for _, test := range []struct {
		pos    Pos
		string string

		// absolute info
		filename  string
		line, col uint

		// relative info
		relFilename     string
		relLine, relCol uint
	}{
		{Pos{}, "<unknown line number>", "", 0, 0, "", 0, 0},
		{MakePos(nil, 2, 3), ":2:3", "", 2, 3, "", 0, 0},
		{MakePos(f0, 2, 3), ":2:3", "", 2, 3, "", 2, 3},
		{MakePos(f1, 1, 1), "f1:1:1", "f1", 1, 1, "f1", 1, 1},
		{MakePos(f2, 7, 10), "f2:17[:7:10]", "", 7, 10, "f2", 17, 0 /* line base doesn't specify a column */},
		{MakePos(f3, 12, 7), "f3:102:7[f1:12:7]", "f1", 12, 7, "f3", 102, 7},
		{MakePos(f4, 25, 1), "f4:115:1[f3:25:1]", "f3", 25, 1, "f4", 115, 1},

		// line directives with non-1 columns
		{MakePos(f5, 5, 5), "f5:10:1[f1:5:5]", "f1", 5, 5, "f5", 10, 1},
		{MakePos(f5, 5, 10), "f5:10:6[f1:5:10]", "f1", 5, 10, "f5", 10, 6},
		{MakePos(f5, 6, 10), "f5:11:10[f1:6:10]", "f1", 6, 10, "f5", 11, 10},

		// positions from issue #19392
		{MakePos(fc, 4, 1), "c.go:10:1[p.go:4:1]", "p.go", 4, 1, "c.go", 10, 1},
		{MakePos(ft, 7, 1), "t.go:20:1[p.go:7:1]", "p.go", 7, 1, "t.go", 20, 1},
		{MakePos(fv, 10, 1), "v.go:30:1[p.go:10:1]", "p.go", 10, 1, "v.go", 30, 1},
		{MakePos(ff, 13, 1), "f.go:40:1[p.go:13:1]", "p.go", 13, 1, "f.go", 40, 1},
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
		if got := pos.RelCol(); got != test.relCol {
			t.Errorf("%s: got relCol %d; want %d", test.string, got, test.relCol)
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
		{0, ":0", 0, 0},
		{makeLico(0, 0), ":0", 0, 0},
		{makeLico(0, 1), ":0:1", 0, 1},
		{makeLico(1, 0), ":1", 1, 0},
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

func TestIsStmt(t *testing.T) {
	def := fmt.Sprintf(":%d", PosDefaultStmt)
	is := fmt.Sprintf(":%d", PosIsStmt)
	not := fmt.Sprintf(":%d", PosNotStmt)

	for _, test := range []struct {
		x         lico
		string    string
		line, col uint
	}{
		{0, ":0" + not, 0, 0},
		{makeLico(0, 0), ":0" + not, 0, 0},
		{makeLico(0, 1), ":0:1" + def, 0, 1},
		{makeLico(1, 0), ":1" + def, 1, 0},
		{makeLico(1, 1), ":1:1" + def, 1, 1},
		{makeLico(1, 1).withIsStmt(), ":1:1" + is, 1, 1},
		{makeLico(1, 1).withNotStmt(), ":1:1" + not, 1, 1},
		{makeLico(lineMax, 1), fmt.Sprintf(":%d:1", lineMax) + def, lineMax, 1},
		{makeLico(lineMax+1, 1), fmt.Sprintf(":%d:1", lineMax) + def, lineMax, 1}, // line too large, stick with max. line
		{makeLico(1, colMax), ":1" + def, 1, colMax},
		{makeLico(1, colMax+1), ":1" + def, 1, 0}, // column too large
		{makeLico(lineMax+1, colMax+1), fmt.Sprintf(":%d", lineMax) + def, lineMax, 0},
		{makeLico(lineMax+1, colMax+1).withIsStmt(), fmt.Sprintf(":%d", lineMax) + is, lineMax, 0},
		{makeLico(lineMax+1, colMax+1).withNotStmt(), fmt.Sprintf(":%d", lineMax) + not, lineMax, 0},
	} {
		x := test.x
		if got := format("", x.Line(), x.Col(), true) + fmt.Sprintf(":%d", x.IsStmt()); got != test.string {
			t.Errorf("%s: got %q", test.string, got)
		}
	}
}

func TestLogue(t *testing.T) {
	defp := fmt.Sprintf(":%d", PosDefaultLogue)
	pro := fmt.Sprintf(":%d", PosPrologueEnd)
	epi := fmt.Sprintf(":%d", PosEpilogueBegin)

	defs := fmt.Sprintf(":%d", PosDefaultStmt)
	not := fmt.Sprintf(":%d", PosNotStmt)

	for i, test := range []struct {
		x         lico
		string    string
		line, col uint
	}{
		{makeLico(0, 0).withXlogue(PosDefaultLogue), ":0" + not + defp, 0, 0},
		{makeLico(0, 0).withXlogue(PosPrologueEnd), ":0" + not + pro, 0, 0},
		{makeLico(0, 0).withXlogue(PosEpilogueBegin), ":0" + not + epi, 0, 0},

		{makeLico(0, 1).withXlogue(PosDefaultLogue), ":0:1" + defs + defp, 0, 1},
		{makeLico(0, 1).withXlogue(PosPrologueEnd), ":0:1" + defs + pro, 0, 1},
		{makeLico(0, 1).withXlogue(PosEpilogueBegin), ":0:1" + defs + epi, 0, 1},

		{makeLico(1, 0).withXlogue(PosDefaultLogue), ":1" + defs + defp, 1, 0},
		{makeLico(1, 0).withXlogue(PosPrologueEnd), ":1" + defs + pro, 1, 0},
		{makeLico(1, 0).withXlogue(PosEpilogueBegin), ":1" + defs + epi, 1, 0},

		{makeLico(1, 1).withXlogue(PosDefaultLogue), ":1:1" + defs + defp, 1, 1},
		{makeLico(1, 1).withXlogue(PosPrologueEnd), ":1:1" + defs + pro, 1, 1},
		{makeLico(1, 1).withXlogue(PosEpilogueBegin), ":1:1" + defs + epi, 1, 1},

		{makeLico(lineMax, 1).withXlogue(PosDefaultLogue), fmt.Sprintf(":%d:1", lineMax) + defs + defp, lineMax, 1},
		{makeLico(lineMax, 1).withXlogue(PosPrologueEnd), fmt.Sprintf(":%d:1", lineMax) + defs + pro, lineMax, 1},
		{makeLico(lineMax, 1).withXlogue(PosEpilogueBegin), fmt.Sprintf(":%d:1", lineMax) + defs + epi, lineMax, 1},
	} {
		x := test.x
		if got := format("", x.Line(), x.Col(), true) + fmt.Sprintf(":%d:%d", x.IsStmt(), x.Xlogue()); got != test.string {
			t.Errorf("%d: %s: got %q", i, test.string, got)
		}
	}
}

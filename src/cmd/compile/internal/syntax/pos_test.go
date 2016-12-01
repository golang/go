// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"testing"
)

func TestPos(t *testing.T) {
	f0 := NewFileBase("")
	f1 := NewFileBase("f1")
	f2 := NewLinePragmaBase(Pos{}, "f2", 10)
	f3 := NewLinePragmaBase(MakePos(f1, 10, 1), "f3", 100)
	f4 := NewLinePragmaBase(MakePos(f3, 10, 1), "f4", 100)

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
		{Pos{}, ":0", "", 0, 0, "", 0},
		{MakePos(nil, 2, 3), ":2:3", "", 2, 3, "", 2},
		{MakePos(f0, 2, 3), ":2:3", "", 2, 3, "", 2},
		{MakePos(f1, 1, 1), "f1:1:1", "f1", 1, 1, "f1", 1},
		{MakePos(f2, 7, 10), "f2:16:10[:0]", "", 7, 10, "f2", 16},
		{MakePos(f3, 12, 7), "f3:101:7[f1:10:1]", "f1", 12, 7, "f3", 101},
		{MakePos(f4, 25, 1), "f4:114:1[f3:99:1[f1:10:1]]", "f3", 25, 1, "f4", 114}, // doesn't occur in Go code
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
		{makeLico(1, colMax), fmt.Sprintf(":1:%d", colMax), 1, colMax},
		{makeLico(1, colMax+1), ":1", 1, 0}, // column too large
		{makeLico(lineMax+1, colMax+1), fmt.Sprintf(":%d", lineMax), lineMax, 0},
	} {
		x := test.x
		if got := posString("", x.Line(), x.Col()); got != test.string {
			t.Errorf("%s: got %q", test.string, got)
		}
	}
}

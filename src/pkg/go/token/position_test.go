// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

import (
	"fmt"
	"testing"
)


func checkPos(t *testing.T, msg string, p, q Position) {
	if p.Filename != q.Filename {
		t.Errorf("%s: expected filename = %q; got %q", msg, q.Filename, p.Filename)
	}
	if p.Offset != q.Offset {
		t.Errorf("%s: expected offset = %d; got %d", msg, q.Offset, p.Offset)
	}
	if p.Line != q.Line {
		t.Errorf("%s: expected line = %d; got %d", msg, q.Line, p.Line)
	}
	if p.Column != q.Column {
		t.Errorf("%s: expected column = %d; got %d", msg, q.Column, p.Column)
	}
}


func TestNoPos(t *testing.T) {
	if NoPos.IsValid() {
		t.Errorf("NoPos should not be valid")
	}
	var fset *FileSet
	checkPos(t, "nil NoPos", fset.Position(NoPos), Position{})
	fset = NewFileSet()
	checkPos(t, "fset NoPos", fset.Position(NoPos), Position{})
}


var tests = []struct {
	filename string
	size     int
	lines    []int
}{
	{"a", 0, []int{}},
	{"b", 5, []int{0}},
	{"c", 10, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
	{"d", 100, []int{0, 5, 10, 20, 30, 70, 71, 72, 80, 85, 90, 99}},
	{"e", 777, []int{0, 80, 100, 120, 130, 180, 267, 455, 500, 567, 620}},
}


func linecol(lines []int, offs int) (int, int) {
	prevLineOffs := 0
	for line, lineOffs := range lines {
		if offs < lineOffs {
			return line, offs - prevLineOffs + 1
		}
		prevLineOffs = lineOffs
	}
	return len(lines), offs - prevLineOffs + 1
}


func verifyPositions(t *testing.T, fset *FileSet, f *File, lines []int) {
	for offs := 0; offs < f.Size(); offs++ {
		p := f.Pos(offs)
		offs2 := f.Offset(p)
		if offs2 != offs {
			t.Errorf("%s, Offset: expected offset %d; got %d", f.Name(), offs, offs2)
		}
		line, col := linecol(lines, offs)
		msg := fmt.Sprintf("%s (offs = %d, p = %d)", f.Name(), offs, p)
		checkPos(t, msg, f.Position(f.Pos(offs)), Position{f.Name(), offs, line, col})
		checkPos(t, msg, fset.Position(p), Position{f.Name(), offs, line, col})
	}
}


func TestPositions(t *testing.T) {
	const delta = 7 // a non-zero base offset increment
	fset := NewFileSet()
	for _, test := range tests {
		// add file and verify name and size
		f := fset.AddFile(test.filename, fset.Base()+delta, test.size)
		if f.Name() != test.filename {
			t.Errorf("expected filename %q; got %q", test.filename, f.Name())
		}
		if f.Size() != test.size {
			t.Errorf("%s: expected file size %d; got %d", f.Name(), test.size, f.Size())
		}
		if fset.File(f.Pos(0)) != f {
			t.Errorf("%s: f.Pos(0) was not found in f", f.Name())
		}

		// add lines individually and verify all positions
		for i, offset := range test.lines {
			f.AddLine(offset)
			if f.LineCount() != i+1 {
				t.Errorf("%s, AddLine: expected line count %d; got %d", f.Name(), i+1, f.LineCount())
			}
			// adding the same offset again should be ignored
			f.AddLine(offset)
			if f.LineCount() != i+1 {
				t.Errorf("%s, AddLine: expected unchanged line count %d; got %d", f.Name(), i+1, f.LineCount())
			}
			verifyPositions(t, fset, f, test.lines[0:i+1])
		}

		// add lines at once and verify all positions
		ok := f.SetLines(test.lines)
		if !ok {
			t.Errorf("%s: SetLines failed", f.Name())
		}
		if f.LineCount() != len(test.lines) {
			t.Errorf("%s, SetLines: expected line count %d; got %d", f.Name(), len(test.lines), f.LineCount())
		}
		verifyPositions(t, fset, f, test.lines)
	}
}


func TestLineInfo(t *testing.T) {
	fset := NewFileSet()
	f := fset.AddFile("foo", fset.Base(), 500)
	lines := []int{0, 42, 77, 100, 210, 220, 277, 300, 333, 401}
	// add lines individually and provide alternative line information
	for _, offs := range lines {
		f.AddLine(offs)
		f.AddLineInfo(offs, "bar", 42)
	}
	// verify positions for all offsets
	for offs := 0; offs <= f.Size(); offs++ {
		p := f.Pos(offs)
		_, col := linecol(lines, offs)
		msg := fmt.Sprintf("%s (offs = %d, p = %d)", f.Name(), offs, p)
		checkPos(t, msg, f.Position(f.Pos(offs)), Position{"bar", offs, 42, col})
		checkPos(t, msg, fset.Position(p), Position{"bar", offs, 42, col})
	}
}


func TestFiles(t *testing.T) {
	fset := NewFileSet()
	for i, test := range tests {
		fset.AddFile(test.filename, fset.Base(), test.size)
		j := 0
		for g := range fset.Files() {
			if g.Name() != tests[j].filename {
				t.Errorf("expected filename = %s; got %s", tests[j].filename, g.Name())
			}
			j++
		}
		if j != i+1 {
			t.Errorf("expected %d files; got %d", i+1, j)
		}
	}
}

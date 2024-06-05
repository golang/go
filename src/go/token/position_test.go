// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

import (
	"fmt"
	"math/rand"
	"reflect"
	"sync"
	"testing"
)

func checkPos(t *testing.T, msg string, got, want Position) {
	if got.Filename != want.Filename {
		t.Errorf("%s: got filename = %q; want %q", msg, got.Filename, want.Filename)
	}
	if got.Offset != want.Offset {
		t.Errorf("%s: got offset = %d; want %d", msg, got.Offset, want.Offset)
	}
	if got.Line != want.Line {
		t.Errorf("%s: got line = %d; want %d", msg, got.Line, want.Line)
	}
	if got.Column != want.Column {
		t.Errorf("%s: got column = %d; want %d", msg, got.Column, want.Column)
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
	source   []byte // may be nil
	size     int
	lines    []int
}{
	{"a", []byte{}, 0, []int{}},
	{"b", []byte("01234"), 5, []int{0}},
	{"c", []byte("\n\n\n\n\n\n\n\n\n"), 9, []int{0, 1, 2, 3, 4, 5, 6, 7, 8}},
	{"d", nil, 100, []int{0, 5, 10, 20, 30, 70, 71, 72, 80, 85, 90, 99}},
	{"e", nil, 777, []int{0, 80, 100, 120, 130, 180, 267, 455, 500, 567, 620}},
	{"f", []byte("package p\n\nimport \"fmt\""), 23, []int{0, 10, 11}},
	{"g", []byte("package p\n\nimport \"fmt\"\n"), 24, []int{0, 10, 11}},
	{"h", []byte("package p\n\nimport \"fmt\"\n "), 25, []int{0, 10, 11, 24}},
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
			t.Errorf("%s, Offset: got offset %d; want %d", f.Name(), offs2, offs)
		}
		line, col := linecol(lines, offs)
		msg := fmt.Sprintf("%s (offs = %d, p = %d)", f.Name(), offs, p)
		checkPos(t, msg, f.Position(f.Pos(offs)), Position{f.Name(), offs, line, col})
		checkPos(t, msg, fset.Position(p), Position{f.Name(), offs, line, col})
	}
}

func makeTestSource(size int, lines []int) []byte {
	src := make([]byte, size)
	for _, offs := range lines {
		if offs > 0 {
			src[offs-1] = '\n'
		}
	}
	return src
}

func TestPositions(t *testing.T) {
	const delta = 7 // a non-zero base offset increment
	fset := NewFileSet()
	for _, test := range tests {
		// verify consistency of test case
		if test.source != nil && len(test.source) != test.size {
			t.Errorf("%s: inconsistent test case: got file size %d; want %d", test.filename, len(test.source), test.size)
		}

		// add file and verify name and size
		f := fset.AddFile(test.filename, fset.Base()+delta, test.size)
		if f.Name() != test.filename {
			t.Errorf("got filename %q; want %q", f.Name(), test.filename)
		}
		if f.Size() != test.size {
			t.Errorf("%s: got file size %d; want %d", f.Name(), f.Size(), test.size)
		}
		if fset.File(f.Pos(0)) != f {
			t.Errorf("%s: f.Pos(0) was not found in f", f.Name())
		}

		// add lines individually and verify all positions
		for i, offset := range test.lines {
			f.AddLine(offset)
			if f.LineCount() != i+1 {
				t.Errorf("%s, AddLine: got line count %d; want %d", f.Name(), f.LineCount(), i+1)
			}
			// adding the same offset again should be ignored
			f.AddLine(offset)
			if f.LineCount() != i+1 {
				t.Errorf("%s, AddLine: got unchanged line count %d; want %d", f.Name(), f.LineCount(), i+1)
			}
			verifyPositions(t, fset, f, test.lines[0:i+1])
		}

		// add lines with SetLines and verify all positions
		if ok := f.SetLines(test.lines); !ok {
			t.Errorf("%s: SetLines failed", f.Name())
		}
		if f.LineCount() != len(test.lines) {
			t.Errorf("%s, SetLines: got line count %d; want %d", f.Name(), f.LineCount(), len(test.lines))
		}
		if !reflect.DeepEqual(f.Lines(), test.lines) {
			t.Errorf("%s, Lines after SetLines(v): got %v; want %v", f.Name(), f.Lines(), test.lines)
		}
		verifyPositions(t, fset, f, test.lines)

		// add lines with SetLinesForContent and verify all positions
		src := test.source
		if src == nil {
			// no test source available - create one from scratch
			src = makeTestSource(test.size, test.lines)
		}
		f.SetLinesForContent(src)
		if f.LineCount() != len(test.lines) {
			t.Errorf("%s, SetLinesForContent: got line count %d; want %d", f.Name(), f.LineCount(), len(test.lines))
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
		base := fset.Base()
		if i%2 == 1 {
			// Setting a negative base is equivalent to
			// fset.Base(), so test some of each.
			base = -1
		}
		fset.AddFile(test.filename, base, test.size)
		j := 0
		fset.Iterate(func(f *File) bool {
			if f.Name() != tests[j].filename {
				t.Errorf("got filename = %s; want %s", f.Name(), tests[j].filename)
			}
			j++
			return true
		})
		if j != i+1 {
			t.Errorf("got %d files; want %d", j, i+1)
		}
	}
}

// FileSet.File should return nil if Pos is past the end of the FileSet.
func TestFileSetPastEnd(t *testing.T) {
	fset := NewFileSet()
	for _, test := range tests {
		fset.AddFile(test.filename, fset.Base(), test.size)
	}
	if f := fset.File(Pos(fset.Base())); f != nil {
		t.Errorf("got %v, want nil", f)
	}
}

func TestFileSetCacheUnlikely(t *testing.T) {
	fset := NewFileSet()
	offsets := make(map[string]int)
	for _, test := range tests {
		offsets[test.filename] = fset.Base()
		fset.AddFile(test.filename, fset.Base(), test.size)
	}
	for file, pos := range offsets {
		f := fset.File(Pos(pos))
		if f.Name() != file {
			t.Errorf("got %q at position %d, want %q", f.Name(), pos, file)
		}
	}
}

// issue 4345. Test that concurrent use of FileSet.Pos does not trigger a
// race in the FileSet position cache.
func TestFileSetRace(t *testing.T) {
	fset := NewFileSet()
	for i := 0; i < 100; i++ {
		fset.AddFile(fmt.Sprintf("file-%d", i), fset.Base(), 1031)
	}
	max := int32(fset.Base())
	var stop sync.WaitGroup
	r := rand.New(rand.NewSource(7))
	for i := 0; i < 2; i++ {
		r := rand.New(rand.NewSource(r.Int63()))
		stop.Add(1)
		go func() {
			for i := 0; i < 1000; i++ {
				fset.Position(Pos(r.Int31n(max)))
			}
			stop.Done()
		}()
	}
	stop.Wait()
}

// issue 16548. Test that concurrent use of File.AddLine and FileSet.PositionFor
// does not trigger a race in the FileSet position cache.
func TestFileSetRace2(t *testing.T) {
	const N = 1e3
	var (
		fset = NewFileSet()
		file = fset.AddFile("", -1, N)
		ch   = make(chan int, 2)
	)

	go func() {
		for i := 0; i < N; i++ {
			file.AddLine(i)
		}
		ch <- 1
	}()

	go func() {
		pos := file.Pos(0)
		for i := 0; i < N; i++ {
			fset.PositionFor(pos, false)
		}
		ch <- 1
	}()

	<-ch
	<-ch
}

func TestPositionFor(t *testing.T) {
	src := []byte(`
foo
b
ar
//line :100
foobar
//line bar:3
done
`)

	const filename = "foo"
	fset := NewFileSet()
	f := fset.AddFile(filename, fset.Base(), len(src))
	f.SetLinesForContent(src)

	// verify position info
	for i, offs := range f.lines {
		got1 := f.PositionFor(f.Pos(offs), false)
		got2 := f.PositionFor(f.Pos(offs), true)
		got3 := f.Position(f.Pos(offs))
		want := Position{filename, offs, i + 1, 1}
		checkPos(t, "1. PositionFor unadjusted", got1, want)
		checkPos(t, "1. PositionFor adjusted", got2, want)
		checkPos(t, "1. Position", got3, want)
	}

	// manually add //line info on lines l1, l2
	const l1, l2 = 5, 7
	f.AddLineInfo(f.lines[l1-1], "", 100)
	f.AddLineInfo(f.lines[l2-1], "bar", 3)

	// unadjusted position info must remain unchanged
	for i, offs := range f.lines {
		got1 := f.PositionFor(f.Pos(offs), false)
		want := Position{filename, offs, i + 1, 1}
		checkPos(t, "2. PositionFor unadjusted", got1, want)
	}

	// adjusted position info should have changed
	for i, offs := range f.lines {
		got2 := f.PositionFor(f.Pos(offs), true)
		got3 := f.Position(f.Pos(offs))
		want := Position{filename, offs, i + 1, 1}
		// manually compute wanted filename and line
		line := want.Line
		if i+1 >= l1 {
			want.Filename = ""
			want.Line = line - l1 + 100
		}
		if i+1 >= l2 {
			want.Filename = "bar"
			want.Line = line - l2 + 3
		}
		checkPos(t, "3. PositionFor adjusted", got2, want)
		checkPos(t, "3. Position", got3, want)
	}
}

func TestLineStart(t *testing.T) {
	const src = "one\ntwo\nthree\n"
	fset := NewFileSet()
	f := fset.AddFile("input", -1, len(src))
	f.SetLinesForContent([]byte(src))

	for line := 1; line <= 3; line++ {
		pos := f.LineStart(line)
		position := fset.Position(pos)
		if position.Line != line || position.Column != 1 {
			t.Errorf("LineStart(%d) returned wrong pos %d: %s", line, pos, position)
		}
	}
}

func TestRemoveFile(t *testing.T) {
	contentA := []byte("this\nis\nfileA")
	contentB := []byte("this\nis\nfileB")
	fset := NewFileSet()
	a := fset.AddFile("fileA", -1, len(contentA))
	a.SetLinesForContent(contentA)
	b := fset.AddFile("fileB", -1, len(contentB))
	b.SetLinesForContent(contentB)

	checkPos := func(pos Pos, want string) {
		if got := fset.Position(pos).String(); got != want {
			t.Errorf("Position(%d) = %s, want %s", pos, got, want)
		}
	}
	checkNumFiles := func(want int) {
		got := 0
		fset.Iterate(func(*File) bool { got++; return true })
		if got != want {
			t.Errorf("Iterate called %d times, want %d", got, want)
		}
	}

	apos3 := a.Pos(3)
	bpos3 := b.Pos(3)
	checkPos(apos3, "fileA:1:4")
	checkPos(bpos3, "fileB:1:4")
	checkNumFiles(2)

	// After removal, queries on fileA fail.
	fset.RemoveFile(a)
	checkPos(apos3, "-")
	checkPos(bpos3, "fileB:1:4")
	checkNumFiles(1)

	// idempotent / no effect
	fset.RemoveFile(a)
	checkPos(apos3, "-")
	checkPos(bpos3, "fileB:1:4")
	checkNumFiles(1)
}

func TestFileAddLineColumnInfo(t *testing.T) {
	const (
		filename = "test.go"
		filesize = 100
	)

	tests := []struct {
		name  string
		infos []lineInfo
		want  []lineInfo
	}{
		{
			name: "normal",
			infos: []lineInfo{
				{Offset: 10, Filename: filename, Line: 2, Column: 1},
				{Offset: 50, Filename: filename, Line: 3, Column: 1},
				{Offset: 80, Filename: filename, Line: 4, Column: 2},
			},
			want: []lineInfo{
				{Offset: 10, Filename: filename, Line: 2, Column: 1},
				{Offset: 50, Filename: filename, Line: 3, Column: 1},
				{Offset: 80, Filename: filename, Line: 4, Column: 2},
			},
		},
		{
			name: "offset1 == file size",
			infos: []lineInfo{
				{Offset: filesize, Filename: filename, Line: 2, Column: 1},
			},
			want: nil,
		},
		{
			name: "offset1 > file size",
			infos: []lineInfo{
				{Offset: filesize + 1, Filename: filename, Line: 2, Column: 1},
			},
			want: nil,
		},
		{
			name: "offset2 == file size",
			infos: []lineInfo{
				{Offset: 10, Filename: filename, Line: 2, Column: 1},
				{Offset: filesize, Filename: filename, Line: 3, Column: 1},
			},
			want: []lineInfo{
				{Offset: 10, Filename: filename, Line: 2, Column: 1},
			},
		},
		{
			name: "offset2 > file size",
			infos: []lineInfo{
				{Offset: 10, Filename: filename, Line: 2, Column: 1},
				{Offset: filesize + 1, Filename: filename, Line: 3, Column: 1},
			},
			want: []lineInfo{
				{Offset: 10, Filename: filename, Line: 2, Column: 1},
			},
		},
		{
			name: "offset2 == offset1",
			infos: []lineInfo{
				{Offset: 10, Filename: filename, Line: 2, Column: 1},
				{Offset: 10, Filename: filename, Line: 3, Column: 1},
			},
			want: []lineInfo{
				{Offset: 10, Filename: filename, Line: 2, Column: 1},
			},
		},
		{
			name: "offset2 < offset1",
			infos: []lineInfo{
				{Offset: 10, Filename: filename, Line: 2, Column: 1},
				{Offset: 9, Filename: filename, Line: 3, Column: 1},
			},
			want: []lineInfo{
				{Offset: 10, Filename: filename, Line: 2, Column: 1},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fs := NewFileSet()
			f := fs.AddFile(filename, -1, filesize)
			for _, info := range test.infos {
				f.AddLineColumnInfo(info.Offset, info.Filename, info.Line, info.Column)
			}
			if !reflect.DeepEqual(f.infos, test.want) {
				t.Errorf("\ngot %+v, \nwant %+v", f.infos, test.want)
			}
		})
	}
}

func TestIssue57490(t *testing.T) {
	// If debug is set, this test is expected to panic.
	if debug {
		defer func() {
			if recover() == nil {
				t.Errorf("got no panic")
			}
		}()
	}

	const fsize = 5
	fset := NewFileSet()
	base := fset.Base()
	f := fset.AddFile("f", base, fsize)

	// out-of-bounds positions must not lead to a panic when calling f.Offset
	if got := f.Offset(NoPos); got != 0 {
		t.Errorf("offset = %d, want %d", got, 0)
	}
	if got := f.Offset(Pos(-1)); got != 0 {
		t.Errorf("offset = %d, want %d", got, 0)
	}
	if got := f.Offset(Pos(base + fsize + 1)); got != fsize {
		t.Errorf("offset = %d, want %d", got, fsize)
	}

	// out-of-bounds offsets must not lead to a panic when calling f.Pos
	if got := f.Pos(-1); got != Pos(base) {
		t.Errorf("pos = %d, want %d", got, base)
	}
	if got := f.Pos(fsize + 1); got != Pos(base+fsize) {
		t.Errorf("pos = %d, want %d", got, base+fsize)
	}

	// out-of-bounds Pos values must not lead to a panic when calling f.Position
	want := fmt.Sprintf("%s:1:1", f.Name())
	if got := f.Position(Pos(-1)).String(); got != want {
		t.Errorf("position = %s, want %s", got, want)
	}
	want = fmt.Sprintf("%s:1:%d", f.Name(), fsize+1)
	if got := f.Position(Pos(fsize + 1)).String(); got != want {
		t.Errorf("position = %s, want %s", got, want)
	}

	// check invariants
	const xsize = fsize + 5
	for offset := -xsize; offset < xsize; offset++ {
		want1 := f.Offset(Pos(f.base + offset))
		if got := f.Offset(f.Pos(offset)); got != want1 {
			t.Errorf("offset = %d, want %d", got, want1)
		}

		want2 := f.Pos(offset)
		if got := f.Pos(f.Offset(want2)); got != want2 {
			t.Errorf("pos = %d, want %d", got, want2)
		}
	}
}

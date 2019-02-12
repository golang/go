// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span

import (
	"go/token"
)

// Range represents a source code range in token.Pos form.
// It also carries the FileSet that produced the positions, so that it is
// self contained.
type Range struct {
	FileSet *token.FileSet
	Start   token.Pos
	End     token.Pos
}

// TokenConverter is a Converter backed by a token file set and file.
// It uses the file set methods to work out determine the conversions which
// make if fast and do not require the file contents.
type TokenConverter struct {
	fset *token.FileSet
	file *token.File
}

// NewRange creates a new Range from a FileSet and two positions.
// To represent a point pass a 0 as the end pos.
func NewRange(fset *token.FileSet, start, end token.Pos) Range {
	return Range{
		FileSet: fset,
		Start:   start,
		End:     end,
	}
}

// NewTokenConverter returns an implementation of Coords and Offsets backed by a
// token.File.
func NewTokenConverter(fset *token.FileSet, f *token.File) *TokenConverter {
	return &TokenConverter{fset: fset, file: f}
}

// NewContentConverter returns an implementation of Coords and Offsets for the
// given file content.
func NewContentConverter(filename string, content []byte) Converter {
	fset := token.NewFileSet()
	f := fset.AddFile(filename, -1, len(content))
	f.SetLinesForContent(content)
	return &TokenConverter{fset: fset, file: f}
}

// IsPoint returns true if the range represents a single point.
func (r Range) IsPoint() bool {
	return r.Start == r.End
}

// Span converts a Range to a Span that represents the Range.
// It will fill in all the members of the Span, calculating the line and column
// information.
func (r Range) Span() Span {
	f := r.FileSet.File(r.Start)
	s := Span{URI: FileURI(f.Name())}
	s.Start.Offset = f.Offset(r.Start)
	if r.End.IsValid() {
		s.End.Offset = f.Offset(r.End)
	}
	converter := NewTokenConverter(r.FileSet, f)
	return s.CleanCoords(converter)
}

// Range converts a Span to a Range that represents the Span for the supplied
// File.
func (s Span) Range(converter *TokenConverter) Range {
	s = s.CleanOffset(converter)
	return Range{
		FileSet: converter.fset,
		Start:   converter.file.Pos(s.Start.Offset),
		End:     converter.file.Pos(s.End.Offset),
	}
}

func (l *TokenConverter) ToCoord(offset int) (int, int) {
	pos := l.file.Pos(offset)
	p := l.fset.Position(pos)
	return p.Line, p.Column
}

func (l *TokenConverter) ToOffset(line, col int) int {
	if line < 0 {
		// before the start of the file
		return -1
	}
	lineMax := l.file.LineCount() + 1
	if line > lineMax {
		// after the end of the file
		return -1
	} else if line == lineMax {
		if col > 1 {
			// after the end of the file
			return -1
		}
		// at the end of the file, allowing for a trailing eol
		return l.file.Size()
	}
	pos := lineStart(l.file, line)
	if !pos.IsValid() {
		return -1
	}
	// we assume that column is in bytes here, and that the first byte of a
	// line is at column 1
	pos += token.Pos(col - 1)
	return l.file.Offset(pos)
}

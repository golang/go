// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span

import (
	"fmt"
	"go/token"
)

// Range represents a source code range in token.Pos form.
// It also carries the FileSet that produced the positions, so that it is
// self contained.
type Range struct {
	FileSet   *token.FileSet
	Start     token.Pos
	End       token.Pos
	Converter Converter
}

// TokenConverter is a Converter backed by a token file set and file.
// It uses the file set methods to work out the conversions, which
// makes it fast and does not require the file contents.
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

// NewTokenConverter returns an implementation of Converter backed by a
// token.File.
func NewTokenConverter(fset *token.FileSet, f *token.File) *TokenConverter {
	return &TokenConverter{fset: fset, file: f}
}

// NewContentConverter returns an implementation of Converter for the
// given file content.
func NewContentConverter(filename string, content []byte) *TokenConverter {
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
func (r Range) Span() (Span, error) {
	if !r.Start.IsValid() {
		return Span{}, fmt.Errorf("start pos is not valid")
	}
	f := r.FileSet.File(r.Start)
	if f == nil {
		return Span{}, fmt.Errorf("file not found in FileSet")
	}
	var s Span
	var err error
	var startFilename string
	startFilename, s.v.Start.Line, s.v.Start.Column, err = position(f, r.Start)
	if err != nil {
		return Span{}, err
	}
	s.v.URI = URIFromPath(startFilename)
	if r.End.IsValid() {
		var endFilename string
		endFilename, s.v.End.Line, s.v.End.Column, err = position(f, r.End)
		if err != nil {
			return Span{}, err
		}
		// In the presence of line directives, a single File can have sections from
		// multiple file names.
		if endFilename != startFilename {
			return Span{}, fmt.Errorf("span begins in file %q but ends in %q", startFilename, endFilename)
		}
	}
	s.v.Start.clean()
	s.v.End.clean()
	s.v.clean()
	if r.Converter != nil {
		return s.WithOffset(r.Converter)
	}
	if startFilename != f.Name() {
		return Span{}, fmt.Errorf("must supply Converter for file %q containing lines from %q", f.Name(), startFilename)
	}
	return s.WithOffset(NewTokenConverter(r.FileSet, f))
}

func position(f *token.File, pos token.Pos) (string, int, int, error) {
	off, err := offset(f, pos)
	if err != nil {
		return "", 0, 0, err
	}
	return positionFromOffset(f, off)
}

func positionFromOffset(f *token.File, offset int) (string, int, int, error) {
	if offset > f.Size() {
		return "", 0, 0, fmt.Errorf("offset %v is past the end of the file %v", offset, f.Size())
	}
	pos := f.Pos(offset)
	p := f.Position(pos)
	// TODO(golang/go#41029): Consider returning line, column instead of line+1, 1 if
	// the file's last character is not a newline.
	if offset == f.Size() {
		return p.Filename, p.Line + 1, 1, nil
	}
	return p.Filename, p.Line, p.Column, nil
}

// offset is a copy of the Offset function in go/token, but with the adjustment
// that it does not panic on invalid positions.
func offset(f *token.File, pos token.Pos) (int, error) {
	if int(pos) < f.Base() || int(pos) > f.Base()+f.Size() {
		return 0, fmt.Errorf("invalid pos")
	}
	return int(pos) - f.Base(), nil
}

// Range converts a Span to a Range that represents the Span for the supplied
// File.
func (s Span) Range(converter *TokenConverter) (Range, error) {
	s, err := s.WithOffset(converter)
	if err != nil {
		return Range{}, err
	}
	// go/token will panic if the offset is larger than the file's size,
	// so check here to avoid panicking.
	if s.Start().Offset() > converter.file.Size() {
		return Range{}, fmt.Errorf("start offset %v is past the end of the file %v", s.Start(), converter.file.Size())
	}
	if s.End().Offset() > converter.file.Size() {
		return Range{}, fmt.Errorf("end offset %v is past the end of the file %v", s.End(), converter.file.Size())
	}
	return Range{
		FileSet:   converter.fset,
		Start:     converter.file.Pos(s.Start().Offset()),
		End:       converter.file.Pos(s.End().Offset()),
		Converter: converter,
	}, nil
}

func (l *TokenConverter) ToPosition(offset int) (int, int, error) {
	_, line, col, err := positionFromOffset(l.file, offset)
	return line, col, err
}

func (l *TokenConverter) ToOffset(line, col int) (int, error) {
	if line < 0 {
		return -1, fmt.Errorf("line is not valid")
	}
	lineMax := l.file.LineCount() + 1
	if line > lineMax {
		return -1, fmt.Errorf("line is beyond end of file %v", lineMax)
	} else if line == lineMax {
		if col > 1 {
			return -1, fmt.Errorf("column is beyond end of file")
		}
		// at the end of the file, allowing for a trailing eol
		return l.file.Size(), nil
	}
	pos := lineStart(l.file, line)
	if !pos.IsValid() {
		return -1, fmt.Errorf("line is not in file")
	}
	// we assume that column is in bytes here, and that the first byte of a
	// line is at column 1
	pos += token.Pos(col - 1)
	return offset(l.file, pos)
}

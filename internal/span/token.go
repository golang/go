// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span

import (
	"errors"
	"fmt"
	"go/token"
)

// Range represents a source code range in token.Pos form.
// It also carries the FileSet that produced the positions, so that it is
// self contained.
type Range struct {
	Start token.Pos
	End   token.Pos
	file  *token.File // possibly nil, if Start or End is invalid
}

// TokenConverter converts between offsets and (col, row) using a token.File.
type TokenConverter struct {
	file *token.File
}

// NewRange creates a new Range from a FileSet and two positions.
// To represent a point pass a 0 as the end pos.
func NewRange(fset *token.FileSet, start, end token.Pos) Range {
	file := fset.File(start)
	if file == nil {
		// TODO: report a bug here via the bug reporting API, once it is available.
	}
	return Range{
		Start: start,
		End:   end,
		file:  file,
	}
}

// NewTokenConverter returns an implementation of Converter backed by a
// token.File.
func NewTokenConverter(f *token.File) *TokenConverter {
	if f == nil {
		// TODO: report a bug here using the bug reporting API.
	}
	return &TokenConverter{file: f}
}

// NewContentConverter returns an implementation of Converter for the
// given file content.
func NewContentConverter(filename string, content []byte) *TokenConverter {
	fset := token.NewFileSet()
	f := fset.AddFile(filename, -1, len(content))
	f.SetLinesForContent(content)
	return NewTokenConverter(f)
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
	if r.file == nil {
		return Span{}, errors.New("range missing file association")
	}
	return FileSpan(r.file, NewTokenConverter(r.file), r.Start, r.End)
}

// FileSpan returns a span within tok, using converter to translate between
// offsets and positions.
func FileSpan(tok *token.File, converter *TokenConverter, start, end token.Pos) (Span, error) {
	var s Span
	var err error
	var startFilename string
	startFilename, s.v.Start.Line, s.v.Start.Column, err = position(tok, start)
	if err != nil {
		return Span{}, err
	}
	s.v.URI = URIFromPath(startFilename)
	if end.IsValid() {
		var endFilename string
		endFilename, s.v.End.Line, s.v.End.Column, err = position(tok, end)
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
	if converter != nil {
		return s.WithOffset(converter)
	}
	if startFilename != tok.Name() {
		return Span{}, fmt.Errorf("must supply Converter for file %q containing lines from %q", tok.Name(), startFilename)
	}
	return s.WithOffset(&TokenConverter{tok})
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
		return 0, fmt.Errorf("invalid pos: %d not in [%d, %d]", pos, f.Base(), f.Base()+f.Size())
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
	file := converter.file
	// go/token will panic if the offset is larger than the file's size,
	// so check here to avoid panicking.
	if s.Start().Offset() > file.Size() {
		// TODO: report a bug here via the future bug reporting API.
		return Range{}, fmt.Errorf("start offset %v is past the end of the file %v", s.Start(), file.Size())
	}
	if s.End().Offset() > file.Size() {
		// TODO: report a bug here via the future bug reporting API.
		return Range{}, fmt.Errorf("end offset %v is past the end of the file %v", s.End(), file.Size())
	}
	return Range{
		Start: file.Pos(s.Start().Offset()),
		End:   file.Pos(s.End().Offset()),
		file:  file,
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
	pos := l.file.LineStart(line)
	if !pos.IsValid() {
		return -1, fmt.Errorf("line is not in file")
	}
	// we assume that column is in bytes here, and that the first byte of a
	// line is at column 1
	pos += token.Pos(col - 1)
	return offset(l.file, pos)
}

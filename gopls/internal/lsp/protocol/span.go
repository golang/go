// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// this file contains protocol<->span converters

package protocol

import (
	"bytes"
	"fmt"
	"go/token"
	"unicode/utf8"

	"golang.org/x/tools/gopls/internal/span"
)

// A ColumnMapper maps between UTF-8 oriented positions (e.g. token.Pos,
// span.Span) and the UTF-16 oriented positions used by the LSP.
type ColumnMapper struct {
	URI     span.URI
	TokFile *token.File
	Content []byte

	// File content is only really needed for UTF-16 column
	// computation, which could be be achieved more compactly.
	// For example, one could record only the lines for which
	// UTF-16 columns differ from the UTF-8 ones, or only the
	// indices of the non-ASCII characters.
	//
	// TODO(adonovan): consider not retaining the entire file
	// content, or at least not exposing the fact that we
	// currently retain it.
}

// NewColumnMapper creates a new column mapper for the given uri and content.
func NewColumnMapper(uri span.URI, content []byte) *ColumnMapper {
	tf := span.NewTokenFile(uri.Filename(), content)
	return &ColumnMapper{
		URI:     uri,
		TokFile: tf,
		Content: content,
	}
}

func URIFromSpanURI(uri span.URI) DocumentURI {
	return DocumentURI(uri)
}

func URIFromPath(path string) DocumentURI {
	return URIFromSpanURI(span.URIFromPath(path))
}

func (u DocumentURI) SpanURI() span.URI {
	return span.URIFromURI(string(u))
}

func (m *ColumnMapper) Location(s span.Span) (Location, error) {
	rng, err := m.Range(s)
	if err != nil {
		return Location{}, err
	}
	return Location{URI: URIFromSpanURI(s.URI()), Range: rng}, nil
}

func (m *ColumnMapper) Range(s span.Span) (Range, error) {
	if span.CompareURI(m.URI, s.URI()) != 0 {
		return Range{}, fmt.Errorf("column mapper is for file %q instead of %q", m.URI, s.URI())
	}
	s, err := s.WithAll(m.TokFile)
	if err != nil {
		return Range{}, err
	}
	start, err := m.Position(s.Start())
	if err != nil {
		return Range{}, err
	}
	end, err := m.Position(s.End())
	if err != nil {
		return Range{}, err
	}
	return Range{Start: start, End: end}, nil
}

// OffsetRange returns a Range for the byte-offset interval Content[start:end],
func (m *ColumnMapper) OffsetRange(start, end int) (Range, error) {
	startPosition, err := m.OffsetPosition(start)
	if err != nil {
		return Range{}, fmt.Errorf("start: %v", err)
	}

	endPosition, err := m.OffsetPosition(end)
	if err != nil {
		return Range{}, fmt.Errorf("end: %v", err)
	}

	return Range{Start: startPosition, End: endPosition}, nil
}

// Position returns the protocol position for the specified point,
// which must have a byte offset.
func (m *ColumnMapper) Position(p span.Point) (Position, error) {
	if !p.HasOffset() {
		return Position{}, fmt.Errorf("point is missing offset")
	}
	return m.OffsetPosition(p.Offset())
}

// OffsetPosition returns the protocol position of the specified
// offset within m.Content.
func (m *ColumnMapper) OffsetPosition(offset int) (Position, error) {
	// We use span.ToPosition for its "line+1 at EOF" workaround.
	// TODO(adonovan): ToPosition honors //line directives. It probably shouldn't.
	line, _, err := span.ToPosition(m.TokFile, offset)
	if err != nil {
		return Position{}, fmt.Errorf("OffsetPosition: %v", err)
	}
	// If that workaround executed, skip the usual column computation.
	char := 0
	if offset != m.TokFile.Size() {
		char = m.utf16Column(offset)
	}
	return Position{
		Line:      uint32(line - 1),
		Character: uint32(char),
	}, nil
}

// utf16Column returns the zero-based column index of the
// specified file offset, measured in UTF-16 codes.
// Precondition: 0 <= offset <= len(m.Content).
func (m *ColumnMapper) utf16Column(offset int) int {
	s := m.Content[:offset]
	if i := bytes.LastIndex(s, []byte("\n")); i >= 0 {
		s = s[i+1:]
	}
	// s is the prefix of the line before offset.
	return utf16len(s)
}

// utf16len returns the number of codes in the UTF-16 transcoding of s.
func utf16len(s []byte) int {
	var n int
	for len(s) > 0 {
		n++

		// Fast path for ASCII.
		if s[0] < 0x80 {
			s = s[1:]
			continue
		}

		r, size := utf8.DecodeRune(s)
		if r >= 0x10000 {
			n++ // surrogate pair
		}
		s = s[size:]
	}
	return n
}

func (m *ColumnMapper) Span(l Location) (span.Span, error) {
	return m.RangeSpan(l.Range)
}

func (m *ColumnMapper) RangeSpan(r Range) (span.Span, error) {
	start, err := m.Point(r.Start)
	if err != nil {
		return span.Span{}, err
	}
	end, err := m.Point(r.End)
	if err != nil {
		return span.Span{}, err
	}
	return span.New(m.URI, start, end).WithAll(m.TokFile)
}

func (m *ColumnMapper) RangeToSpanRange(r Range) (span.Range, error) {
	spn, err := m.RangeSpan(r)
	if err != nil {
		return span.Range{}, err
	}
	return spn.Range(m.TokFile)
}

// Pos returns the token.Pos of p within the mapped file.
func (m *ColumnMapper) Pos(p Position) (token.Pos, error) {
	start, err := m.Point(p)
	if err != nil {
		return token.NoPos, err
	}
	// TODO: refactor the span package to avoid creating this unnecessary end position.
	spn, err := span.New(m.URI, start, start).WithAll(m.TokFile)
	if err != nil {
		return token.NoPos, err
	}
	rng, err := spn.Range(m.TokFile)
	if err != nil {
		return token.NoPos, err
	}
	return rng.Start, nil
}

// Offset returns the utf-8 byte offset of p within the mapped file.
func (m *ColumnMapper) Offset(p Position) (int, error) {
	start, err := m.Point(p)
	if err != nil {
		return 0, err
	}
	return start.Offset(), nil
}

// Point returns a span.Point for p within the mapped file. The resulting point
// always has an Offset.
func (m *ColumnMapper) Point(p Position) (span.Point, error) {
	line := int(p.Line) + 1
	offset, err := span.ToOffset(m.TokFile, line, 1)
	if err != nil {
		return span.Point{}, err
	}
	lineStart := span.NewPoint(line, 1, offset)
	return span.FromUTF16Column(lineStart, int(p.Character)+1, m.Content)
}

func IsPoint(r Range) bool {
	return r.Start.Line == r.End.Line && r.Start.Character == r.End.Character
}

// CompareRange returns -1 if a is before b, 0 if a == b, and 1 if a is after
// b.
//
// A range a is defined to be 'before' b if a.Start is before b.Start, or
// a.Start == b.Start and a.End is before b.End.
func CompareRange(a, b Range) int {
	if r := ComparePosition(a.Start, b.Start); r != 0 {
		return r
	}
	return ComparePosition(a.End, b.End)
}

// ComparePosition returns -1 if a is before b, 0 if a == b, and 1 if a is
// after b.
func ComparePosition(a, b Position) int {
	if a.Line < b.Line {
		return -1
	}
	if a.Line > b.Line {
		return 1
	}
	if a.Character < b.Character {
		return -1
	}
	if a.Character > b.Character {
		return 1
	}
	return 0
}

func Intersect(a, b Range) bool {
	if a.Start.Line > b.End.Line || a.End.Line < b.Start.Line {
		return false
	}
	return !((a.Start.Line == b.End.Line) && a.Start.Character > b.End.Character ||
		(a.End.Line == b.Start.Line) && a.End.Character < b.Start.Character)
}

// Format implements fmt.Formatter.
//
// Note: Formatter is implemented instead of Stringer (presumably) for
// performance reasons, though it is not clear that it matters in practice.
func (r Range) Format(f fmt.State, _ rune) {
	fmt.Fprintf(f, "%v-%v", r.Start, r.End)
}

// Format implements fmt.Formatter.
//
// See Range.Format for discussion of why the Formatter interface is
// implemented rather than Stringer.
func (p Position) Format(f fmt.State, _ rune) {
	fmt.Fprintf(f, "%v:%v", p.Line, p.Character)
}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// this file contains protocol<->span converters

package protocol

import (
	"fmt"

	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

type ColumnMapper struct {
	URI       span.URI
	Converter *span.TokenConverter
	Content   []byte
}

func NewURI(uri span.URI) string {
	return string(uri)
}

func (m *ColumnMapper) Location(s span.Span) (Location, error) {
	rng, err := m.Range(s)
	if err != nil {
		return Location{}, err
	}
	return Location{URI: NewURI(s.URI()), Range: rng}, nil
}

func (m *ColumnMapper) Range(s span.Span) (Range, error) {
	if span.CompareURI(m.URI, s.URI()) != 0 {
		return Range{}, errors.Errorf("column mapper is for file %q instead of %q", m.URI, s.URI())
	}
	s, err := s.WithAll(m.Converter)
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

func (m *ColumnMapper) Position(p span.Point) (Position, error) {
	chr, err := span.ToUTF16Column(p, m.Content)
	if err != nil {
		return Position{}, err
	}
	return Position{
		Line:      float64(p.Line() - 1),
		Character: float64(chr - 1),
	}, nil
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
	return span.New(m.URI, start, end).WithAll(m.Converter)
}

func (m *ColumnMapper) PointSpan(p Position) (span.Span, error) {
	start, err := m.Point(p)
	if err != nil {
		return span.Span{}, err
	}
	return span.New(m.URI, start, start).WithAll(m.Converter)
}

func (m *ColumnMapper) Point(p Position) (span.Point, error) {
	line := int(p.Line) + 1
	offset, err := m.Converter.ToOffset(line, 1)
	if err != nil {
		return span.Point{}, err
	}
	lineStart := span.NewPoint(line, 1, offset)
	return span.FromUTF16Column(lineStart, int(p.Character)+1, m.Content)
}

func IsPoint(r Range) bool {
	return r.Start.Line == r.End.Line && r.Start.Character == r.End.Character
}

func CompareRange(a, b Range) int {
	if r := ComparePosition(a.Start, b.Start); r != 0 {
		return r
	}
	return ComparePosition(a.End, b.End)
}

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

func (r Range) Format(f fmt.State, _ rune) {
	fmt.Fprintf(f, "%v:%v-%v:%v", r.Start.Line, r.Start.Character, r.End.Line, r.End.Character)
}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// this file contains protocol<->span converters

package protocol

import (
	"go/token"
	"golang.org/x/tools/internal/span"
)

type ColumnMapper struct {
	URI       span.URI
	Converter *span.TokenConverter
	Content   []byte
}

func NewURI(uri span.URI) string {
	return string(uri)
}

func NewColumnMapper(uri span.URI, fset *token.FileSet, f *token.File, content []byte) *ColumnMapper {
	return &ColumnMapper{
		URI:       uri,
		Converter: span.NewTokenConverter(fset, f),
		Content:   content,
	}
}

func (m *ColumnMapper) Location(s span.Span) Location {
	return Location{
		URI:   NewURI(s.URI),
		Range: m.Range(s),
	}
}

func (m *ColumnMapper) Range(s span.Span) Range {
	return Range{
		Start: m.Position(s.Start),
		End:   m.Position(s.End),
	}
}

func (m *ColumnMapper) Position(p span.Point) Position {
	chr := span.ToUTF16Column(m.Converter, p, m.Content)
	return Position{
		Line:      float64(p.Line - 1),
		Character: float64(chr - 1),
	}
}

func (m *ColumnMapper) Span(l Location) span.Span {
	return span.Span{
		URI:   m.URI,
		Start: m.Point(l.Range.Start),
		End:   m.Point(l.Range.End),
	}.Clean(m.Converter)
}

func (m *ColumnMapper) RangeSpan(r Range) span.Span {
	return span.Span{
		URI:   m.URI,
		Start: m.Point(r.Start),
		End:   m.Point(r.End),
	}.Clean(m.Converter)
}

func (m *ColumnMapper) PointSpan(p Position) span.Span {
	return span.Span{
		URI:   m.URI,
		Start: m.Point(p),
	}.Clean(m.Converter)
}

func (m *ColumnMapper) Point(p Position) span.Point {
	return span.FromUTF16Column(m.Converter, int(p.Line)+1, int(p.Character)+1, m.Content)
}

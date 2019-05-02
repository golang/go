// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span

import (
	"encoding/json"
	"fmt"
	"path"
)

// Span represents a source code range in standardized form.
type Span struct {
	v span
}

// Point represents a single point within a file.
// In general this should only be used as part of a Span, as on its own it
// does not carry enough information.
type Point struct {
	v point
}

type span struct {
	URI   URI   `json:"uri"`
	Start point `json:"start"`
	End   point `json:"end"`
}

type point struct {
	Line   int `json:"line"`
	Column int `json:"column"`
	Offset int `json:"offset"`
}

var invalidPoint = Point{v: point{Line: 0, Column: 0, Offset: -1}}

// Converter is the interface to an object that can convert between line:column
// and offset forms for a single file.
type Converter interface {
	//ToPosition converts from an offset to a line:column pair.
	ToPosition(offset int) (int, int, error)
	//ToOffset converts from a line:column pair to an offset.
	ToOffset(line, col int) (int, error)
}

func New(uri URI, start Point, end Point) Span {
	s := Span{v: span{URI: uri, Start: start.v, End: end.v}}
	s.v.clean()
	return s
}

func NewPoint(line, col, offset int) Point {
	p := Point{v: point{Line: line, Column: col, Offset: offset}}
	p.v.clean()
	return p
}

func Compare(a, b Span) int {
	if r := CompareURI(a.URI(), b.URI()); r != 0 {
		return r
	}
	if r := comparePoint(a.v.Start, b.v.Start); r != 0 {
		return r
	}
	return comparePoint(a.v.End, b.v.End)
}

func ComparePoint(a, b Point) int {
	return comparePoint(a.v, b.v)
}

func comparePoint(a, b point) int {
	if !a.hasPosition() {
		if a.Offset < b.Offset {
			return -1
		}
		if a.Offset > b.Offset {
			return 1
		}
		return 0
	}
	if a.Line < b.Line {
		return -1
	}
	if a.Line > b.Line {
		return 1
	}
	if a.Column < b.Column {
		return -1
	}
	if a.Column > b.Column {
		return 1
	}
	return 0
}

func (s Span) HasPosition() bool             { return s.v.Start.hasPosition() }
func (s Span) HasOffset() bool               { return s.v.Start.hasOffset() }
func (s Span) IsValid() bool                 { return s.v.Start.isValid() }
func (s Span) IsPoint() bool                 { return s.v.Start == s.v.End }
func (s Span) URI() URI                      { return s.v.URI }
func (s Span) Start() Point                  { return Point{s.v.Start} }
func (s Span) End() Point                    { return Point{s.v.End} }
func (s *Span) MarshalJSON() ([]byte, error) { return json.Marshal(&s.v) }
func (s *Span) UnmarshalJSON(b []byte) error { return json.Unmarshal(b, &s.v) }

func (p Point) HasPosition() bool             { return p.v.hasPosition() }
func (p Point) HasOffset() bool               { return p.v.hasOffset() }
func (p Point) IsValid() bool                 { return p.v.isValid() }
func (p *Point) MarshalJSON() ([]byte, error) { return json.Marshal(&p.v) }
func (p *Point) UnmarshalJSON(b []byte) error { return json.Unmarshal(b, &p.v) }
func (p Point) Line() int {
	if !p.v.hasPosition() {
		panic(fmt.Errorf("position not set in %v", p.v))
	}
	return p.v.Line
}
func (p Point) Column() int {
	if !p.v.hasPosition() {
		panic(fmt.Errorf("position not set in %v", p.v))
	}
	return p.v.Column
}
func (p Point) Offset() int {
	if !p.v.hasOffset() {
		panic(fmt.Errorf("offset not set in %v", p.v))
	}
	return p.v.Offset
}

func (p point) hasPosition() bool { return p.Line > 0 }
func (p point) hasOffset() bool   { return p.Offset >= 0 }
func (p point) isValid() bool     { return p.hasPosition() || p.hasOffset() }
func (p point) isZero() bool {
	return (p.Line == 1 && p.Column == 1) || (!p.hasPosition() && p.Offset == 0)
}

func (s *span) clean() {
	//this presumes the points are already clean
	if !s.End.isValid() || (s.End == point{}) {
		s.End = s.Start
	}
}

func (p *point) clean() {
	if p.Line < 0 {
		p.Line = 0
	}
	if p.Column <= 0 {
		if p.Line > 0 {
			p.Column = 1
		} else {
			p.Column = 0
		}
	}
	if p.Offset == 0 && (p.Line > 1 || p.Column > 1) {
		p.Offset = -1
	}
}

// Format implements fmt.Formatter to print the Location in a standard form.
// The format produced is one that can be read back in using Parse.
func (s Span) Format(f fmt.State, c rune) {
	fullForm := f.Flag('+')
	preferOffset := f.Flag('#')
	// we should always have a uri, simplify if it is file format
	//TODO: make sure the end of the uri is unambiguous
	uri := string(s.v.URI)
	if c == 'f' {
		uri = path.Base(uri)
	} else if !fullForm {
		if filename, err := s.v.URI.Filename(); err == nil {
			uri = filename
		}
	}
	fmt.Fprint(f, uri)
	if !s.IsValid() || (!fullForm && s.v.Start.isZero() && s.v.End.isZero()) {
		return
	}
	// see which bits of start to write
	printOffset := s.HasOffset() && (fullForm || preferOffset || !s.HasPosition())
	printLine := s.HasPosition() && (fullForm || !printOffset)
	printColumn := printLine && (fullForm || (s.v.Start.Column > 1 || s.v.End.Column > 1))
	fmt.Fprint(f, ":")
	if printLine {
		fmt.Fprintf(f, "%d", s.v.Start.Line)
	}
	if printColumn {
		fmt.Fprintf(f, ":%d", s.v.Start.Column)
	}
	if printOffset {
		fmt.Fprintf(f, "#%d", s.v.Start.Offset)
	}
	// start is written, do we need end?
	if s.IsPoint() {
		return
	}
	// we don't print the line if it did not change
	printLine = fullForm || (printLine && s.v.End.Line > s.v.Start.Line)
	fmt.Fprint(f, "-")
	if printLine {
		fmt.Fprintf(f, "%d", s.v.End.Line)
	}
	if printColumn {
		if printLine {
			fmt.Fprint(f, ":")
		}
		fmt.Fprintf(f, "%d", s.v.End.Column)
	}
	if printOffset {
		fmt.Fprintf(f, "#%d", s.v.End.Offset)
	}
}

func (s Span) WithPosition(c Converter) (Span, error) {
	if err := s.update(c, true, false); err != nil {
		return Span{}, err
	}
	return s, nil
}

func (s Span) WithOffset(c Converter) (Span, error) {
	if err := s.update(c, false, true); err != nil {
		return Span{}, err
	}
	return s, nil
}

func (s Span) WithAll(c Converter) (Span, error) {
	if err := s.update(c, true, true); err != nil {
		return Span{}, err
	}
	return s, nil
}

func (s *Span) update(c Converter, withPos, withOffset bool) error {
	if !s.IsValid() {
		return fmt.Errorf("cannot add information to an invalid span")
	}
	if withPos && !s.HasPosition() {
		if err := s.v.Start.updatePosition(c); err != nil {
			return err
		}
		if s.v.End.Offset == s.v.Start.Offset {
			s.v.End = s.v.Start
		} else if err := s.v.End.updatePosition(c); err != nil {
			return err
		}
	}
	if withOffset && (!s.HasOffset() || (s.v.End.hasPosition() && !s.v.End.hasOffset())) {
		if err := s.v.Start.updateOffset(c); err != nil {
			return err
		}
		if s.v.End.Line == s.v.Start.Line && s.v.End.Column == s.v.Start.Column {
			s.v.End.Offset = s.v.Start.Offset
		} else if err := s.v.End.updateOffset(c); err != nil {
			return err
		}
	}
	return nil
}

func (p *point) updatePosition(c Converter) error {
	line, col, err := c.ToPosition(p.Offset)
	if err != nil {
		return err
	}
	p.Line = line
	p.Column = col
	return nil
}

func (p *point) updateOffset(c Converter) error {
	offset, err := c.ToOffset(p.Line, p.Column)
	if err != nil {
		return err
	}
	p.Offset = offset
	return nil
}

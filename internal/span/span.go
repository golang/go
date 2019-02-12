// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span

import (
	"fmt"
)

// Span represents a source code range in standardized form.
type Span struct {
	URI   URI   `json:"uri"`
	Start Point `json:"start"`
	End   Point `json:"end"`
}

// Point represents a single point within a file.
// In general this should only be used as part of a Span, as on its own it
// does not carry enough information.
type Point struct {
	Line   int `json:"line"`
	Column int `json:"column"`
	Offset int `json:"offset"`
}

// Offsets is the interface to an object that can convert to offset
// from line:column forms for a single file.
type Offsets interface {
	ToOffset(line, col int) int
}

// Coords is the interface to an object that can convert to line:column
// from offset forms for a single file.
type Coords interface {
	ToCoord(offset int) (int, int)
}

// Converter is the interface to an object that can convert between line:column
// and offset forms for a single file.
type Converter interface {
	Offsets
	Coords
}

// Format implements fmt.Formatter to print the Location in a standard form.
// The format produced is one that can be read back in using Parse.
func (s Span) Format(f fmt.State, c rune) {
	fullForm := f.Flag('+')
	preferOffset := f.Flag('#')
	// we should always have a uri, simplify if it is file format
	//TODO: make sure the end of the uri is unambiguous
	uri := string(s.URI)
	if !fullForm {
		if filename, err := s.URI.Filename(); err == nil {
			uri = filename
		}
	}
	fmt.Fprint(f, uri)
	// see which bits of start to write
	printOffset := fullForm || (s.Start.Offset > 0 && (preferOffset || s.Start.Line <= 0))
	printLine := fullForm || (s.Start.Line > 0 && !printOffset)
	printColumn := fullForm || (printLine && (s.Start.Column > 1 || s.End.Column > 1))
	if !printLine && !printColumn && !printOffset {
		return
	}
	fmt.Fprint(f, ":")
	if printLine {
		fmt.Fprintf(f, "%d", clamp(s.Start.Line))
	}
	if printColumn {
		fmt.Fprintf(f, ":%d", clamp(s.Start.Column))
	}
	if printOffset {
		fmt.Fprintf(f, "#%d", clamp(s.Start.Offset))
	}
	// start is written, do we need end?
	printLine = fullForm || (printLine && s.End.Line > s.Start.Line)
	isPoint := s.End.Line == s.Start.Line && s.End.Column == s.Start.Column
	printColumn = fullForm || (printColumn && s.End.Column > 0 && !isPoint)
	printOffset = fullForm || (printOffset && s.End.Offset > s.Start.Offset)
	if !printLine && !printColumn && !printOffset {
		return
	}
	fmt.Fprint(f, "-")
	if printLine {
		fmt.Fprintf(f, "%d", clamp(s.End.Line))
	}
	if printColumn {
		if printLine {
			fmt.Fprint(f, ":")
		}
		fmt.Fprintf(f, "%d", clamp(s.End.Column))
	}
	if printOffset {
		fmt.Fprintf(f, "#%d", clamp(s.End.Offset))
	}
}

// CleanOffset returns a copy of the Span with the Offset field updated.
// If the field is missing and Offsets is supplied it will be used to
// calculate it from the line and column.
// The value will then be adjusted to the canonical form.
func (s Span) CleanOffset(offsets Offsets) Span {
	if offsets != nil {
		if (s.Start.Line > 1 || s.Start.Column > 1) && s.Start.Offset == 0 {
			s.Start.updateOffset(offsets)
		}
		if (s.End.Line > 1 || s.End.Column > 1) && s.End.Offset == 0 {
			s.End.updateOffset(offsets)
		}
	}
	if s.Start.Offset < 0 {
		s.Start.Offset = 0
	}
	if s.End.Offset <= s.Start.Offset {
		s.End.Offset = s.Start.Offset
	}
	return s
}

// CleanCoords returns a copy of the Span with the Line and Column fields
// cleaned.
// If the fields are missing and Coords is supplied it will be used to
// calculate them from the offset.
// The values will then be adjusted to the canonical form.
func (s Span) CleanCoords(coords Coords) Span {
	if coords != nil {
		if s.Start.Line == 0 && s.Start.Offset > 0 {
			s.Start.Line, s.Start.Column = coords.ToCoord(s.Start.Offset)
		}
		if s.End.Line == 0 && s.End.Offset > 0 {
			s.End.Line, s.End.Column = coords.ToCoord(s.End.Offset)
		}
	}
	if s.Start.Line <= 0 {
		s.Start.Line = 0
	}
	if s.Start.Line == 0 {
		s.Start.Column = 0
	} else if s.Start.Column <= 0 {
		s.Start.Column = 0
	}
	if s.End.Line < s.Start.Line {
		s.End.Line = s.Start.Line
	}
	if s.End.Column < s.Start.Column {
		s.End.Column = s.Start.Column
	}
	if s.Start.Column <= 1 && s.End.Column <= 1 {
		s.Start.Column = 0
		s.End.Column = 0
	}
	if s.Start.Line <= 1 && s.End.Line <= 1 && s.Start.Column <= 1 && s.End.Column <= 1 {
		s.Start.Line = 0
		s.End.Line = 0
	}
	return s
}

// Clean returns a copy of the Span fully normalized.
// If passed a converter, it will use it to fill in any missing fields by
// converting between offset and line column fields.
// It does not attempt to validate that already filled fields have consistent
// values.
func (s Span) Clean(converter Converter) Span {
	s = s.CleanOffset(converter)
	s = s.CleanCoords(converter)
	if s.End.Offset == 0 {
		// in case CleanCoords adjusted the end position
		s.End.Offset = s.Start.Offset
	}
	return s
}

// IsPoint returns true if the span represents a single point.
// It is only valid on spans that are "clean".
func (s Span) IsPoint() bool {
	return s.Start == s.End
}

func (p *Point) updateOffset(offsets Offsets) {
	p.Offset = 0
	if p.Line <= 0 {
		return
	}
	c := p.Column
	if c < 1 {
		c = 1
	}
	if p.Line == 1 && c == 1 {
		return
	}
	p.Offset = offsets.ToOffset(p.Line, c)
}

func clamp(v int) int {
	if v < 0 {
		return 0
	}
	return v
}

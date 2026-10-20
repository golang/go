// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"fmt"
)

// EditMapVersion is the only edit map version this package reads.
const EditMapVersion = 1

// EditSpan is one applied edit of a lowering: the bytes of the original
// text from Start up to End became Len bytes of the lowered text. An
// insertion has Start equal to End.
type EditSpan struct {
	Start int `json:"start"`
	End   int `json:"end"`
	Len   int `json:"len"`
}

// EditMap is the JSON document the compiler writes beside each lowered
// file (cmd/compile/internal/logos.EditMap). Edits are in source order and
// do not overlap. Size and Out are the byte sizes of the original and
// lowered texts.
type EditMap struct {
	Version int        `json:"version"`
	Source  string     `json:"source"`
	Size    int        `json:"size"`
	Out     int        `json:"out"`
	Edits   []EditSpan `json:"edits"`
}

// edge says which side of an insertion a boundary offset belongs to. A
// leading offset is the start of a token, and sits after text inserted at
// that offset; a trailing offset is the end of a token, and sits before it.
type edge int

const (
	leading edge = iota
	trailing
)

// posMap translates positions between an original loGo/s text and its
// lowered Go text. It rests on the edit spans of the lowering, refined so
// that text the lowering copied unchanged inside a replaced span (the
// operand of a ? statement, say) still maps exactly.
type posMap struct {
	orig, low *text
	spans     []EditSpan
}

// newPosMap checks m against the two texts and builds the map.
func newPosMap(src, out []byte, m EditMap) (*posMap, error) {
	if err := checkEditMap(src, out, m); err != nil {
		return nil, err
	}
	return &posMap{orig: newText(src), low: newText(out), spans: refineSpans(src, out, m.Edits)}, nil
}

// checkEditMap rejects a map that cannot describe the transformation from
// src to out: another version, other sizes, or spans that are unordered,
// overlap, leave the text, or do not add up to the lowered size.
func checkEditMap(src, out []byte, m EditMap) error {
	if m.Version != EditMapVersion {
		return fmt.Errorf("edit map for %s has unsupported version %d", m.Source, m.Version)
	}
	if m.Size != len(src) || m.Out != len(out) {
		return fmt.Errorf("edit map for %s does not match its texts", m.Source)
	}
	prev, delta := 0, 0
	for _, s := range m.Edits {
		if !s.fits(prev, len(src)) {
			return fmt.Errorf("edit map for %s holds the invalid span %d:%d", m.Source, s.Start, s.End)
		}
		prev = s.End
		delta += s.Len - (s.End - s.Start)
	}
	if len(src)+delta != len(out) {
		return fmt.Errorf("edit map for %s does not add up to the lowered size", m.Source)
	}
	return nil
}

// fits reports whether s starts at or after prev, the end of the span
// before it, ends inside a text of size bytes, and does not start after it
// ends.
func (s EditSpan) fits(prev, size int) bool {
	return s.Start >= prev && s.End >= s.Start && s.End <= size && s.Len >= 0
}

// lowOffset maps an offset of the original text to the lowered text. An
// offset inside replaced text maps to the start of its replacement, or to
// the end of it on the trailing edge.
func (m *posMap) lowOffset(o int, e edge) int {
	shift := 0
	for _, s := range m.spans {
		switch {
		case o < s.Start:
			return o + shift
		case s.Start == s.End:
			if o == s.Start && e == trailing {
				return o + shift
			}
		case o < s.End:
			return s.lowered(o, shift, e)
		}
		shift += s.Len - (s.End - s.Start)
	}
	return o + shift
}

// lowered maps an offset o inside the replaced bytes of s, which sit at
// shift, to the replacement text.
func (s EditSpan) lowered(o, shift int, e edge) int {
	start := s.Start + shift
	if e == trailing && o > s.Start {
		return start + s.Len
	}
	return start
}

// origOffset maps an offset of the lowered text to the original text. An
// offset inside generated text maps to the start of the replaced original
// text, or to the end of it on the trailing edge. At the seam of a
// deletion, the trailing edge stays before the deleted text and the
// leading edge goes after it.
func (m *posMap) origOffset(o int, e edge) int {
	shift := 0
	for _, s := range m.spans {
		start := s.Start + shift
		switch {
		case o < start:
			return o - shift
		case o < start+s.Len:
			return s.original(o, start, e)
		case s.Len == 0 && o == start && e == trailing:
			return s.Start
		}
		shift += s.Len - (s.End - s.Start)
	}
	return o - shift
}

// original maps an offset o inside the generated text of s, which starts
// at start, to the original text.
func (s EditSpan) original(o, start int, e edge) int {
	if e == trailing && o > start {
		return s.End
	}
	return s.Start
}

// toLowered maps a position of the original text to the lowered text.
func (m *posMap) toLowered(p Position, e edge) Position {
	return m.low.position(m.lowOffset(m.orig.offset(p), e))
}

// toOriginal maps a position of the lowered text to the original text.
func (m *posMap) toOriginal(p Position, e edge) Position {
	return m.orig.position(m.origOffset(m.low.offset(p), e))
}

// rangeToLowered maps a range of the original text to the lowered text. It
// stays tight: the start sits after text inserted at its offset and the end
// before text inserted at its offset.
func (m *posMap) rangeToLowered(r Range) Range {
	start := m.lowOffset(m.orig.offset(r.Start), leading)
	end := max(start, m.lowOffset(m.orig.offset(r.End), trailing))
	return Range{Start: m.low.position(start), End: m.low.position(end)}
}

// rangeToOriginal maps a range of the lowered text to the original text. A
// range that touches generated text widens to the original text that
// produced it.
func (m *posMap) rangeToOriginal(r Range) Range {
	start := m.origOffset(m.low.offset(r.Start), leading)
	end := max(start, m.origOffset(m.low.offset(r.End), trailing))
	return Range{Start: m.orig.position(start), End: m.orig.position(end)}
}

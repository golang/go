// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"sort"
	"unicode/utf8"

	"cmd/internal/logos/editmap"
)

type Position struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

type Range struct {
	Start Position `json:"start"`
	End   Position `json:"end"`
}

type text struct {
	data   []byte
	starts []int
}

func newText(data []byte) *text {
	starts := []int{0}
	for i, c := range data {
		if c == '\n' {
			starts = append(starts, i+1)
		}
	}
	return &text{data: data, starts: starts}
}

func (t *text) lineEnd(line int) int {
	end := len(t.data)
	if line+1 < len(t.starts) {
		end = t.starts[line+1] - 1
	}
	if end > t.starts[line] && t.data[end-1] == '\r' {
		end--
	}
	return end
}

func (t *text) offset(p Position) int {
	if p.Line < 0 {
		return 0
	}
	if p.Line >= len(t.starts) {
		return len(t.data)
	}
	off, end := t.starts[p.Line], t.lineEnd(p.Line)
	for units := 0; off < end && units < p.Character; {
		r, size := utf8.DecodeRune(t.data[off:end])
		units += utf16Len(r)
		off += size
	}
	return off
}

func (t *text) position(off int) Position {
	off = min(max(off, 0), len(t.data))
	line := sort.SearchInts(t.starts, off+1) - 1
	return Position{Line: line, Character: utf16Units(t.data[t.starts[line]:off])}
}

func (t *text) lineCol(line, col int) (int, bool) {
	if line < 1 || line > len(t.starts) || col < 1 {
		return 0, false
	}
	off := t.starts[line-1] + col - 1
	if off > t.lineEnd(line-1) {
		return 0, false
	}
	return off, true
}

func (t *text) replace(start, end int, repl string) *text {
	data := make([]byte, 0, len(t.data)-(end-start)+len(repl))
	data = append(data, t.data[:start]...)
	data = append(data, repl...)
	data = append(data, t.data[end:]...)
	return newText(data)
}

func utf16Len(r rune) int {
	if r >= 0x10000 {
		return 2
	}
	return 1
}

func utf16Units(b []byte) int {
	n := 0
	for len(b) > 0 {
		r, size := utf8.DecodeRune(b)
		n += utf16Len(r)
		b = b[size:]
	}
	return n
}

type posMap struct {
	*editmap.Map
	orig, low *text
}

func newPosMap(src, out []byte, m editmap.EditMap) (*posMap, error) {
	em, err := editmap.New(src, out, m)
	if err != nil {
		return nil, err
	}
	return &posMap{Map: em, orig: newText(src), low: newText(out)}, nil
}

func (m *posMap) toLowered(p Position) Position {
	o := m.orig.offset(p)
	if v, ok := m.LoweredOffset(o, editmap.Leading); ok {
		return m.low.position(v)
	}
	return m.low.position(m.LoweredOffsetNear(o, editmap.Leading))
}

func (m *posMap) toOriginal(p Position) Position {
	o, _ := m.OriginalOffset(m.low.offset(p), editmap.Leading)
	return m.orig.position(o)
}

func (m *posMap) rangeToLowered(r Range) Range {
	start := m.LoweredOffsetNear(m.orig.offset(r.Start), editmap.Leading)
	end := max(start, m.LoweredOffsetNear(m.orig.offset(r.End), editmap.Trailing))
	return Range{Start: m.low.position(start), End: m.low.position(end)}
}

func (m *posMap) rangeToOriginal(r Range) Range {
	loStart := m.low.offset(r.Start)
	loEnd := m.low.offset(r.End)
	if loEnd < loStart {
		loEnd = loStart
	}
	start, _ := m.OriginalOffset(loStart, editmap.Leading)
	end, _ := m.OriginalOffset(loEnd, editmap.Trailing)
	if loStart < loEnd {
		start, end = m.WidenOverGenerated(loStart, loEnd, start, end)
	}
	end = max(start, end)
	return Range{Start: m.orig.position(start), End: m.orig.position(end)}
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"sort"
	"unicode/utf8"
)

// Position is an LSP position: a zero-based line and a character offset
// counted in UTF-16 code units. The proxy forces both of its peers onto
// this encoding, so it is the only one it converts.
type Position struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

// Range is an LSP range: Start is inclusive and End is exclusive.
type Range struct {
	Start Position `json:"start"`
	End   Position `json:"end"`
}

// text is an immutable document text with a line index. Lines end at "\n";
// a "\r" before it belongs to the line.
type text struct {
	data   []byte
	starts []int
}

// newText indexes data. The text keeps data without copying it.
func newText(data []byte) *text {
	starts := []int{0}
	for i, c := range data {
		if c == '\n' {
			starts = append(starts, i+1)
		}
	}
	return &text{data: data, starts: starts}
}

// lineEnd returns the offset where the content of line ends.
func (t *text) lineEnd(line int) int {
	if line+1 < len(t.starts) {
		return t.starts[line+1] - 1
	}
	return len(t.data)
}

// offset converts an LSP position to a byte offset. A line past the end
// maps to the end of the text, and a character past the end of its line
// maps to the end of the line, as the protocol requires.
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

// position converts a byte offset to an LSP position. An offset outside
// the text is clamped to it.
func (t *text) position(off int) Position {
	off = min(max(off, 0), len(t.data))
	line := sort.SearchInts(t.starts, off+1) - 1
	return Position{Line: line, Character: utf16Units(t.data[t.starts[line]:off])}
}

// lineCol converts a 1-based line and 1-based byte column, the position
// form of compiler diagnostics, to a byte offset. It reports false for a
// line or column outside the text.
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

// replace returns the text with the bytes from start up to end replaced
// by repl.
func (t *text) replace(start, end int, repl string) *text {
	data := make([]byte, 0, len(t.data)-(end-start)+len(repl))
	data = append(data, t.data[:start]...)
	data = append(data, repl...)
	data = append(data, t.data[end:]...)
	return newText(data)
}

// utf16Len returns the number of UTF-16 code units that encode r.
func utf16Len(r rune) int {
	if r >= 0x10000 {
		return 2
	}
	return 1
}

// utf16Units returns the number of UTF-16 code units that encode b. An
// invalid byte counts as one unit, as the replacement character does.
func utf16Units(b []byte) int {
	n := 0
	for len(b) > 0 {
		r, size := utf8.DecodeRune(b)
		n += utf16Len(r)
		b = b[size:]
	}
	return n
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"sort"
	"unicode/utf16"
)

// lineIndex converts between the compiler's positions (1-based line and byte
// column, byte offsets) and protocol positions (0-based line and UTF-16
// character) of one file.
type lineIndex struct {
	src    []byte
	starts []int
}

// newLineIndex indexes the lines of src.
func newLineIndex(src []byte) lineIndex {
	return lineIndex{src: src, starts: lineStarts(src)}
}

// offsetOf returns the byte offset of the 1-based line and byte column.
func (x lineIndex) offsetOf(line, col int) (int, bool) {
	off, err := lineColOffset(x.starts, len(x.src), line, col)
	return off, err == nil
}

// pos returns the protocol position of byte offset off.
func (x lineIndex) pos(off int) HintPos {
	line := sort.SearchInts(x.starts, off+1) - 1
	return HintPos{Line: line, Character: utf16Len(x.src[x.starts[line]:off])}
}

// utf16Len returns the number of UTF-16 code units of the UTF-8 text b.
func utf16Len(b []byte) int {
	n := 0
	for _, r := range string(b) {
		n += utf16.RuneLen(r)
	}
	return n
}

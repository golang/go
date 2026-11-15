// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"sort"
	"unicode/utf16"
)

type lineIndex struct {
	src    []byte
	starts []int
}

func newLineIndex(src []byte) lineIndex {
	return lineIndex{src: src, starts: lineStarts(src)}
}

func (x lineIndex) offsetOf(line, col int) (int, bool) {
	off, err := lineColOffset(x.starts, len(x.src), line, col)
	return off, err == nil
}

func (x lineIndex) pos(off int) HintPos {
	line := sort.SearchInts(x.starts, off+1) - 1
	return HintPos{Line: line, Character: utf16Len(x.src[x.starts[line]:off])}
}

func utf16Len(b []byte) int {
	n := 0
	for _, r := range string(b) {
		n += utf16.RuneLen(r)
	}
	return n
}

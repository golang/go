// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hints

import (
	"bytes"
	"sort"
	"unicode/utf16"
)

type lineIndex struct {
	src    []byte
	starts []int
}

func newLineIndex(src []byte) lineIndex {
	starts := []int{0}
	for off := 0; ; {
		i := bytes.IndexByte(src[off:], '\n')
		if i < 0 {
			break
		}
		off += i + 1
		starts = append(starts, off)
	}
	return lineIndex{src: src, starts: starts}
}

func (x lineIndex) position(line, col int) (Position, bool) {
	if line < 1 || line > len(x.starts) || col < 1 {
		return Position{}, false
	}
	end := len(x.src)
	if line < len(x.starts) {
		end = x.starts[line] - 1
	}
	off := x.starts[line-1] + col - 1
	if off > end {
		return Position{}, false
	}
	n := sort.SearchInts(x.starts, off+1) - 1
	return Position{Line: n, Character: utf16Len(x.src[x.starts[n]:off])}, true
}

func utf16Len(b []byte) int {
	n := 0
	for _, r := range string(b) {
		n += utf16.RuneLen(r)
	}
	return n
}

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span

import (
	"fmt"
	"unicode/utf8"
)

// ToUTF16Column calculates the utf16 column expressed by the point given the
// supplied file contents.
// This is used to convert from the native (always in bytes) column
// representation and the utf16 counts used by some editors.
//
// TODO(adonovan): this function is unused except by its test. Delete,
// or consolidate with (*protocol.ColumnMapper).utf16Column.
func ToUTF16Column(p Point, content []byte) (int, error) {
	if !p.HasPosition() {
		return -1, fmt.Errorf("ToUTF16Column: point is missing position")
	}
	if !p.HasOffset() {
		return -1, fmt.Errorf("ToUTF16Column: point is missing offset")
	}
	offset := p.Offset()      // 0-based
	colZero := p.Column() - 1 // 0-based
	if colZero == 0 {
		// 0-based column 0, so it must be chr 1
		return 1, nil
	} else if colZero < 0 {
		return -1, fmt.Errorf("ToUTF16Column: column is invalid (%v)", colZero)
	}
	// work out the offset at the start of the line using the column
	lineOffset := offset - colZero
	if lineOffset < 0 || offset > len(content) {
		return -1, fmt.Errorf("ToUTF16Column: offsets %v-%v outside file contents (%v)", lineOffset, offset, len(content))
	}
	// Use the offset to pick out the line start.
	// This cannot panic: offset > len(content) and lineOffset < offset.
	start := content[lineOffset:]

	// Now, truncate down to the supplied column.
	start = start[:colZero]

	cnt := 0
	for _, r := range string(start) {
		cnt++
		if r > 0xffff {
			cnt++
		}
	}
	return cnt + 1, nil // the +1 is for 1-based columns
}

// FromUTF16Column advances the point by the utf16 character offset given the
// supplied line contents.
// This is used to convert from the utf16 counts used by some editors to the
// native (always in bytes) column representation.
//
// The resulting Point always has an offset.
//
// TODO: it looks like this may incorrectly confer a "position" to the
// resulting Point, when it shouldn't. If p.HasPosition() == false, the
// resulting Point will return p.HasPosition() == true, but have the wrong
// position.
func FromUTF16Column(p Point, chr int, content []byte) (Point, error) {
	if !p.HasOffset() {
		return Point{}, fmt.Errorf("FromUTF16Column: point is missing offset")
	}
	// if chr is 1 then no adjustment needed
	if chr <= 1 {
		return p, nil
	}
	if p.Offset() >= len(content) {
		return p, fmt.Errorf("FromUTF16Column: offset (%v) greater than length of content (%v)", p.Offset(), len(content))
	}
	remains := content[p.Offset():]
	// scan forward the specified number of characters
	for count := 1; count < chr; count++ {
		if len(remains) <= 0 {
			return Point{}, fmt.Errorf("FromUTF16Column: chr goes beyond the content")
		}
		r, w := utf8.DecodeRune(remains)
		if r == '\n' {
			// Per the LSP spec:
			//
			// > If the character value is greater than the line length it
			// > defaults back to the line length.
			break
		}
		remains = remains[w:]
		if r >= 0x10000 {
			// a two point rune
			count++
			// if we finished in a two point rune, do not advance past the first
			if count >= chr {
				break
			}
		}
		p.v.Column += w
		p.v.Offset += w
	}
	return p, nil
}

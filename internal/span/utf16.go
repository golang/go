// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span

import (
	"fmt"
	"unicode/utf16"
	"unicode/utf8"
)

// ToUTF16Column calculates the utf16 column expressed by the point given the
// supplied file contents.
// This is used to convert from the native (always in bytes) column
// representation and the utf16 counts used by some editors.
func ToUTF16Column(p Point, content []byte) (int, error) {
	if content == nil {
		return -1, fmt.Errorf("ToUTF16Column: missing content")
	}
	if !p.HasPosition() {
		return -1, fmt.Errorf("ToUTF16Column: point is missing position")
	}
	if !p.HasOffset() {
		return -1, fmt.Errorf("ToUTF16Column: point is missing offset")
	}
	offset := p.Offset()
	col := p.Column()
	if col == 1 {
		// column 1, so it must be chr 1
		return 1, nil
	}
	// work out the offset at the start of the line using the column
	lineOffset := offset - (col - 1)
	if lineOffset < 0 || offset > len(content) {
		return -1, fmt.Errorf("ToUTF16Column: offsets %v-%v outside file contents (%v)", lineOffset, offset, len(content))
	}
	// Use the offset to pick out the line start.
	// This cannot panic: offset > len(content) and lineOffset < offset.
	start := content[lineOffset:]

	// Now, truncate down to the supplied column.
	start = start[:col]
	// and count the number of utf16 characters
	// in theory we could do this by hand more efficiently...
	return len(utf16.Encode([]rune(string(start)))), nil
}

// FromUTF16Column advances the point by the utf16 character offset given the
// supplied line contents.
// This is used to convert from the utf16 counts used by some editors to the
// native (always in bytes) column representation.
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
			return Point{}, fmt.Errorf("FromUTF16Column: chr goes beyond the line")
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

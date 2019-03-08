// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span

import (
	"unicode/utf16"
	"unicode/utf8"
)

// ToUTF16Column calculates the utf16 column expressed by the point given the
// supplied file contents.
// This is used to convert from the native (always in bytes) column
// representation and the utf16 counts used by some editors.
func ToUTF16Column(offsets Offsets, p Point, content []byte) int {
	if content == nil {
		return -1
	}
	// make sure we have a valid offset
	p.updateOffset(offsets)
	lineOffset := p.Offset - (p.Column - 1)
	if lineOffset < 0 || p.Offset > len(content) {
		return -1
	}
	// use the offset to pick out the line start
	start := content[lineOffset:]
	// now truncate down to the supplied column
	start = start[:p.Column]
	// and count the number of utf16 characters
	// in theory we could do this by hand more efficiently...
	return len(utf16.Encode([]rune(string(start))))
}

// FromUTF16Column calculates the byte column expressed by the utf16 character
// offset given the supplied file contents.
// This is used to convert from the utf16 counts used by some editors to the
// native (always in bytes) column representation.
func FromUTF16Column(offsets Offsets, line, chr int, content []byte) Point {
	// first build a point for the start of the line the normal way
	p := Point{Line: line, Column: 1, Offset: 0}
	// now use that to work out the byte offset of the start of the line
	p.updateOffset(offsets)
	if chr <= 1 {
		return p
	}
	// use that to pick the line out of the file content
	remains := content[p.Offset:]
	// and now scan forward the specified number of characters
	for count := 1; count < chr; count++ {
		if len(remains) <= 0 {
			return Point{Offset: -1}
		}
		r, w := utf8.DecodeRune(remains)
		if r == '\n' {
			return Point{Offset: -1}
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
		p.Column += w
		p.Offset += w
	}
	return p
}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"fmt"
	"unicode/utf8"

	"golang.org/x/tools/gopls/internal/span"
)

func URIFromSpanURI(uri span.URI) DocumentURI {
	return DocumentURI(uri) // simple conversion
}

func URIFromPath(path string) DocumentURI {
	return URIFromSpanURI(span.URIFromPath(path)) // normalizing conversion
}

func (u DocumentURI) SpanURI() span.URI {
	return span.URIFromURI(string(u)) // normalizing conversion
}

func IsPoint(r Range) bool {
	return r.Start.Line == r.End.Line && r.Start.Character == r.End.Character
}

// CompareLocation defines a three-valued comparison over locations,
// lexicographically ordered by (URI, Range).
func CompareLocation(x, y Location) int {
	if x.URI != y.URI {
		if x.URI < y.URI {
			return -1
		} else {
			return +1
		}
	}
	return CompareRange(x.Range, y.Range)
}

// CompareRange returns -1 if a is before b, 0 if a == b, and 1 if a is after b.
//
// A range a is defined to be 'before' b if a.Start is before b.Start, or
// a.Start == b.Start and a.End is before b.End.
func CompareRange(a, b Range) int {
	if r := ComparePosition(a.Start, b.Start); r != 0 {
		return r
	}
	return ComparePosition(a.End, b.End)
}

// ComparePosition returns -1 if a is before b, 0 if a == b, and 1 if a is after b.
func ComparePosition(a, b Position) int {
	if a.Line != b.Line {
		if a.Line < b.Line {
			return -1
		} else {
			return +1
		}
	}
	if a.Character != b.Character {
		if a.Character < b.Character {
			return -1
		} else {
			return +1
		}
	}
	return 0
}

func Intersect(a, b Range) bool {
	if a.Start.Line > b.End.Line || a.End.Line < b.Start.Line {
		return false
	}
	return !((a.Start.Line == b.End.Line) && a.Start.Character > b.End.Character ||
		(a.End.Line == b.Start.Line) && a.End.Character < b.Start.Character)
}

// Format implements fmt.Formatter.
//
// Note: Formatter is implemented instead of Stringer (presumably) for
// performance reasons, though it is not clear that it matters in practice.
func (r Range) Format(f fmt.State, _ rune) {
	fmt.Fprintf(f, "%v-%v", r.Start, r.End)
}

// Format implements fmt.Formatter.
//
// See Range.Format for discussion of why the Formatter interface is
// implemented rather than Stringer.
func (p Position) Format(f fmt.State, _ rune) {
	fmt.Fprintf(f, "%v:%v", p.Line, p.Character)
}

// -- implementation helpers --

// UTF16Len returns the number of codes in the UTF-16 transcoding of s.
func UTF16Len(s []byte) int {
	var n int
	for len(s) > 0 {
		n++

		// Fast path for ASCII.
		if s[0] < 0x80 {
			s = s[1:]
			continue
		}

		r, size := utf8.DecodeRune(s)
		if r >= 0x10000 {
			n++ // surrogate pair
		}
		s = s[size:]
	}
	return n
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package provides data and functions to test some properties of Unicode code points.
// It is rudimentary but will improve.
package unicode

// The representation of a range of Unicode code points.  The range runs from Lo to Hi
// inclusive and has the specified stride.
type Range struct {
	Lo int;
	Hi int;
	Stride int;
}

// Is tests whether rune is in the specified table of ranges.
func Is(ranges []Range, rune int) bool {
	// common case: rune is ASCII or Latin-1
	if rune < 0x100 {
		for i, r := range ranges {
			if rune > r.Hi {
				continue;
			}
			if rune < r.Lo {
				return false;
			}
			return (rune - r.Lo) % r.Stride == 0;
		}
		return false;
	}

	// binary search over ranges
	lo := 0;
	hi := len(ranges);
	for lo < hi {
		m := lo + (hi - lo)/2;
		r := ranges[m];
		if r.Lo <= rune && rune <= r.Hi {
			return (rune - r.Lo) % r.Stride == 0;
		}
		if rune < r.Lo {
			hi = m;
		} else {
			lo = m+1;
		}
	}
	return false;
}

// IsUpper reports whether the rune is an upper case letter.
func IsUpper(rune int) bool {
	if rune < 0x80 {	// quick ASCII check
		return 'A' <= rune && rune <= 'Z';
	}
	return Is(Upper, rune);
}

// IsLower reports whether the rune is a lower case letter.
func IsLower(rune int) bool {
	if rune < 0x80 {	// quick ASCII check
		return 'a' <= rune && rune <= 'z';
	}
	return Is(Lower, rune);
}

// IsTitle reports whether the rune is a title case letter.
func IsTitle(rune int) bool {
	if rune < 0x80 {	// quick ASCII check
		return false;
	}
	return Is(Title, rune);
}

// IsLetter reports whether the rune is a letter.
func IsLetter(rune int) bool {
	if rune < 0x80 {	// quick ASCII check
		rune &^= ' ';
		return 'A' <= rune && rune <= 'Z';
	}
	return Is(Letter, rune);
}

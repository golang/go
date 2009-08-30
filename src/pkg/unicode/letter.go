// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package provides data and functions to test some properties of Unicode code points.
// It is rudimentary but will improve.
package unicode

// The representation of a range of Unicode code points.  The range runs from Lo to Hi
// inclusive and has the specified stride.
type Range struct {
	Lo	int;
	Hi	int;
	Stride	int;
}

// The representation of a range of Unicode code points for case conversion.
// The range runs from Lo to Hi inclusive, with a fixed stride of 1.  Deltas
// are the number to add to the code point to reach the code point for a
// different case for that character.  They may be negative.  If zero, it
// means the character is in the corresponding case. There is a special
// case representing sequences of alternating corresponding Upper and Lower
// pairs.  It appears with the usual Lo and Hi values and a Delta of
//	{0, UpperLower, 0}
// The constant UpperLower has (meaningful) value 1.  The lower case
// letters in such sequences are assumed; were they present they would
// have a Delta of
//	{LowerUpper, 0, LowerUpper}
// where LowerUpper has value -1.
type CaseRange struct {
	Lo	int;
	Hi	int;
	Delta	d;
}

// Indices into the Delta arrays inside CaseRanges for case mapping.
const (
	UpperCase = iota;
	LowerCase;
	TitleCase;
	MaxCase;
)
type d [MaxCase]int32	// to make the CaseRanges text shorter

// If the Delta field of a CaseRange is UpperLower or LowerUpper, it means
// this CaseRange represents a sequence of the form (say)
// Upper Lower Upper Lower.
const (
	MaxChar		= 0x10FFFF;
	UpperLower      = MaxChar + 2;	// cannot be a valid delta
	LowerUpper	= MaxChar + 3;
)

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
		rune &^= 'a'-'A';
		return 'A' <= rune && rune <= 'Z';
	}
	return Is(Letter, rune);
}

// In an Upper-Lower sequence, which always starts with an UpperCase letter,
// the real deltas always look like:
//	0 1 0
//	-1 0 -1
// This is a single-dimensioned array addressed by the case shifted up one bit
// (the column of this table) or'ed with the low bit of the position in
// the sequence (the row of the table).
var ulDelta = [8]int{
	(UpperCase<<1) | 0: 0,
	(UpperCase<<1) | 1: -1,
	(LowerCase<<1) | 0: 1,
	(LowerCase<<1) | 1: 0,
	(TitleCase<<1) | 0: 0,
	(TitleCase<<1) | 1: -1,
}

// To maps the rune to the specified case, UpperCase, LowerCase, or TitleCase
func To(_case int, rune int) int {
	if _case < 0 || MaxCase <= _case {
		return 0xFFFD	// as reasonable an error as any
	}
	// binary search over ranges
	lo := 0;
	hi := len(CaseRanges);
	for lo < hi {
		m := lo + (hi - lo)/2;
		r := CaseRanges[m];
		if r.Lo <= rune && rune <= r.Hi {
			delta := int(r.Delta[_case]);
			if delta > MaxChar {
				// Somewhere inside an UpperLower sequence. Use
				// the precomputed delta table to get our offset.
				delta = ulDelta[((_case<<1) | ((rune-r.Lo)&1))];
			}
			return rune + delta;
		}
		if rune < r.Lo {
			hi = m;
		} else {
			lo = m+1;
		}
	}
	return rune;
}

// ToUpper maps the rune to upper case
func ToUpper(rune int) int {
	if rune < 0x80 {	// quick ASCII check
		if 'a' <= rune && rune <= 'z' {
			rune -= 'a'-'A'
		}
		return rune
	}
	return To(UpperCase, rune);
}

// ToLower maps the rune to lower case
func ToLower(rune int) int {
	if rune < 0x80 {	// quick ASCII check
		if 'A' <= rune && rune <= 'Z' {
			rune += 'a'-'A'
		}
		return rune
	}
	return To(LowerCase, rune);
}

// ToTitle maps the rune to title case
func ToTitle(rune int) int {
	if rune < 0x80 {	// quick ASCII check
		if 'a' <= rune && rune <= 'z' {	// title case is upper case for ASCII
			rune -= 'a'-'A'
		}
		return rune
	}
	return To(TitleCase, rune);
}

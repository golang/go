// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package provides data and functions to test some properties of Unicode code points.
package unicode

const (
	MaxRune         = 0x10FFFF // Maximum valid Unicode code point.
	ReplacementChar = 0xFFFD   // Represents invalid code points.
)


// The representation of a range of Unicode code points.  The range runs from Lo to Hi
// inclusive and has the specified stride.
type Range struct {
	Lo     int
	Hi     int
	Stride int
}

// CaseRange represents a range of Unicode code points for simple (one
// code point to one code point) case conversion.
// The range runs from Lo to Hi inclusive, with a fixed stride of 1.  Deltas
// are the number to add to the code point to reach the code point for a
// different case for that character.  They may be negative.  If zero, it
// means the character is in the corresponding case. There is a special
// case representing sequences of alternating corresponding Upper and Lower
// pairs.  It appears with a fixed Delta of
//	{UpperLower, UpperLower, UpperLower}
// The constant UpperLower has an otherwise impossible delta value.
type CaseRange struct {
	Lo    int
	Hi    int
	Delta d
}

// SpecialCase represents language-specific case mappings such as Turkish.
// Methods of SpecialCase customize (by overriding) the standard mappings.
type SpecialCase []CaseRange

//BUG(r): Provide a mechanism for full case folding (those that involve
// multiple runes in the input or output).

// Indices into the Delta arrays inside CaseRanges for case mapping.
const (
	UpperCase = iota
	LowerCase
	TitleCase
	MaxCase
)

type d [MaxCase]int32 // to make the CaseRanges text shorter

// If the Delta field of a CaseRange is UpperLower or LowerUpper, it means
// this CaseRange represents a sequence of the form (say)
// Upper Lower Upper Lower.
const (
	UpperLower = MaxRune + 1 // (Cannot be a valid delta.)
)

// Is tests whether rune is in the specified table of ranges.
func Is(ranges []Range, rune int) bool {
	// common case: rune is ASCII or Latin-1
	if rune < 0x100 {
		for _, r := range ranges {
			if rune > r.Hi {
				continue
			}
			if rune < r.Lo {
				return false
			}
			return (rune-r.Lo)%r.Stride == 0
		}
		return false
	}

	// binary search over ranges
	lo := 0
	hi := len(ranges)
	for lo < hi {
		m := lo + (hi-lo)/2
		r := ranges[m]
		if r.Lo <= rune && rune <= r.Hi {
			return (rune-r.Lo)%r.Stride == 0
		}
		if rune < r.Lo {
			hi = m
		} else {
			lo = m + 1
		}
	}
	return false
}

// IsUpper reports whether the rune is an upper case letter.
func IsUpper(rune int) bool {
	if rune < 0x80 { // quick ASCII check
		return 'A' <= rune && rune <= 'Z'
	}
	return Is(Upper, rune)
}

// IsLower reports whether the rune is a lower case letter.
func IsLower(rune int) bool {
	if rune < 0x80 { // quick ASCII check
		return 'a' <= rune && rune <= 'z'
	}
	return Is(Lower, rune)
}

// IsTitle reports whether the rune is a title case letter.
func IsTitle(rune int) bool {
	if rune < 0x80 { // quick ASCII check
		return false
	}
	return Is(Title, rune)
}

// IsLetter reports whether the rune is a letter.
func IsLetter(rune int) bool {
	if rune < 0x80 { // quick ASCII check
		rune &^= 'a' - 'A'
		return 'A' <= rune && rune <= 'Z'
	}
	return Is(Letter, rune)
}

// IsSpace reports whether the rune is a white space character.
func IsSpace(rune int) bool {
	if rune <= 0xFF { // quick Latin-1 check
		switch rune {
		case '\t', '\n', '\v', '\f', '\r', ' ', 0x85, 0xA0:
			return true
		}
		return false
	}
	return Is(White_Space, rune)
}

// to maps the rune using the specified case mapping.
func to(_case int, rune int, caseRange []CaseRange) int {
	if _case < 0 || MaxCase <= _case {
		return ReplacementChar // as reasonable an error as any
	}
	// binary search over ranges
	lo := 0
	hi := len(caseRange)
	for lo < hi {
		m := lo + (hi-lo)/2
		r := caseRange[m]
		if r.Lo <= rune && rune <= r.Hi {
			delta := int(r.Delta[_case])
			if delta > MaxRune {
				// In an Upper-Lower sequence, which always starts with
				// an UpperCase letter, the real deltas always look like:
				//	{0, 1, 0}    UpperCase (Lower is next)
				//	{-1, 0, -1}  LowerCase (Upper, Title are previous)
				// The characters at even offsets from the beginning of the
				// sequence are upper case; the ones at odd offsets are lower.
				// The correct mapping can be done by clearing or setting the low
				// bit in the sequence offset.
				// The constants UpperCase and TitleCase are even while LowerCase
				// is odd so we take the low bit from _case.
				return r.Lo + ((rune-r.Lo)&^1 | _case&1)
			}
			return rune + delta
		}
		if rune < r.Lo {
			hi = m
		} else {
			lo = m + 1
		}
	}
	return rune
}

// To maps the rune to the specified case: UpperCase, LowerCase, or TitleCase.
func To(_case int, rune int) int {
	return to(_case, rune, CaseRanges)
}

// ToUpper maps the rune to upper case.
func ToUpper(rune int) int {
	if rune < 0x80 { // quick ASCII check
		if 'a' <= rune && rune <= 'z' {
			rune -= 'a' - 'A'
		}
		return rune
	}
	return To(UpperCase, rune)
}

// ToLower maps the rune to lower case.
func ToLower(rune int) int {
	if rune < 0x80 { // quick ASCII check
		if 'A' <= rune && rune <= 'Z' {
			rune += 'a' - 'A'
		}
		return rune
	}
	return To(LowerCase, rune)
}

// ToTitle maps the rune to title case.
func ToTitle(rune int) int {
	if rune < 0x80 { // quick ASCII check
		if 'a' <= rune && rune <= 'z' { // title case is upper case for ASCII
			rune -= 'a' - 'A'
		}
		return rune
	}
	return To(TitleCase, rune)
}

// ToUpper maps the rune to upper case giving priority to the special mapping.
func (special SpecialCase) ToUpper(rune int) int {
	r := to(UpperCase, rune, []CaseRange(special))
	if r == rune {
		r = ToUpper(rune)
	}
	return r
}

// ToTitle maps the rune to title case giving priority to the special mapping.
func (special SpecialCase) ToTitle(rune int) int {
	r := to(TitleCase, rune, []CaseRange(special))
	if r == rune {
		r = ToTitle(rune)
	}
	return r
}

// ToLower maps the rune to lower case giving priority to the special mapping.
func (special SpecialCase) ToLower(rune int) int {
	r := to(LowerCase, rune, []CaseRange(special))
	if r == rune {
		r = ToLower(rune)
	}
	return r
}

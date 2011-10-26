// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unicode provides data and functions to test some properties of
// Unicode code points.
package unicode

const (
	MaxRune         = 0x10FFFF // Maximum valid Unicode code point.
	ReplacementChar = 0xFFFD   // Represents invalid code points.
	MaxASCII        = 0x7F     // maximum ASCII value.
	MaxLatin1       = 0xFF     // maximum Latin-1 value.
)

// RangeTable defines a set of Unicode code points by listing the ranges of
// code points within the set. The ranges are listed in two slices
// to save space: a slice of 16-bit ranges and a slice of 32-bit ranges.
// The two slices must be in sorted order and non-overlapping.
// Also, R32 should contain only values >= 0x10000 (1<<16).
type RangeTable struct {
	R16 []Range16
	R32 []Range32
}

// Range16 represents of a range of 16-bit Unicode code points.  The range runs from Lo to Hi
// inclusive and has the specified stride.
type Range16 struct {
	Lo     uint16
	Hi     uint16
	Stride uint16
}

// Range32 represents of a range of Unicode code points and is used when one or
// more of the values will not fit in 16 bits.  The range runs from Lo to Hi
// inclusive and has the specified stride. Lo and Hi must always be >= 1<<16.
type Range32 struct {
	Lo     uint32
	Hi     uint32
	Stride uint32
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
	Lo    uint32
	Hi    uint32
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

type d [MaxCase]rune // to make the CaseRanges text shorter

// If the Delta field of a CaseRange is UpperLower or LowerUpper, it means
// this CaseRange represents a sequence of the form (say)
// Upper Lower Upper Lower.
const (
	UpperLower = MaxRune + 1 // (Cannot be a valid delta.)
)

// is16 uses binary search to test whether rune is in the specified slice of 16-bit ranges.
func is16(ranges []Range16, r uint16) bool {
	// binary search over ranges
	lo := 0
	hi := len(ranges)
	for lo < hi {
		m := lo + (hi-lo)/2
		range_ := ranges[m]
		if range_.Lo <= r && r <= range_.Hi {
			return (r-range_.Lo)%range_.Stride == 0
		}
		if r < range_.Lo {
			hi = m
		} else {
			lo = m + 1
		}
	}
	return false
}

// is32 uses binary search to test whether rune is in the specified slice of 32-bit ranges.
func is32(ranges []Range32, r uint32) bool {
	// binary search over ranges
	lo := 0
	hi := len(ranges)
	for lo < hi {
		m := lo + (hi-lo)/2
		range_ := ranges[m]
		if range_.Lo <= r && r <= range_.Hi {
			return (r-range_.Lo)%range_.Stride == 0
		}
		if r < range_.Lo {
			hi = m
		} else {
			lo = m + 1
		}
	}
	return false
}

// Is tests whether rune is in the specified table of ranges.
func Is(rangeTab *RangeTable, r rune) bool {
	// common case: rune is ASCII or Latin-1.
	if uint32(r) <= MaxLatin1 {
		// Only need to check R16, since R32 is always >= 1<<16.
		r16 := uint16(r)
		for _, r := range rangeTab.R16 {
			if r16 > r.Hi {
				continue
			}
			if r16 < r.Lo {
				return false
			}
			return (r16-r.Lo)%r.Stride == 0
		}
		return false
	}
	r16 := rangeTab.R16
	if len(r16) > 0 && r <= rune(r16[len(r16)-1].Hi) {
		return is16(r16, uint16(r))
	}
	r32 := rangeTab.R32
	if len(r32) > 0 && r >= rune(r32[0].Lo) {
		return is32(r32, uint32(r))
	}
	return false
}

// IsUpper reports whether the rune is an upper case letter.
func IsUpper(r rune) bool {
	// See comment in IsGraphic.
	if uint32(r) <= MaxLatin1 {
		return properties[uint8(r)]&pLu != 0
	}
	return Is(Upper, r)
}

// IsLower reports whether the rune is a lower case letter.
func IsLower(r rune) bool {
	// See comment in IsGraphic.
	if uint32(r) <= MaxLatin1 {
		return properties[uint8(r)]&pLl != 0
	}
	return Is(Lower, r)
}

// IsTitle reports whether the rune is a title case letter.
func IsTitle(r rune) bool {
	if r <= MaxLatin1 {
		return false
	}
	return Is(Title, r)
}

// to maps the rune using the specified case mapping.
func to(_case int, r rune, caseRange []CaseRange) rune {
	if _case < 0 || MaxCase <= _case {
		return ReplacementChar // as reasonable an error as any
	}
	// binary search over ranges
	lo := 0
	hi := len(caseRange)
	for lo < hi {
		m := lo + (hi-lo)/2
		cr := caseRange[m]
		if rune(cr.Lo) <= r && r <= rune(cr.Hi) {
			delta := rune(cr.Delta[_case])
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
				return rune(cr.Lo) + ((r-rune(cr.Lo))&^1 | rune(_case&1))
			}
			return r + delta
		}
		if r < rune(cr.Lo) {
			hi = m
		} else {
			lo = m + 1
		}
	}
	return r
}

// To maps the rune to the specified case: UpperCase, LowerCase, or TitleCase.
func To(_case int, r rune) rune {
	return to(_case, r, CaseRanges)
}

// ToUpper maps the rune to upper case.
func ToUpper(r rune) rune {
	if r <= MaxASCII {
		if 'a' <= r && r <= 'z' {
			r -= 'a' - 'A'
		}
		return r
	}
	return To(UpperCase, r)
}

// ToLower maps the rune to lower case.
func ToLower(r rune) rune {
	if r <= MaxASCII {
		if 'A' <= r && r <= 'Z' {
			r += 'a' - 'A'
		}
		return r
	}
	return To(LowerCase, r)
}

// ToTitle maps the rune to title case.
func ToTitle(r rune) rune {
	if r <= MaxASCII {
		if 'a' <= r && r <= 'z' { // title case is upper case for ASCII
			r -= 'a' - 'A'
		}
		return r
	}
	return To(TitleCase, r)
}

// ToUpper maps the rune to upper case giving priority to the special mapping.
func (special SpecialCase) ToUpper(r rune) rune {
	r1 := to(UpperCase, r, []CaseRange(special))
	if r1 == r {
		r1 = ToUpper(r)
	}
	return r1
}

// ToTitle maps the rune to title case giving priority to the special mapping.
func (special SpecialCase) ToTitle(r rune) rune {
	r1 := to(TitleCase, r, []CaseRange(special))
	if r1 == r {
		r1 = ToTitle(r)
	}
	return r1
}

// ToLower maps the rune to lower case giving priority to the special mapping.
func (special SpecialCase) ToLower(r rune) rune {
	r1 := to(LowerCase, r, []CaseRange(special))
	if r1 == r {
		r1 = ToLower(r)
	}
	return r1
}

// caseOrbit is defined in tables.go as []foldPair.  Right now all the
// entries fit in uint16, so use uint16.  If that changes, compilation
// will fail (the constants in the composite literal will not fit in uint16)
// and the types here can change to uint32.
type foldPair struct {
	From uint16
	To   uint16
}

// SimpleFold iterates over Unicode code points equivalent under
// the Unicode-defined simple case folding.  Among the code points
// equivalent to rune (including rune itself), SimpleFold returns the
// smallest r >= rune if one exists, or else the smallest r >= 0. 
//
// For example:
//	SimpleFold('A') = 'a'
//	SimpleFold('a') = 'A'
//
//	SimpleFold('K') = 'k'
//	SimpleFold('k') = '\u212A' (Kelvin symbol, K)
//	SimpleFold('\u212A') = 'K'
//
//	SimpleFold('1') = '1'
//
func SimpleFold(r rune) rune {
	// Consult caseOrbit table for special cases.
	lo := 0
	hi := len(caseOrbit)
	for lo < hi {
		m := lo + (hi-lo)/2
		if rune(caseOrbit[m].From) < r {
			lo = m + 1
		} else {
			hi = m
		}
	}
	if lo < len(caseOrbit) && rune(caseOrbit[lo].From) == r {
		return rune(caseOrbit[lo].To)
	}

	// No folding specified.  This is a one- or two-element
	// equivalence class containing rune and ToLower(rune)
	// and ToUpper(rune) if they are different from rune.
	if l := ToLower(r); l != r {
		return l
	}
	return ToUpper(r)
}

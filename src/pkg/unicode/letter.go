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

type d [MaxCase]int32 // to make the CaseRanges text shorter

// If the Delta field of a CaseRange is UpperLower or LowerUpper, it means
// this CaseRange represents a sequence of the form (say)
// Upper Lower Upper Lower.
const (
	UpperLower = MaxRune + 1 // (Cannot be a valid delta.)
)

// is16 uses binary search to test whether rune is in the specified slice of 16-bit ranges.
func is16(ranges []Range16, rune uint16) bool {
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

// is32 uses binary search to test whether rune is in the specified slice of 32-bit ranges.
func is32(ranges []Range32, rune uint32) bool {
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

// Is tests whether rune is in the specified table of ranges.
func Is(rangeTab *RangeTable, rune int) bool {
	// common case: rune is ASCII or Latin-1.
	if uint32(rune) <= MaxLatin1 {
		// Only need to check R16, since R32 is always >= 1<<16.
		r16 := uint16(rune)
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
	if len(r16) > 0 && rune <= int(r16[len(r16)-1].Hi) {
		return is16(r16, uint16(rune))
	}
	r32 := rangeTab.R32
	if len(r32) > 0 && rune >= int(r32[0].Lo) {
		return is32(r32, uint32(rune))
	}
	return false
}

// IsUpper reports whether the rune is an upper case letter.
func IsUpper(rune int) bool {
	// See comment in IsGraphic.
	if uint32(rune) <= MaxLatin1 {
		return properties[uint8(rune)]&pLu != 0
	}
	return Is(Upper, rune)
}

// IsLower reports whether the rune is a lower case letter.
func IsLower(rune int) bool {
	// See comment in IsGraphic.
	if uint32(rune) <= MaxLatin1 {
		return properties[uint8(rune)]&pLl != 0
	}
	return Is(Lower, rune)
}

// IsTitle reports whether the rune is a title case letter.
func IsTitle(rune int) bool {
	if rune <= MaxLatin1 {
		return false
	}
	return Is(Title, rune)
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
		if int(r.Lo) <= rune && rune <= int(r.Hi) {
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
				return int(r.Lo) + ((rune-int(r.Lo))&^1 | _case&1)
			}
			return rune + delta
		}
		if rune < int(r.Lo) {
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
	if rune <= MaxASCII {
		if 'a' <= rune && rune <= 'z' {
			rune -= 'a' - 'A'
		}
		return rune
	}
	return To(UpperCase, rune)
}

// ToLower maps the rune to lower case.
func ToLower(rune int) int {
	if rune <= MaxASCII {
		if 'A' <= rune && rune <= 'Z' {
			rune += 'a' - 'A'
		}
		return rune
	}
	return To(LowerCase, rune)
}

// ToTitle maps the rune to title case.
func ToTitle(rune int) int {
	if rune <= MaxASCII {
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
func SimpleFold(rune int) int {
	// Consult caseOrbit table for special cases.
	lo := 0
	hi := len(caseOrbit)
	for lo < hi {
		m := lo + (hi-lo)/2
		if int(caseOrbit[m].From) < rune {
			lo = m + 1
		} else {
			hi = m
		}
	}
	if lo < len(caseOrbit) && int(caseOrbit[lo].From) == rune {
		return int(caseOrbit[lo].To)
	}

	// No folding specified.  This is a one- or two-element
	// equivalence class containing rune and ToLower(rune)
	// and ToUpper(rune) if they are different from rune.
	if l := ToLower(rune); l != rune {
		return l
	}
	return ToUpper(rune)
}

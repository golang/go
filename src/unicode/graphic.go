// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode

// Bit masks for each code point under U+0100, for fast lookup.
const (
	pC     = 1 << iota // a control character.
	pP                 // a punctuation character.
	pN                 // a numeral.
	pS                 // a symbolic character.
	pZ                 // a spacing character.
	pLu                // an upper-case letter.
	pLl                // a lower-case letter.
	pp                 // a printable character according to Go's definition.
	pg     = pp | pZ   // a graphical character according to the Unicode definition.
	pLo    = pLl | pLu // a letter that is neither upper nor lower case.
	pLmask = pLo
)

// GraphicRanges defines the set of graphic characters according to Unicode.
var GraphicRanges = []*RangeTable{
	L, M, N, P, S, Zs,
}

// PrintRanges defines the set of printable characters according to Go.
// ASCII space, U+0020, is handled separately.
var PrintRanges = []*RangeTable{
	L, M, N, P, S,
}

// IsGraphic reports whether the rune is defined as a Graphic by Unicode.
// Such characters include letters, marks, numbers, punctuation, symbols, and
// spaces, from categories [L], [M], [N], [P], [S], [Zs].
func IsGraphic(r rune) bool {
	// We convert to uint32 to avoid the extra test for negative,
	// and in the index we convert to uint8 to avoid the range check.
	if uint32(r) <= MaxLatin1 {
		return properties[uint8(r)]&pg != 0
	}
	return In(r, GraphicRanges...)
}

// IsPrint reports whether the rune is defined as printable by Go. Such
// characters include letters, marks, numbers, punctuation, symbols, and the
// ASCII space character, from categories [L], [M], [N], [P], [S] and the ASCII space
// character. This categorization is the same as [IsGraphic] except that the
// only spacing character is ASCII space, U+0020.
func IsPrint(r rune) bool {
	if uint32(r) <= MaxLatin1 {
		return properties[uint8(r)]&pp != 0
	}
	return In(r, PrintRanges...)
}

// IsOneOf reports whether the rune is a member of one of the ranges.
// The function "In" provides a nicer signature and should be used in preference to IsOneOf.
func IsOneOf(ranges []*RangeTable, r rune) bool {
	for _, inside := range ranges {
		if Is(inside, r) {
			return true
		}
	}
	return false
}

// In reports whether the rune is a member of one of the ranges.
func In(r rune, ranges ...*RangeTable) bool {
	for _, inside := range ranges {
		if Is(inside, r) {
			return true
		}
	}
	return false
}

// IsControl reports whether the rune is a control character.
// The [C] ([Other]) Unicode category includes more code points
// such as surrogates; use [Is](C, r) to test for them.
func IsControl(r rune) bool {
	if uint32(r) <= MaxLatin1 {
		return properties[uint8(r)]&pC != 0
	}
	// All control characters are < MaxLatin1.
	return false
}

// IsLetter reports whether the rune is a letter (category [L]).
func IsLetter(r rune) bool {
	if uint32(r) <= MaxLatin1 {
		return properties[uint8(r)]&(pLmask) != 0
	}
	return isExcludingLatin(Letter, r)
}

// IsMark reports whether the rune is a mark character (category [M]).
func IsMark(r rune) bool {
	// There are no mark characters in Latin-1.
	return isExcludingLatin(Mark, r)
}

// IsNumber reports whether the rune is a number (category [N]).
func IsNumber(r rune) bool {
	if uint32(r) <= MaxLatin1 {
		return properties[uint8(r)]&pN != 0
	}
	return isExcludingLatin(Number, r)
}

// IsPunct reports whether the rune is a Unicode punctuation character
// (category [P]).
func IsPunct(r rune) bool {
	if uint32(r) <= MaxLatin1 {
		return properties[uint8(r)]&pP != 0
	}
	return Is(Punct, r)
}

// IsSpace reports whether the rune is a space character as defined
// by Unicode's White Space property; in the Latin-1 space
// this is
//
//	'\t', '\n', '\v', '\f', '\r', ' ', U+0085 (NEL), U+00A0 (NBSP).
//
// Other definitions of spacing characters are set by category
// Z and property [Pattern_White_Space].
func IsSpace(r rune) bool {
	// This property isn't the same as Z; special-case it.
	if uint32(r) <= MaxLatin1 {
		switch r {
		case '\t', '\n', '\v', '\f', '\r', ' ', 0x85, 0xA0:
			return true
		}
		return false
	}
	return isExcludingLatin(White_Space, r)
}

// IsSymbol reports whether the rune is a symbolic character.
func IsSymbol(r rune) bool {
	if uint32(r) <= MaxLatin1 {
		return properties[uint8(r)]&pS != 0
	}
	return isExcludingLatin(Symbol, r)
}

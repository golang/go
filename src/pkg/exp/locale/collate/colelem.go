// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

import (
	"unicode"
)

const (
	defaultSecondary = 0x20
	defaultTertiary  = 0x2
	maxTertiary      = 0x1F
	maxQuaternary    = 0x1FFFFF // 21 bits.
)

// colElem is a representation of a collation element.
// In the typical case, a rune maps to a single collation element. If a rune
// can be the start of a contraction or expands into multiple collation elements,
// then the colElem that is associated with a rune will have a special form to represent
// such m to n mappings.  Such special colElems have a value >= 0x80000000.
type colElem uint32

const (
	maxCE       colElem = 0x80FFFFFF
	minContract         = 0xC0000000
	maxContract         = 0xDFFFFFFF
	minExpand           = 0xE0000000
	maxExpand           = 0xEFFFFFFF
	minDecomp           = 0xF0000000
)

type ceType int

const (
	ceNormal           ceType = iota // ceNormal includes implicits (ce == 0)
	ceContractionIndex               // rune can be a start of a contraction
	ceExpansionIndex                 // rune expands into a sequence of collation elements
	ceDecompose                      // rune expands using NFKC decomposition
)

func (ce colElem) ctype() ceType {
	if ce <= maxCE {
		return ceNormal
	}
	if ce <= maxContract {
		return ceContractionIndex
	} else {
		if ce <= maxExpand {
			return ceExpansionIndex
		}
		return ceDecompose
	}
	panic("should not reach here")
	return ceType(-1)
}

// For normal collation elements, we assume that a collation element either has
// a primary or non-default secondary value, not both.
// Collation elements with a primary value are of the form
// 01pppppp pppppppp ppppppp0 ssssssss
//   - p* is primary collation value
//   - s* is the secondary collation value
// or
// 00pppppp pppppppp ppppppps sssttttt, where
//   - p* is primary collation value
//   - s* offset of secondary from default value.
//   - t* is the tertiary collation value
// Collation elements with a secondary value are of the form
// 10000000 0000ssss ssssssss tttttttt, where
//   - 16 BMP implicit -> weight
//   - 8 bit s
//   - default tertiary
// 11qqqqqq qqqqqqqq qqqqqqq0 00000000
//   - q* quaternary value
const (
	ceTypeMask            = 0xC0000000
	ceType1               = 0x40000000
	ceType2               = 0x00000000
	ceType3               = 0x80000000
	ceTypeQ               = 0xC0000000
	ceIgnore              = ceType3
	firstNonPrimary       = 0x80000000
	secondaryMask         = 0x80000000
	hasTertiaryMask       = 0x40000000
	primaryValueMask      = 0x3FFFFE00
	primaryShift          = 9
	compactSecondaryShift = 5
	minCompactSecondary   = defaultSecondary - 4
)

func makeImplicitCE(primary int) colElem {
	return ceType1 | colElem(primary<<primaryShift) | defaultSecondary
}

func makeQuaternary(primary int) colElem {
	return ceTypeQ | colElem(primary<<primaryShift)
}

func (ce colElem) primary() int {
	if ce >= firstNonPrimary {
		return 0
	}
	return int(ce&primaryValueMask) >> primaryShift
}

func (ce colElem) secondary() int {
	switch ce & ceTypeMask {
	case ceType1:
		return int(uint8(ce))
	case ceType2:
		return minCompactSecondary + int((ce>>compactSecondaryShift)&0xF)
	case ceType3:
		return int(uint16(ce >> 8))
	case ceTypeQ:
		return 0
	}
	panic("should not reach here")
}

func (ce colElem) tertiary() uint8 {
	if ce&hasTertiaryMask == 0 {
		if ce&ceType3 == 0 {
			return uint8(ce & 0x1F)
		}
		return uint8(ce)
	} else if ce&ceTypeMask == ceType1 {
		return defaultTertiary
	}
	// ce is a quaternary value. 
	return 0
}

func (ce colElem) updateTertiary(t uint8) colElem {
	if ce&ceTypeMask == ceType1 {
		nce := ce & primaryValueMask
		nce |= colElem(uint8(ce)-minCompactSecondary) << compactSecondaryShift
		ce = nce
	} else {
		ce &= ^colElem(maxTertiary)
	}
	return ce | colElem(t)
}

// quaternary returns the quaternary value if explicitly specified,
// 0 if ce == ceIgnore, or maxQuaternary otherwise.
// Quaternary values are used only for shifted variants.
func (ce colElem) quaternary() int {
	if ce&ceTypeMask == ceTypeQ {
		return int(ce&primaryValueMask) >> primaryShift
	} else if ce == ceIgnore {
		return 0
	}
	return maxQuaternary
}

// For contractions, collation elements are of the form
// 110bbbbb bbbbbbbb iiiiiiii iiiinnnn, where
//   - n* is the size of the first node in the contraction trie.
//   - i* is the index of the first node in the contraction trie.
//   - b* is the offset into the contraction collation element table.
// See contract.go for details on the contraction trie.
const (
	maxNBits              = 4
	maxTrieIndexBits      = 12
	maxContractOffsetBits = 13
)

func splitContractIndex(ce colElem) (index, n, offset int) {
	n = int(ce & (1<<maxNBits - 1))
	ce >>= maxNBits
	index = int(ce & (1<<maxTrieIndexBits - 1))
	ce >>= maxTrieIndexBits
	offset = int(ce & (1<<maxContractOffsetBits - 1))
	return
}

// For expansions, colElems are of the form 11100000 00000000 bbbbbbbb bbbbbbbb,
// where b* is the index into the expansion sequence table.
const maxExpandIndexBits = 16

func splitExpandIndex(ce colElem) (index int) {
	return int(uint16(ce))
}

// Some runes can be expanded using NFKD decomposition. Instead of storing the full
// sequence of collation elements, we decompose the rune and lookup the collation
// elements for each rune in the decomposition and modify the tertiary weights.
// The colElem, in this case, is of the form 11110000 00000000 wwwwwwww vvvvvvvv, where
//   - v* is the replacement tertiary weight for the first rune,
//   - w* is the replacement tertiary weight for the second rune,
// Tertiary weights of subsequent runes should be replaced with maxTertiary.
// See http://www.unicode.org/reports/tr10/#Compatibility_Decompositions for more details.
func splitDecompose(ce colElem) (t1, t2 uint8) {
	return uint8(ce), uint8(ce >> 8)
}

const (
	// These constants were taken from http://www.unicode.org/versions/Unicode6.0.0/ch12.pdf.
	minUnified       rune = 0x4E00
	maxUnified            = 0x9FFF
	minCompatibility      = 0xF900
	maxCompatibility      = 0xFAFF
	minRare               = 0x3400
	maxRare               = 0x4DBF
)
const (
	commonUnifiedOffset = 0x10000
	rareUnifiedOffset   = 0x20000 // largest rune in common is U+FAFF
	otherOffset         = 0x50000 // largest rune in rare is U+2FA1D
	illegalOffset       = otherOffset + int(unicode.MaxRune)
	maxPrimary          = illegalOffset + 1
)

// implicitPrimary returns the primary weight for the a rune
// for which there is no entry for the rune in the collation table.
// We take a different approach from the one specified in
// http://unicode.org/reports/tr10/#Implicit_Weights,
// but preserve the resulting relative ordering of the runes.
func implicitPrimary(r rune) int {
	if unicode.Is(unicode.Ideographic, r) {
		if r >= minUnified && r <= maxUnified {
			// The most common case for CJK.
			return int(r) + commonUnifiedOffset
		}
		if r >= minCompatibility && r <= maxCompatibility {
			// This will typically not hit. The DUCET explicitly specifies mappings
			// for all characters that do not decompose.
			return int(r) + commonUnifiedOffset
		}
		return int(r) + rareUnifiedOffset
	}
	return int(r) + otherOffset
}

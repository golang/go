// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

import (
	"unicode"
)

// weights holds the decoded weights per collation level.
type weights struct {
	primary   uint32
	secondary uint16
	tertiary  uint8
	// TODO: compute quaternary on the fly or compress this value into 8 bits
	// such that weights fit within 64bit.
	quaternary uint32
}

const (
	defaultSecondary = 0x20
	defaultTertiary  = 0x2
	maxTertiary      = 0x1F
)

// colElem is a representation of a collation element.
// In the typical case, a rune maps to a single collation element. If a rune
// can be the start of a contraction or expands into multiple collation elements,
// then the colElem that is associated with a rune will have a special form to represent
// such m to n mappings.  Such special colElems have a value >= 0x80000000.
type colElem uint32

const (
	maxCE       colElem = 0x7FFFFFFF
	minContract         = 0x80000000
	maxContract         = 0xBFFFFFFF
	minExpand           = 0xC0000000
	maxExpand           = 0xDFFFFFFF
	minDecomp           = 0xE0000000
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
// 010ppppp pppppppp pppppppp tttttttt, where
//   - p* is primary collation value
//   - t* is the tertiary collation value
// Collation elements with a secondary value are of the form
// 00000000 ssssssss ssssssss tttttttt, where
//   - s* is the secondary collation value
//   - t* is the tertiary collation value
func splitCE(ce colElem) weights {
	w := weights{}
	w.tertiary = uint8(ce)
	if ce&0x40000000 != 0 {
		// primary weight form
		w.primary = uint32((ce >> 8) & 0x1FFFFF)
		w.secondary = defaultSecondary
	} else {
		// secondary weight form
		w.secondary = uint16(ce >> 8)
	}
	return w
}

// For contractions, colElems are of the form 10bbbbbb bbbbbbbb hhhhhhhh hhhhhhhh, where
//   - h* is the compTrieHandle.
//   - b* is the offset into the contraction collation element table.
// See contract.go for details on the contraction trie.
const (
	maxNBits              = 5
	maxTrieIndexBits      = 11
	maxContractOffsetBits = 14
)

func splitContractIndex(ce colElem) (index, n, offset int) {
	h := uint16(ce)
	return int(h >> maxNBits), int(h & (1<<maxNBits - 1)), int(ce>>16) & (1<<maxContractOffsetBits - 1)
}

// For expansions, colElems are of the form 110bbbbb bbbbbbbb bbbbbbbb bbbbbbbb,
// where b* is the index into the expansion sequence table.
const (
	maxExpandIndexBits = 29
)

func splitExpandIndex(ce colElem) (index int) {
	index = int(ce) & (1<<maxExpandIndexBits - 1)
	return
}

// Some runes can be expanded using NFKD decomposition. Instead of storing the full
// sequence of collation elements, we decompose the rune and lookup the collation
// elements for each rune in the decomposition and modify the tertiary weights.
// The colElem, in this case, is of the form 11100000 00000000 wwwwwwww vvvvvvvv, where
//   - v* is the replacement tertiary weight for the first rune,
//   - w* is the replacement tertiary weight for the second rune,
// Tertiary weights of subsequent runes should be replaced with maxTertiary.
// See http://www.unicode.org/reports/tr10/#Compatibility_Decompositions for more details.
const (
	decompID = 0xE0000000
)

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
	commonUnifiedOffset = 0xFB40
	rareUnifiedOffset   = 0x1FB40
	otherOffset         = 0x4FB40
	maxPrimary          = otherOffset + unicode.MaxRune
)

// implicitPrimary returns the primary weight for the a rune
// for which there is no entry for the rune in the collation table.
// We take a different approach from the one specified in
// http://unicode.org/reports/tr10/#Implicit_Weights,
// but preserve the resulting relative ordering of the runes.
func implicitPrimary(r rune) int {

	if r >= minUnified && r <= maxUnified {
		// The most common case for CJK.
		return int(r) + commonUnifiedOffset
	}
	if r >= minCompatibility && r <= maxCompatibility {
		// This will never hit as long as we don't remove the characters
		// that would match from the table.
		return int(r) + commonUnifiedOffset
	}
	if unicode.Is(unicode.Unified_Ideograph, r) {
		return int(r) + rareUnifiedOffset
	}
	return int(r) + otherOffset
}

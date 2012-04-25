// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"fmt"
	"unicode"
)

const (
	defaultSecondary = 0x20
	defaultTertiary  = 0x2
	maxTertiary      = 0x1F
)

// A collation element is represented as an uint32.
// In the typical case, a rune maps to a single collation element. If a rune
// can be the start of a contraction or expands into multiple collation elements,
// then the collation element that is associated with a rune will have a special
// form to represent such m to n mappings.  Such special collation elements
// have a value >= 0x80000000.

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
const (
	maxPrimaryBits   = 21
	maxSecondaryBits = 16
	maxTertiaryBits  = 8

	isPrimary = 0x40000000
)

func makeCE(weights []int) (uint32, error) {
	if w := weights[0]; w >= 1<<maxPrimaryBits || w < 0 {
		return 0, fmt.Errorf("makeCE: primary weight out of bounds: %x >= %x", w, 1<<maxPrimaryBits)
	}
	if w := weights[1]; w >= 1<<maxSecondaryBits || w < 0 {
		return 0, fmt.Errorf("makeCE: secondary weight out of bounds: %x >= %x", w, 1<<maxSecondaryBits)
	}
	if w := weights[2]; w >= 1<<maxTertiaryBits || w < 0 {
		return 0, fmt.Errorf("makeCE: tertiary weight out of bounds: %d >= %d", w, 1<<maxTertiaryBits)
	}
	ce := uint32(0)
	if weights[0] != 0 {
		// primary weight form
		if weights[1] != defaultSecondary {
			return 0, fmt.Errorf("makeCE: non-default secondary weight for non-zero primary: %X", weights)
		}
		ce = uint32(weights[0]<<maxTertiaryBits + weights[2])
		ce |= isPrimary
	} else {
		// secondary weight form
		ce = uint32(weights[1]<<maxTertiaryBits + weights[2])
	}
	return ce, nil
}

// For contractions, collation elements are of the form
// 10bbbbbb bbbbbbbb iiiiiiii iiinnnnn, where
//   - n* is the size of the first node in the contraction trie.
//   - i* is the index of the first node in the contraction trie.
//   - b* is the offset into the contraction collation element table.
// See contract.go for details on the contraction trie.
const (
	contractID            = 0x80000000
	maxNBits              = 5
	maxTrieIndexBits      = 11
	maxContractOffsetBits = 14
)

func makeContractIndex(h ctHandle, offset int) (uint32, error) {
	if h.n >= 1<<maxNBits {
		return 0, fmt.Errorf("size of contraction trie node too large: %d >= %d", h.n, 1<<maxNBits)
	}
	if h.index >= 1<<maxTrieIndexBits {
		return 0, fmt.Errorf("size of contraction trie offset too large: %d >= %d", h.index, 1<<maxTrieIndexBits)
	}
	if offset >= 1<<maxContractOffsetBits {
		return 0, fmt.Errorf("offset out of bounds: %x >= %x", offset, 1<<maxContractOffsetBits)
	}
	ce := uint32(contractID)
	ce += uint32(offset << (maxTrieIndexBits + maxNBits))
	ce += uint32(h.index << maxNBits)
	ce += uint32(h.n)
	return ce, nil
}

// For expansions, collation elements are of the form
// 110bbbbb bbbbbbbb bbbbbbbb bbbbbbbb,
// where b* is the index into the expansion sequence table.
const (
	expandID           = 0xC0000000
	maxExpandIndexBits = 29
)

func makeExpandIndex(index int) (uint32, error) {
	if index >= 1<<maxExpandIndexBits {
		return 0, fmt.Errorf("index out of bounds: %x >= %x", index, 1<<maxExpandIndexBits)
	}
	return expandID + uint32(index), nil
}

// Each list of collation elements corresponding to an expansion starts with 
// a header indicating the length of the sequence.
func makeExpansionHeader(n int) (uint32, error) {
	return uint32(n), nil
}

// Some runes can be expanded using NFKD decomposition. Instead of storing the full
// sequence of collation elements, we decompose the rune and lookup the collation
// elements for each rune in the decomposition and modify the tertiary weights.
// The collation element, in this case, is of the form
// 11100000 00000000 wwwwwwww vvvvvvvv, where
//   - v* is the replacement tertiary weight for the first rune,
//   - w* is the replacement tertiary weight for the second rune,
// Tertiary weights of subsequent runes should be replaced with maxTertiary.
// See http://www.unicode.org/reports/tr10/#Compatibility_Decompositions for more details.
const (
	decompID = 0xE0000000
)

func makeDecompose(t1, t2 int) (uint32, error) {
	if t1 >= 256 || t1 < 0 {
		return 0, fmt.Errorf("first tertiary weight out of bounds: %d >= 256", t1)
	}
	if t2 >= 256 || t2 < 0 {
		return 0, fmt.Errorf("second tertiary weight out of bounds: %d >= 256", t2)
	}
	return uint32(t2<<8+t1) + decompID, nil
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
	illegalOffset       = otherOffset + unicode.MaxRune
	maxPrimary          = illegalOffset + 2 // there are 2 illegal values.
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

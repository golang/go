// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"exp/locale/collate"
	"fmt"
	"unicode"
)

const (
	defaultSecondary = 0x20
	defaultTertiary  = 0x2
	maxTertiary      = 0x1F
)

type rawCE struct {
	w   []int
	ccc uint8
}

func makeRawCE(w []int, ccc uint8) rawCE {
	ce := rawCE{w: make([]int, 4), ccc: ccc}
	copy(ce.w, w)
	return ce
}

// A collation element is represented as an uint32.
// In the typical case, a rune maps to a single collation element. If a rune
// can be the start of a contraction or expands into multiple collation elements,
// then the collation element that is associated with a rune will have a special
// form to represent such m to n mappings.  Such special collation elements
// have a value >= 0x80000000.

// For normal collation elements, we assume that a collation element either has
// a primary or non-default secondary value, not both.
// Collation elements with a primary value are of the form
// 01pppppp pppppppp ppppppp0 ssssssss
//   - p* is primary collation value
//   - s* is the secondary collation value
// 00pppppp pppppppp ppppppps sssttttt, where
//   - p* is primary collation value
//   - s* offset of secondary from default value.
//   - t* is the tertiary collation value
// 100ttttt cccccccc pppppppp pppppppp
//   - t* is the tertiar collation value
//   - c* is the cannonical combining class
//   - p* is the primary collation value
// Collation elements with a secondary value are of the form
// 1010cccc ccccssss ssssssss tttttttt, where
//   - c* is the canonical combining class
//   - s* is the secondary collation value
//   - t* is the tertiary collation value
const (
	maxPrimaryBits          = 21
	maxPrimaryCompactBits   = 16
	maxSecondaryBits        = 12
	maxSecondaryCompactBits = 8
	maxCCCBits              = 8
	maxSecondaryDiffBits    = 4
	maxTertiaryBits         = 8
	maxTertiaryCompactBits  = 5

	isPrimary    = 0x40000000
	isPrimaryCCC = 0x80000000
	isSecondary  = 0xA0000000
)

func makeCE(rce rawCE) (uint32, error) {
	weights := rce.w
	if w := weights[0]; w >= 1<<maxPrimaryBits || w < 0 {
		return 0, fmt.Errorf("makeCE: primary weight out of bounds: %x >= %x", w, 1<<maxPrimaryBits)
	}
	if w := weights[1]; w >= 1<<maxSecondaryBits || w < 0 {
		return 0, fmt.Errorf("makeCE: secondary weight out of bounds: %x >= %x", w, 1<<maxSecondaryBits)
	}
	if w := weights[2]; w >= 1<<maxTertiaryBits || w < 0 {
		return 0, fmt.Errorf("makeCE: tertiary weight out of bounds: %x >= %x", w, 1<<maxTertiaryBits)
	}
	ce := uint32(0)
	if weights[0] != 0 {
		if rce.ccc != 0 {
			if weights[0] >= 1<<maxPrimaryCompactBits {
				return 0, fmt.Errorf("makeCE: primary weight with non-zero CCC out of bounds: %x >= %x", weights[0], 1<<maxPrimaryCompactBits)
			}
			if weights[1] != defaultSecondary {
				return 0, fmt.Errorf("makeCE: cannot combine non-default secondary value (%x) with non-zero CCC (%x)", weights[1], rce.ccc)
			}
			ce = uint32(weights[2] << (maxPrimaryCompactBits + maxCCCBits))
			ce |= uint32(rce.ccc) << maxPrimaryCompactBits
			ce |= uint32(weights[0])
			ce |= isPrimaryCCC
		} else if weights[2] == defaultTertiary {
			if weights[1] >= 1<<maxSecondaryCompactBits {
				return 0, fmt.Errorf("makeCE: secondary weight with non-zero primary out of bounds: %x >= %x", weights[1], 1<<maxSecondaryCompactBits)
			}
			ce = uint32(weights[0]<<(maxSecondaryCompactBits+1) + weights[1])
			ce |= isPrimary
		} else {
			d := weights[1] - defaultSecondary + maxSecondaryDiffBits
			if d >= 1<<maxSecondaryDiffBits || d < 0 {
				return 0, fmt.Errorf("makeCE: secondary weight diff out of bounds: %x < 0 || %x > %x", d, d, 1<<maxSecondaryDiffBits)
			}
			if weights[2] >= 1<<maxTertiaryCompactBits {
				return 0, fmt.Errorf("makeCE: tertiary weight with non-zero primary out of bounds: %x > %x (%X)", weights[2], 1<<maxTertiaryCompactBits, weights)
			}
			ce = uint32(weights[0]<<maxSecondaryDiffBits + d)
			ce = ce<<maxTertiaryCompactBits + uint32(weights[2])
		}
	} else {
		ce = uint32(weights[1]<<maxTertiaryBits + weights[2])
		ce += uint32(rce.ccc) << (maxSecondaryBits + maxTertiaryBits)
		ce |= isSecondary
	}
	return ce, nil
}

// For contractions, collation elements are of the form
// 110bbbbb bbbbbbbb iiiiiiii iiiinnnn, where
//   - n* is the size of the first node in the contraction trie.
//   - i* is the index of the first node in the contraction trie.
//   - b* is the offset into the contraction collation element table.
// See contract.go for details on the contraction trie.
const (
	contractID            = 0xC0000000
	maxNBits              = 4
	maxTrieIndexBits      = 12
	maxContractOffsetBits = 13
)

func makeContractIndex(h ctHandle, offset int) (uint32, error) {
	if h.n >= 1<<maxNBits {
		return 0, fmt.Errorf("size of contraction trie node too large: %d >= %d", h.n, 1<<maxNBits)
	}
	if h.index >= 1<<maxTrieIndexBits {
		return 0, fmt.Errorf("size of contraction trie offset too large: %d >= %d", h.index, 1<<maxTrieIndexBits)
	}
	if offset >= 1<<maxContractOffsetBits {
		return 0, fmt.Errorf("contraction offset out of bounds: %x >= %x", offset, 1<<maxContractOffsetBits)
	}
	ce := uint32(contractID)
	ce += uint32(offset << (maxNBits + maxTrieIndexBits))
	ce += uint32(h.index << maxNBits)
	ce += uint32(h.n)
	return ce, nil
}

// For expansions, collation elements are of the form
// 11100000 00000000 bbbbbbbb bbbbbbbb,
// where b* is the index into the expansion sequence table.
const (
	expandID           = 0xE0000000
	maxExpandIndexBits = 16
)

func makeExpandIndex(index int) (uint32, error) {
	if index >= 1<<maxExpandIndexBits {
		return 0, fmt.Errorf("expansion index out of bounds: %x >= %x", index, 1<<maxExpandIndexBits)
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
// 11110000 00000000 wwwwwwww vvvvvvvv, where
//   - v* is the replacement tertiary weight for the first rune,
//   - w* is the replacement tertiary weight for the second rune,
// Tertiary weights of subsequent runes should be replaced with maxTertiary.
// See http://www.unicode.org/reports/tr10/#Compatibility_Decompositions for more details.
const (
	decompID = 0xF0000000
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

// convertLargeWeights converts collation elements with large
// primaries (either double primaries or for illegal runes)
// to our own representation.
// A CJK character C is represented in the DUCET as
//   [.FBxx.0020.0002.C][.BBBB.0000.0000.C]
// We will rewrite these characters to a single CE.
// We assume the CJK values start at 0x8000.
// See http://unicode.org/reports/tr10/#Implicit_Weights
func convertLargeWeights(elems []rawCE) (res []rawCE, err error) {
	const (
		cjkPrimaryStart   = 0xFB40
		rarePrimaryStart  = 0xFB80
		otherPrimaryStart = 0xFBC0
		illegalPrimary    = 0xFFFE
		highBitsMask      = 0x3F
		lowBitsMask       = 0x7FFF
		lowBitsFlag       = 0x8000
		shiftBits         = 15
	)
	for i := 0; i < len(elems); i++ {
		ce := elems[i].w
		p := ce[0]
		if p < cjkPrimaryStart {
			continue
		}
		if p > 0xFFFF {
			return elems, fmt.Errorf("found primary weight %X; should be <= 0xFFFF", p)
		}
		if p >= illegalPrimary {
			ce[0] = illegalOffset + p - illegalPrimary
		} else {
			if i+1 >= len(elems) {
				return elems, fmt.Errorf("second part of double primary weight missing: %v", elems)
			}
			if elems[i+1].w[0]&lowBitsFlag == 0 {
				return elems, fmt.Errorf("malformed second part of double primary weight: %v", elems)
			}
			np := ((p & highBitsMask) << shiftBits) + elems[i+1].w[0]&lowBitsMask
			switch {
			case p < rarePrimaryStart:
				np += commonUnifiedOffset
			case p < otherPrimaryStart:
				np += rareUnifiedOffset
			default:
				p += otherOffset
			}
			ce[0] = np
			for j := i + 1; j+1 < len(elems); j++ {
				elems[j] = elems[j+1]
			}
			elems = elems[:len(elems)-1]
		}
	}
	return elems, nil
}

// nextWeight computes the first possible collation weights following elems
// for the given level.
func nextWeight(level collate.Level, elems []rawCE) []rawCE {
	if level == collate.Identity {
		next := make([]rawCE, len(elems))
		copy(next, elems)
		return next
	}
	next := []rawCE{makeRawCE(elems[0].w, elems[0].ccc)}
	next[0].w[level]++
	if level < collate.Secondary {
		next[0].w[collate.Secondary] = defaultSecondary
	}
	if level < collate.Tertiary {
		next[0].w[collate.Tertiary] = defaultTertiary
	}
	// Filter entries that cannot influence ordering.
	for _, ce := range elems[1:] {
		skip := true
		for i := collate.Primary; i < level; i++ {
			skip = skip && ce.w[i] == 0
		}
		if !skip {
			next = append(next, ce)
		}
	}
	return next
}

func nextVal(elems []rawCE, i int, level collate.Level) (index, value int) {
	for ; i < len(elems) && elems[i].w[level] == 0; i++ {
	}
	if i < len(elems) {
		return i, elems[i].w[level]
	}
	return i, 0
}

// compareWeights returns -1 if a < b, 1 if a > b, or 0 otherwise.
// It also returns the collation level at which the difference is found.
func compareWeights(a, b []rawCE) (result int, level collate.Level) {
	for level := collate.Primary; level < collate.Identity; level++ {
		var va, vb int
		for ia, ib := 0, 0; ia < len(a) || ib < len(b); ia, ib = ia+1, ib+1 {
			ia, va = nextVal(a, ia, level)
			ib, vb = nextVal(b, ib, level)
			if va != vb {
				if va < vb {
					return -1, level
				} else {
					return 1, level
				}
			}
		}
	}
	return 0, collate.Identity
}

func equalCE(a, b rawCE) bool {
	for i := 0; i < 3; i++ {
		if b.w[i] != a.w[i] {
			return false
		}
	}
	return true
}

func equalCEArrays(a, b []rawCE) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !equalCE(a[i], b[i]) {
			return false
		}
	}
	return true
}

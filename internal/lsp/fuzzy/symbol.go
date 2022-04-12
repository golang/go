// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzzy

import (
	"unicode"
)

// SymbolMatcher implements a fuzzy matching algorithm optimized for Go symbols
// of the form:
//
//	example.com/path/to/package.object.field
//
// Knowing that we are matching symbols like this allows us to make the
// following optimizations:
//   - We can incorporate right-to-left relevance directly into the score
//     calculation.
//   - We can match from right to left, discarding leading bytes if the input is
//     too long.
//   - We just take the right-most match without losing too much precision. This
//     allows us to use an O(n) algorithm.
//   - We can operate directly on chunked strings; in many cases we will
//     be storing the package path and/or package name separately from the
//     symbol or identifiers, so doing this avoids allocating strings.
//   - We can return the index of the right-most match, allowing us to trim
//     irrelevant qualification.
//
// This implementation is experimental, serving as a reference fast algorithm
// to compare to the fuzzy algorithm implemented by Matcher.
type SymbolMatcher struct {
	// Using buffers of length 256 is both a reasonable size for most qualified
	// symbols, and makes it easy to avoid bounds checks by using uint8 indexes.
	pattern     [256]rune
	patternLen  uint8
	inputBuffer [256]rune   // avoid allocating when considering chunks
	roles       [256]uint32 // which roles does a rune play (word start, etc.)
	segments    [256]uint8  // how many segments from the right is each rune
}

const (
	segmentStart uint32 = 1 << iota
	wordStart
	separator
)

// NewSymbolMatcher creates a SymbolMatcher that may be used to match the given
// search pattern.
//
// Currently this matcher only accepts case-insensitive fuzzy patterns.
//
// An empty pattern matches no input.
func NewSymbolMatcher(pattern string) *SymbolMatcher {
	m := &SymbolMatcher{}
	for _, p := range pattern {
		m.pattern[m.patternLen] = unicode.ToLower(p)
		m.patternLen++
		if m.patternLen == 255 || int(m.patternLen) == len(pattern) {
			// break at 255 so that we can represent patternLen with a uint8.
			break
		}
	}
	return m
}

// Match looks for the right-most match of the search pattern within the symbol
// represented by concatenating the given chunks, returning its offset and
// score.
//
// If a match is found, the first return value will hold the absolute byte
// offset within all chunks for the start of the symbol. In other words, the
// index of the match within strings.Join(chunks, ""). If no match is found,
// the first return value will be -1.
//
// The second return value will be the score of the match, which is always
// between 0 and 1, inclusive. A score of 0 indicates no match.
func (m *SymbolMatcher) Match(chunks []string) (int, float64) {
	// Explicit behavior for an empty pattern.
	//
	// As a minor optimization, this also avoids nilness checks later on, since
	// the compiler can prove that m != nil.
	if m.patternLen == 0 {
		return -1, 0
	}

	// First phase: populate the input buffer with lower-cased runes.
	//
	// We could also check for a forward match here, but since we'd have to write
	// the entire input anyway this has negligible impact on performance.

	var (
		inputLen  = uint8(0)
		modifiers = wordStart | segmentStart
	)

input:
	for _, chunk := range chunks {
		for _, r := range chunk {
			if r == '.' || r == '/' {
				modifiers |= separator
			}
			// optimization: avoid calls to unicode.ToLower, which can't be inlined.
			l := r
			if r <= unicode.MaxASCII {
				if 'A' <= r && r <= 'Z' {
					l = r + 'a' - 'A'
				}
			} else {
				l = unicode.ToLower(r)
			}
			if l != r {
				modifiers |= wordStart
			}
			m.inputBuffer[inputLen] = l
			m.roles[inputLen] = modifiers
			inputLen++
			if m.roles[inputLen-1]&separator != 0 {
				modifiers = wordStart | segmentStart
			} else {
				modifiers = 0
			}
			// TODO: we should prefer the right-most input if it overflows, rather
			//       than the left-most as we're doing here.
			if inputLen == 255 {
				break input
			}
		}
	}

	// Second phase: find the right-most match, and count segments from the
	// right.

	var (
		pi    = uint8(m.patternLen - 1) // pattern index
		p     = m.pattern[pi]           // pattern rune
		start = -1                      // start offset of match
		rseg  = uint8(0)
	)
	const maxSeg = 3 // maximum number of segments from the right to count, for scoring purposes.

	for ii := inputLen - 1; ; ii-- {
		r := m.inputBuffer[ii]
		if rseg < maxSeg && m.roles[ii]&separator != 0 {
			rseg++
		}
		m.segments[ii] = rseg
		if p == r {
			if pi == 0 {
				start = int(ii)
				break
			}
			pi--
			p = m.pattern[pi]
		}
		// Don't check ii >= 0 in the loop condition: ii is a uint8.
		if ii == 0 {
			break
		}
	}

	if start < 0 {
		// no match: skip scoring
		return -1, 0
	}

	// Third phase: find the shortest match, and compute the score.

	// Score is the average score for each character.
	//
	// A character score is the multiple of:
	//   1. 1.0 if the character starts a segment, .8 if the character start a
	//      mid-segment word, otherwise 0.6. This carries over to immediately
	//      following characters.
	//   2. For the final character match, the multiplier from (1) is reduced to
	//     .8 if the next character in the input is a mid-segment word, or 0.6 if
	//      the next character in the input is not a word or segment start. This
	//      ensures that we favor whole-word or whole-segment matches over prefix
	//      matches.
	//   3. 1.0 if the character is part of the last segment, otherwise
	//      1.0-.2*<segments from the right>, with a max segment count of 3.
	//
	// This is a very naive algorithm, but it is fast. There's lots of prior art
	// here, and we should leverage it. For example, we could explicitly consider
	// character distance, and exact matches of words or segments.
	//
	// Also note that this might not actually find the highest scoring match, as
	// doing so could require a non-linear algorithm, depending on how the score
	// is calculated.

	pi = 0
	p = m.pattern[pi]

	const (
		segStreak  = 1.0
		wordStreak = 0.8
		noStreak   = 0.6
		perSegment = 0.2 // we count at most 3 segments above
	)

	streakBonus := noStreak
	totScore := 0.0
	for ii := uint8(start); ii < inputLen; ii++ {
		r := m.inputBuffer[ii]
		if r == p {
			pi++
			p = m.pattern[pi]
			// Note: this could be optimized with some bit operations.
			switch {
			case m.roles[ii]&segmentStart != 0 && segStreak > streakBonus:
				streakBonus = segStreak
			case m.roles[ii]&wordStart != 0 && wordStreak > streakBonus:
				streakBonus = wordStreak
			}
			finalChar := pi >= m.patternLen
			// finalCost := 1.0
			if finalChar && streakBonus > noStreak {
				switch {
				case ii == inputLen-1 || m.roles[ii+1]&segmentStart != 0:
					// Full segment: no reduction
				case m.roles[ii+1]&wordStart != 0:
					streakBonus = wordStreak
				default:
					streakBonus = noStreak
				}
			}
			totScore += streakBonus * (1.0 - float64(m.segments[ii])*perSegment)
			if finalChar {
				break
			}
		} else {
			streakBonus = noStreak
		}
	}

	return start, totScore / float64(m.patternLen)
}

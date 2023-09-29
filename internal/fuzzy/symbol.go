// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzzy

import (
	"bytes"
	"fmt"
	"log"
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
type SymbolMatcher struct {
	// Using buffers of length 256 is both a reasonable size for most qualified
	// symbols, and makes it easy to avoid bounds checks by using uint8 indexes.
	pattern     [256]rune
	patternLen  uint8
	inputBuffer [256]rune   // avoid allocating when considering chunks
	roles       [256]uint32 // which roles does a rune play (word start, etc.)
	segments    [256]uint8  // how many segments from the right is each rune
}

// Rune roles.
const (
	segmentStart uint32 = 1 << iota // input rune starts a segment (i.e. follows '/' or '.')
	wordStart                       // input rune starts a word, per camel-case naming rules
	separator                       // input rune is a separator ('/' or '.')
	upper                           // input rune is an upper case letter
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

// Match searches for the right-most match of the search pattern within the
// symbol represented by concatenating the given chunks.
//
// If a match is found, the first result holds the absolute byte offset within
// all chunks for the start of the symbol. In other words, the index of the
// match within strings.Join(chunks, "").
//
// The second return value will be the score of the match, which is always
// between 0 and 1, inclusive. A score of 0 indicates no match.
//
// If no match is found, Match returns (-1, 0).
func (m *SymbolMatcher) Match(chunks []string) (int, float64) {
	// Explicit behavior for an empty pattern.
	//
	// As a minor optimization, this also avoids nilness checks later on, since
	// the compiler can prove that m != nil.
	if m.patternLen == 0 {
		return -1, 0
	}

	// Matching implements a heavily optimized linear scoring algorithm on the
	// input. This is not guaranteed to produce the highest score, but works well
	// enough, particularly due to the right-to-left significance of qualified
	// symbols.
	//
	// Matching proceeds in three passes through the input:
	//  - The first pass populates the input buffer and collects rune roles.
	//  - The second pass proceeds right-to-left to find the right-most match.
	//  - The third pass proceeds left-to-right from the start of the right-most
	//    match, to find the most *compact* match, and computes the score of this
	//    match.
	//
	// See below for more details of each pass, as well as the scoring algorithm.

	// First pass: populate the input buffer out of the provided chunks
	// (lower-casing in the process), and collect rune roles.
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
				modifiers |= upper

				// If the current rune is capitalized *and the preceding rune was not*,
				// mark this as a word start. This avoids spuriously high ranking of
				// non-camelcase naming schemas, such as the
				// yaml_PARSE_FLOW_SEQUENCE_ENTRY_MAPPING_END_STATE example of
				// golang/go#60201.
				if inputLen == 0 || m.roles[inputLen-1]&upper == 0 {
					modifiers |= wordStart
				}
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

	// Second pass: find the right-most match, and count segments from the
	// right.
	var (
		pi    = uint8(m.patternLen - 1) // pattern index
		p     = m.pattern[pi]           // pattern rune
		start = -1                      // start offset of match
		rseg  = uint8(0)                // effective "depth" from the right of the current rune in consideration
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
				// TODO(rfindley): BUG: the docstring for Match says that it returns an
				// absolute byte offset, but clearly it is returning a rune offset here.
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

	// Third pass: find the shortest match and compute the score.

	// Score is the average score for each rune.
	//
	// A rune score is the multiple of:
	//   1. The base score, which is 1.0 if the rune starts a segment, 0.9 if the
	//      rune starts a mid-segment word, else 0.6.
	//
	//      Runes preceded by a matching rune are treated the same as the start
	//      of a mid-segment word (with a 0.9 score), so that sequential or exact
	//      matches are preferred. We call this a sequential bonus.
	//
	//      For the final rune match, this sequential bonus is reduced to 0.8 if
	//      the next rune in the input is a mid-segment word, or 0.7 if the next
	//      rune in the input is not a word or segment start. This ensures that
	//      we favor whole-word or whole-segment matches over prefix matches.
	//
	//   2. 1.0 if the rune is part of the last segment, otherwise
	//      1.0-0.1*<segments from the right>, with a max segment count of 3.
	//      Notably 1.0-0.1*3 = 0.7 > 0.6, so that foo/_/_/_/_ (a match very
	//      early in a qualified symbol name) still scores higher than _f_o_o_ (a
	//      completely split match).
	//
	// This is a naive algorithm, but it is fast. There's lots of prior art here
	// that could be leveraged. For example, we could explicitly consider
	// rune distance, and exact matches of words or segments.
	//
	// Also note that this might not actually find the highest scoring match, as
	// doing so could require a non-linear algorithm, depending on how the score
	// is calculated.

	// debugging support
	const debug = false // enable to log debugging information
	var (
		runeScores []float64
		runeIdxs   []int
	)

	pi = 0
	p = m.pattern[pi]

	const (
		segStartScore = 1.0 // base score of runes starting a segment
		wordScore     = 0.9 // base score of runes starting or continuing a word
		noStreak      = 0.6
		perSegment    = 0.1 // we count at most 3 segments above
	)

	totScore := 0.0
	lastMatch := uint8(255)
	for ii := uint8(start); ii < inputLen; ii++ {
		r := m.inputBuffer[ii]
		if r == p {
			pi++
			finalRune := pi >= m.patternLen
			p = m.pattern[pi]

			baseScore := noStreak

			// Calculate the sequence bonus based on preceding matches.
			//
			// We do this first as it is overridden by role scoring below.
			if lastMatch == ii-1 {
				baseScore = wordScore
				// Reduce the sequence bonus for the final rune of the pattern based on
				// whether it borders a new segment or word.
				if finalRune {
					switch {
					case ii == inputLen-1 || m.roles[ii+1]&separator != 0:
						// Full segment: no reduction
					case m.roles[ii+1]&wordStart != 0:
						baseScore = wordScore - 0.1
					default:
						baseScore = wordScore - 0.2
					}
				}
			}
			lastMatch = ii

			// Calculate the rune's role score. If the rune starts a segment or word,
			// this overrides the sequence score, as the rune starts a new sequence.
			switch {
			case m.roles[ii]&segmentStart != 0:
				baseScore = segStartScore
			case m.roles[ii]&wordStart != 0:
				baseScore = wordScore
			}

			// Apply the segment-depth penalty (segments from the right).
			runeScore := baseScore * (1.0 - float64(m.segments[ii])*perSegment)
			if debug {
				runeScores = append(runeScores, runeScore)
				runeIdxs = append(runeIdxs, int(ii))
			}
			totScore += runeScore
			if finalRune {
				break
			}
		}
	}

	if debug {
		// Format rune roles and scores in line:
		// fo[o:.52].[b:1]a[r:.6]
		var summary bytes.Buffer
		last := 0
		for i, idx := range runeIdxs {
			summary.WriteString(string(m.inputBuffer[last:idx])) // encode runes
			fmt.Fprintf(&summary, "[%s:%.2g]", string(m.inputBuffer[idx]), runeScores[i])
			last = idx + 1
		}
		summary.WriteString(string(m.inputBuffer[last:inputLen])) // encode runes
		log.Println(summary.String())
	}

	return start, totScore / float64(m.patternLen)
}

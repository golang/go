// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fuzzy implements a fuzzy matching algorithm.
package fuzzy

import (
	"bytes"
	"fmt"
)

const (
	// MaxInputSize is the maximum size of the input scored against the fuzzy matcher. Longer inputs
	// will be truncated to this size.
	MaxInputSize = 127
	// MaxPatternSize is the maximum size of the pattern used to construct the fuzzy matcher. Longer
	// inputs are truncated to this size.
	MaxPatternSize = 63
)

type scoreVal int

func (s scoreVal) val() int {
	return int(s) >> 1
}

func (s scoreVal) prevK() int {
	return int(s) & 1
}

func score(val int, prevK int /*0 or 1*/) scoreVal {
	return scoreVal(val<<1 + prevK)
}

// Matcher implements a fuzzy matching algorithm for scoring candidates against a pattern.
// The matcher does not support parallel usage.
type Matcher struct {
	pattern       string
	patternLower  []byte // lower-case version of the pattern
	patternShort  []byte // first characters of the pattern
	caseSensitive bool   // set if the pattern is mix-cased

	patternRoles []RuneRole // the role of each character in the pattern
	roles        []RuneRole // the role of each character in the tested string

	scores [MaxInputSize + 1][MaxPatternSize + 1][2]scoreVal

	scoreScale float32

	lastCandidateLen     int // in bytes
	lastCandidateMatched bool

	// Here we save the last candidate in lower-case. This is basically a byte slice we reuse for
	// performance reasons, so the slice is not reallocated for every candidate.
	lowerBuf [MaxInputSize]byte
	rolesBuf [MaxInputSize]RuneRole
}

func (m *Matcher) bestK(i, j int) int {
	if m.scores[i][j][0].val() < m.scores[i][j][1].val() {
		return 1
	}
	return 0
}

// NewMatcher returns a new fuzzy matcher for scoring candidates against the provided pattern.
func NewMatcher(pattern string) *Matcher {
	if len(pattern) > MaxPatternSize {
		pattern = pattern[:MaxPatternSize]
	}

	m := &Matcher{
		pattern:      pattern,
		patternLower: ToLower(pattern, nil),
	}

	for i, c := range m.patternLower {
		if pattern[i] != c {
			m.caseSensitive = true
			break
		}
	}

	if len(pattern) > 3 {
		m.patternShort = m.patternLower[:3]
	} else {
		m.patternShort = m.patternLower
	}

	m.patternRoles = RuneRoles(pattern, nil)

	if len(pattern) > 0 {
		maxCharScore := 4
		m.scoreScale = 1 / float32(maxCharScore*len(pattern))
	}

	return m
}

// Score returns the score returned by matching the candidate to the pattern.
// This is not designed for parallel use. Multiple candidates must be scored sequentially.
// Returns a score between 0 and 1 (0 - no match, 1 - perfect match).
func (m *Matcher) Score(candidate string) float32 {
	if len(candidate) > MaxInputSize {
		candidate = candidate[:MaxInputSize]
	}
	lower := ToLower(candidate, m.lowerBuf[:])
	m.lastCandidateLen = len(candidate)

	if len(m.pattern) == 0 {
		// Empty patterns perfectly match candidates.
		return 1
	}

	if m.match(candidate, lower) {
		sc := m.computeScore(candidate, lower)
		if sc > minScore/2 && !m.poorMatch() {
			m.lastCandidateMatched = true
			if len(m.pattern) == len(candidate) {
				// Perfect match.
				return 1
			}

			if sc < 0 {
				sc = 0
			}
			normalizedScore := float32(sc) * m.scoreScale
			if normalizedScore > 1 {
				normalizedScore = 1
			}

			return normalizedScore
		}
	}

	m.lastCandidateMatched = false
	return -1
}

const minScore = -10000

// MatchedRanges returns matches ranges for the last scored string as a flattened array of
// [begin, end) byte offset pairs.
func (m *Matcher) MatchedRanges() []int {
	if len(m.pattern) == 0 || !m.lastCandidateMatched {
		return nil
	}
	i, j := m.lastCandidateLen, len(m.pattern)
	if m.scores[i][j][0].val() < minScore/2 && m.scores[i][j][1].val() < minScore/2 {
		return nil
	}

	var ret []int
	k := m.bestK(i, j)
	for i > 0 {
		take := (k == 1)
		k = m.scores[i][j][k].prevK()
		if take {
			if len(ret) == 0 || ret[len(ret)-1] != i {
				ret = append(ret, i)
				ret = append(ret, i-1)
			} else {
				ret[len(ret)-1] = i - 1
			}
			j--
		}
		i--
	}
	// Reverse slice.
	for i := 0; i < len(ret)/2; i++ {
		ret[i], ret[len(ret)-1-i] = ret[len(ret)-1-i], ret[i]
	}
	return ret
}

func (m *Matcher) match(candidate string, candidateLower []byte) bool {
	i, j := 0, 0
	for ; i < len(candidateLower) && j < len(m.patternLower); i++ {
		if candidateLower[i] == m.patternLower[j] {
			j++
		}
	}
	if j != len(m.patternLower) {
		return false
	}

	// The input passes the simple test against pattern, so it is time to classify its characters.
	// Character roles are used below to find the last segment.
	m.roles = RuneRoles(candidate, m.rolesBuf[:])

	return true
}

func (m *Matcher) computeScore(candidate string, candidateLower []byte) int {
	pattLen, candLen := len(m.pattern), len(candidate)

	for j := 0; j <= len(m.pattern); j++ {
		m.scores[0][j][0] = minScore << 1
		m.scores[0][j][1] = minScore << 1
	}
	m.scores[0][0][0] = score(0, 0) // Start with 0.

	segmentsLeft, lastSegStart := 1, 0
	for i := 0; i < candLen; i++ {
		if m.roles[i] == RSep {
			segmentsLeft++
			lastSegStart = i + 1
		}
	}

	// A per-character bonus for a consecutive match.
	consecutiveBonus := 2
	wordIdx := 0 // Word count within segment.
	for i := 1; i <= candLen; i++ {

		role := m.roles[i-1]
		isHead := role == RHead

		if isHead {
			wordIdx++
		} else if role == RSep && segmentsLeft > 1 {
			wordIdx = 0
			segmentsLeft--
		}

		var skipPenalty int
		if i == 1 || (i-1) == lastSegStart {
			// Skipping the start of first or last segment.
			skipPenalty += 1
		}

		for j := 0; j <= pattLen; j++ {
			// By default, we don't have a match. Fill in the skip data.
			m.scores[i][j][1] = minScore << 1

			// Compute the skip score.
			k := 0
			if m.scores[i-1][j][0].val() < m.scores[i-1][j][1].val() {
				k = 1
			}

			skipScore := m.scores[i-1][j][k].val()
			// Do not penalize missing characters after the last matched segment.
			if j != pattLen {
				skipScore -= skipPenalty
			}
			m.scores[i][j][0] = score(skipScore, k)

			if j == 0 || candidateLower[i-1] != m.patternLower[j-1] {
				// Not a match.
				continue
			}
			pRole := m.patternRoles[j-1]

			if role == RTail && pRole == RHead {
				if j > 1 {
					// Not a match: a head in the pattern matches a tail character in the candidate.
					continue
				}
				// Special treatment for the first character of the pattern. We allow
				// matches in the middle of a word if they are long enough, at least
				// min(3, pattern.length) characters.
				if !bytes.HasPrefix(candidateLower[i-1:], m.patternShort) {
					continue
				}
			}

			// Compute the char score.
			var charScore int
			// Bonus 1: the char is in the candidate's last segment.
			if segmentsLeft <= 1 {
				charScore++
			}
			// Bonus 2: Case match or a Head in the pattern aligns with one in the word.
			// Single-case patterns lack segmentation signals and we assume any character
			// can be a head of a segment.
			if candidate[i-1] == m.pattern[j-1] || role == RHead && (!m.caseSensitive || pRole == RHead) {
				charScore++
			}

			// Penalty 1: pattern char is Head, candidate char is Tail.
			if role == RTail && pRole == RHead {
				charScore--
			}
			// Penalty 2: first pattern character matched in the middle of a word.
			if j == 1 && role == RTail {
				charScore -= 4
			}

			// Third dimension encodes whether there is a gap between the previous match and the current
			// one.
			for k := 0; k < 2; k++ {
				sc := m.scores[i-1][j-1][k].val() + charScore

				isConsecutive := k == 1 || i-1 == 0 || i-1 == lastSegStart
				if isConsecutive {
					// Bonus 3: a consecutive match. First character match also gets a bonus to
					// ensure prefix final match score normalizes to 1.0.
					// Logically, this is a part of charScore, but we have to compute it here because it
					// only applies for consecutive matches (k == 1).
					sc += consecutiveBonus
				}
				if k == 0 {
					// Penalty 3: Matching inside a segment (and previous char wasn't matched). Penalize for the lack
					// of alignment.
					if role == RTail || role == RUCTail {
						sc -= 3
					}
				}

				if sc > m.scores[i][j][1].val() {
					m.scores[i][j][1] = score(sc, k)
				}
			}
		}
	}

	result := m.scores[len(candidate)][len(m.pattern)][m.bestK(len(candidate), len(m.pattern))].val()

	return result
}

// ScoreTable returns the score table computed for the provided candidate. Used only for debugging.
func (m *Matcher) ScoreTable(candidate string) string {
	var buf bytes.Buffer

	var line1, line2, separator bytes.Buffer
	line1.WriteString("\t")
	line2.WriteString("\t")
	for j := 0; j < len(m.pattern); j++ {
		line1.WriteString(fmt.Sprintf("%c\t\t", m.pattern[j]))
		separator.WriteString("----------------")
	}

	buf.WriteString(line1.String())
	buf.WriteString("\n")
	buf.WriteString(separator.String())
	buf.WriteString("\n")

	for i := 1; i <= len(candidate); i++ {
		line1.Reset()
		line2.Reset()

		line1.WriteString(fmt.Sprintf("%c\t", candidate[i-1]))
		line2.WriteString("\t")

		for j := 1; j <= len(m.pattern); j++ {
			line1.WriteString(fmt.Sprintf("M%6d(%c)\t", m.scores[i][j][0].val(), dir(m.scores[i][j][0].prevK())))
			line2.WriteString(fmt.Sprintf("H%6d(%c)\t", m.scores[i][j][1].val(), dir(m.scores[i][j][1].prevK())))
		}
		buf.WriteString(line1.String())
		buf.WriteString("\n")
		buf.WriteString(line2.String())
		buf.WriteString("\n")
		buf.WriteString(separator.String())
		buf.WriteString("\n")
	}

	return buf.String()
}

func dir(prevK int) rune {
	if prevK == 0 {
		return 'M'
	}
	return 'H'
}

func (m *Matcher) poorMatch() bool {
	if len(m.pattern) < 2 {
		return false
	}

	i, j := m.lastCandidateLen, len(m.pattern)
	k := m.bestK(i, j)

	var counter, len int
	for i > 0 {
		take := (k == 1)
		k = m.scores[i][j][k].prevK()
		if take {
			len++
			if k == 0 && len < 3 && m.roles[i-1] == RTail {
				// Short match in the middle of a word
				counter++
				if counter > 1 {
					return true
				}
			}
			j--
		} else {
			len = 0
		}
		i--
	}
	return false
}

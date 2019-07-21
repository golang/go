// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package path

import (
	"errors"
	"strings"
	"unicode/utf8"
)

// ErrBadPattern indicates a pattern was malformed.
var ErrBadPattern = errors.New("syntax error in pattern")

type patternChunk struct {
	star    bool
	chunk   string
	pending bool // pending pattern text after this chunk
}

// Match reports whether name matches the shell pattern.
// The pattern syntax is:
//
//	pattern:
//		{ term }
//	term:
//		'*'         matches any sequence of non-/ characters
//		'?'         matches any single non-/ character
//		'[' [ '^' ] { character-range } ']'
//		            character class (must be non-empty)
//		c           matches character c (c != '*', '?', '\\', '[')
//		'\\' c      matches character c
//
//	character-range:
//		c           matches character c (c != '\\', '-', ']')
//		'\\' c      matches character c
//		lo '-' hi   matches character c for lo <= c <= hi
//
// Match requires pattern to match all of name, not just a substring.
// The only possible returned error is ErrBadPattern, when pattern
// is malformed.
//
func Match(pattern, name string) (matched bool, err error) {
	pcs, err := collectChunks(pattern)
	if err != nil {
		return false, err
	}
Pattern:
	for _, pc := range pcs {
		if err != nil {
			return false, err
		}
		if pc.star && pc.chunk == "" {
			// Trailing * matches rest of string unless it has a /.
			return !strings.Contains(name, "/"), nil
		}
		// Look for match at current position.
		t, ok, err := matchChunk(pc.chunk, name)
		// if we're the last chunk, make sure we've exhausted the name
		// otherwise we'll give a false result even if we could still match
		// using the star
		if ok && (len(t) == 0 || pc.pending) {
			name = t
			continue
		}
		if err != nil {
			return false, err
		}
		if pc.star {
			// Look for match skipping i+1 bytes.
			// Cannot skip /.
			for i := 0; i < len(name) && name[i] != '/'; i++ {
				t, ok, err := matchChunk(pc.chunk, name[i+1:])
				if ok {
					// if we're the last chunk, make sure we exhausted the name
					if !pc.pending && len(t) > 0 {
						continue
					}
					name = t
					continue Pattern
				}
				if err != nil {
					return false, err
				}
			}
		}
		return false, nil
	}
	return len(name) == 0, nil
}

// collectChunks gets all the segments of the pattern, which are a non-star string
// possibly preceded by a star. This collector enforces check the whole
// pattern syntax but saving the segments for future use in the match process
func collectChunks(pattern string) (pcs []patternChunk, err error) {
	pcs = []patternChunk{}
	for len(pattern) > 0 {
		var star bool
		var chunk string
		star, chunk, pattern, err = scanChunk(pattern)
		if err != nil {
			return pcs, err
		}
		pcs = append(pcs, patternChunk{star, chunk, len(pattern) > 0})
	}
	return

}

// scanChunk gets the next segment and validate the syntax
func scanChunk(pattern string) (star bool, chunk, rest string, err error) {
	for len(pattern) > 0 && pattern[0] == '*' {
		pattern = pattern[1:]
		star = true
	}
	inrange := false
	hasRangeStart := false
	hasHyphen := false
	var i int
Scan:
	for i = 0; i < len(pattern); i++ {
		switch pattern[i] {
		case '\\':
			if i+1 == len(pattern) { // nothing after the escape
				return false, "", "", ErrBadPattern
			}
			if inrange {
				hasRangeStart = true
			}
			i++
		case '[':
			if inrange || i+1 == len(pattern) { // already in rage or opened at last position
				return false, "", "", ErrBadPattern
			}
			if pattern[i+1] == '^' { // negation
				i++
			}
			inrange = true
		case ']':
			if !inrange || !hasRangeStart { // not started or empty range
				return false, "", "", ErrBadPattern
			}
			inrange = false
			hasRangeStart = false
			hasHyphen = false
		case '^':
			if !inrange || hasRangeStart { // allowed only in a range start
				return false, "", "", ErrBadPattern
			}
		case '?', '/':
			if inrange { // not allowed in a range
				return false, "", "", ErrBadPattern
			}
		case '-':
			if inrange && (!hasRangeStart || hasHyphen) { // allowed only one in a range
				return false, "", "", ErrBadPattern
			}
			if i+1 == len(pattern) || pattern[i+1] == ']' { // no high ch range
				return false, "", "", ErrBadPattern
			}
			hasHyphen = true
		case '*':
			if inrange { // not allowed in a range
				return false, "", "", ErrBadPattern
			}
			break Scan
		default:
			if inrange {
				hasRangeStart = true
			}
		}
	}
	if inrange {
		return false, "", "", ErrBadPattern
	}
	return star, pattern[0:i], pattern[i:], nil
}

// matchChunk checks whether chunk matches the beginning of s.
// If so, it returns the remainder of s (after the match).
// Chunk is all single-character operators: literals, char classes, and ?.
func matchChunk(chunk, s string) (rest string, ok bool, err error) {
	for len(chunk) > 0 {
		if len(s) == 0 {
			return
		}
		switch chunk[0] {
		case '[':
			// character class
			r, n := utf8.DecodeRuneInString(s)
			s = s[n:]
			chunk = chunk[1:]
			// possibly negated
			notNegated := true
			if len(chunk) > 0 && chunk[0] == '^' {
				notNegated = false
				chunk = chunk[1:]
			}
			// parse all ranges
			match := false
			nrange := 0
			for {
				if len(chunk) > 0 && chunk[0] == ']' && nrange > 0 {
					chunk = chunk[1:]
					break
				}
				var lo, hi rune
				if lo, chunk, err = getEsc(chunk); err != nil {
					return
				}
				hi = lo
				if chunk[0] == '-' {
					if hi, chunk, err = getEsc(chunk[1:]); err != nil {
						return
					}
				}
				if lo <= r && r <= hi {
					match = true
				}
				nrange++
			}
			if match != notNegated {
				return
			}

		case '?':
			if s[0] == '/' {
				return
			}
			_, n := utf8.DecodeRuneInString(s)
			s = s[n:]
			chunk = chunk[1:]

		case '\\':
			chunk = chunk[1:]
			fallthrough

		default:
			if chunk[0] != s[0] {
				return
			}
			s = s[1:]
			chunk = chunk[1:]
		}
	}
	return s, true, nil
}

// getEsc gets a possibly-escaped character from chunk, for a character class.
func getEsc(chunk string) (r rune, nchunk string, err error) {
	if chunk[0] == '\\' {
		chunk = chunk[1:]
	}
	r, n := utf8.DecodeRuneInString(chunk)
	if r == utf8.RuneError && n == 1 {
		err = ErrBadPattern
	}
	nchunk = chunk[n:]
	if len(nchunk) == 0 {
		err = ErrBadPattern
	}
	return
}

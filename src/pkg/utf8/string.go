// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utf8

import "errors"

// String wraps a regular string with a small structure that provides more
// efficient indexing by code point index, as opposed to byte index.
// Scanning incrementally forwards or backwards is O(1) per index operation
// (although not as fast a range clause going forwards).  Random access is
// O(N) in the length of the string, but the overhead is less than always
// scanning from the beginning.
// If the string is ASCII, random access is O(1).
// Unlike the built-in string type, String has internal mutable state and
// is not thread-safe.
type String struct {
	str      string
	numRunes int
	// If width > 0, the rune at runePos starts at bytePos and has the specified width.
	width    int
	bytePos  int
	runePos  int
	nonASCII int // byte index of the first non-ASCII rune.
}

// NewString returns a new UTF-8 string with the provided contents.
func NewString(contents string) *String {
	return new(String).Init(contents)
}

// Init initializes an existing String to hold the provided contents.
// It returns a pointer to the initialized String.
func (s *String) Init(contents string) *String {
	s.str = contents
	s.bytePos = 0
	s.runePos = 0
	for i := 0; i < len(contents); i++ {
		if contents[i] >= RuneSelf {
			// Not ASCII.
			s.numRunes = RuneCountInString(contents)
			_, s.width = DecodeRuneInString(contents)
			s.nonASCII = i
			return s
		}
	}
	// ASCII is simple.  Also, the empty string is ASCII.
	s.numRunes = len(contents)
	s.width = 0
	s.nonASCII = len(contents)
	return s
}

// String returns the contents of the String.  This method also means the
// String is directly printable by fmt.Print.
func (s *String) String() string {
	return s.str
}

// RuneCount returns the number of runes (Unicode code points) in the String.
func (s *String) RuneCount() int {
	return s.numRunes
}

// IsASCII returns a boolean indicating whether the String contains only ASCII bytes.
func (s *String) IsASCII() bool {
	return s.width == 0
}

// Slice returns the string sliced at rune positions [i:j].
func (s *String) Slice(i, j int) string {
	// ASCII is easy.  Let the compiler catch the indexing error if there is one.
	if j < s.nonASCII {
		return s.str[i:j]
	}
	if i < 0 || j > s.numRunes || i > j {
		panic(sliceOutOfRange)
	}
	if i == j {
		return ""
	}
	// For non-ASCII, after At(i), bytePos is always the position of the indexed character.
	var low, high int
	switch {
	case i < s.nonASCII:
		low = i
	case i == s.numRunes:
		low = len(s.str)
	default:
		s.At(i)
		low = s.bytePos
	}
	switch {
	case j == s.numRunes:
		high = len(s.str)
	default:
		s.At(j)
		high = s.bytePos
	}
	return s.str[low:high]
}

// At returns the rune with index i in the String.  The sequence of runes is the same
// as iterating over the contents with a "for range" clause.
func (s *String) At(i int) rune {
	// ASCII is easy.  Let the compiler catch the indexing error if there is one.
	if i < s.nonASCII {
		return rune(s.str[i])
	}

	// Now we do need to know the index is valid.
	if i < 0 || i >= s.numRunes {
		panic(outOfRange)
	}

	var r rune

	// Five easy common cases: within 1 spot of bytePos/runePos, or the beginning, or the end.
	// With these cases, all scans from beginning or end work in O(1) time per rune.
	switch {

	case i == s.runePos-1: // backing up one rune
		r, s.width = DecodeLastRuneInString(s.str[0:s.bytePos])
		s.runePos = i
		s.bytePos -= s.width
		return r
	case i == s.runePos+1: // moving ahead one rune
		s.runePos = i
		s.bytePos += s.width
		fallthrough
	case i == s.runePos:
		r, s.width = DecodeRuneInString(s.str[s.bytePos:])
		return r
	case i == 0: // start of string
		r, s.width = DecodeRuneInString(s.str)
		s.runePos = 0
		s.bytePos = 0
		return r

	case i == s.numRunes-1: // last rune in string
		r, s.width = DecodeLastRuneInString(s.str)
		s.runePos = i
		s.bytePos = len(s.str) - s.width
		return r
	}

	// We need to do a linear scan.  There are three places to start from:
	// 1) The beginning
	// 2) bytePos/runePos.
	// 3) The end
	// Choose the closest in rune count, scanning backwards if necessary.
	forward := true
	if i < s.runePos {
		// Between beginning and pos.  Which is closer?
		// Since both i and runePos are guaranteed >= nonASCII, that's the
		// lowest location we need to start from.
		if i < (s.runePos-s.nonASCII)/2 {
			// Scan forward from beginning
			s.bytePos, s.runePos = s.nonASCII, s.nonASCII
		} else {
			// Scan backwards from where we are
			forward = false
		}
	} else {
		// Between pos and end.  Which is closer?
		if i-s.runePos < (s.numRunes-s.runePos)/2 {
			// Scan forward from pos
		} else {
			// Scan backwards from end
			s.bytePos, s.runePos = len(s.str), s.numRunes
			forward = false
		}
	}
	if forward {
		// TODO: Is it much faster to use a range loop for this scan?
		for {
			r, s.width = DecodeRuneInString(s.str[s.bytePos:])
			if s.runePos == i {
				break
			}
			s.runePos++
			s.bytePos += s.width
		}
	} else {
		for {
			r, s.width = DecodeLastRuneInString(s.str[0:s.bytePos])
			s.runePos--
			s.bytePos -= s.width
			if s.runePos == i {
				break
			}
		}
	}
	return r
}

var outOfRange = errors.New("utf8.String: index out of range")
var sliceOutOfRange = errors.New("utf8.String: slice index out of range")

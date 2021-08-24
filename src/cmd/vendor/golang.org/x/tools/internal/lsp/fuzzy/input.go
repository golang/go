// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzzy

import (
	"unicode"
)

// RuneRole specifies the role of a rune in the context of an input.
type RuneRole byte

const (
	// RNone specifies a rune without any role in the input (i.e., whitespace/non-ASCII).
	RNone RuneRole = iota
	// RSep specifies a rune with the role of segment separator.
	RSep
	// RTail specifies a rune which is a lower-case tail in a word in the input.
	RTail
	// RUCTail specifies a rune which is an upper-case tail in a word in the input.
	RUCTail
	// RHead specifies a rune which is the first character in a word in the input.
	RHead
)

// RuneRoles detects the roles of each byte rune in an input string and stores it in the output
// slice. The rune role depends on the input type. Stops when it parsed all the runes in the string
// or when it filled the output. If output is nil, then it gets created.
func RuneRoles(candidate []byte, reuse []RuneRole) []RuneRole {
	var output []RuneRole
	if cap(reuse) < len(candidate) {
		output = make([]RuneRole, 0, len(candidate))
	} else {
		output = reuse[:0]
	}

	prev, prev2 := rtNone, rtNone
	for i := 0; i < len(candidate); i++ {
		r := rune(candidate[i])

		role := RNone

		curr := rtLower
		if candidate[i] <= unicode.MaxASCII {
			curr = runeType(rt[candidate[i]] - '0')
		}

		if curr == rtLower {
			if prev == rtNone || prev == rtPunct {
				role = RHead
			} else {
				role = RTail
			}
		} else if curr == rtUpper {
			role = RHead

			if prev == rtUpper {
				// This and previous characters are both upper case.

				if i+1 == len(candidate) {
					// This is last character, previous was also uppercase -> this is UCTail
					// i.e., (current char is C): aBC / BC / ABC
					role = RUCTail
				}
			}
		} else if curr == rtPunct {
			switch r {
			case '.', ':':
				role = RSep
			}
		}
		if curr != rtLower {
			if i > 1 && output[i-1] == RHead && prev2 == rtUpper && (output[i-2] == RHead || output[i-2] == RUCTail) {
				// The previous two characters were uppercase. The current one is not a lower case, so the
				// previous one can't be a HEAD. Make it a UCTail.
				// i.e., (last char is current char - B must be a UCTail): ABC / ZABC / AB.
				output[i-1] = RUCTail
			}
		}

		output = append(output, role)
		prev2 = prev
		prev = curr
	}
	return output
}

type runeType byte

const (
	rtNone runeType = iota
	rtPunct
	rtLower
	rtUpper
)

const rt = "00000000000000000000000000000000000000000000001122222222221000000333333333333333333333333330000002222222222222222222222222200000"

// LastSegment returns the substring representing the last segment from the input, where each
// byte has an associated RuneRole in the roles slice. This makes sense only for inputs of Symbol
// or Filename type.
func LastSegment(input string, roles []RuneRole) string {
	// Exclude ending separators.
	end := len(input) - 1
	for end >= 0 && roles[end] == RSep {
		end--
	}
	if end < 0 {
		return ""
	}

	start := end - 1
	for start >= 0 && roles[start] != RSep {
		start--
	}

	return input[start+1 : end+1]
}

// fromChunks copies string chunks into the given buffer.
func fromChunks(chunks []string, buffer []byte) []byte {
	ii := 0
	for _, chunk := range chunks {
		for i := 0; i < len(chunk); i++ {
			if ii >= cap(buffer) {
				break
			}
			buffer[ii] = chunk[i]
			ii++
		}
	}
	return buffer[:ii]
}

// toLower transforms the input string to lower case, which is stored in the output byte slice.
// The lower casing considers only ASCII values - non ASCII values are left unmodified.
// Stops when parsed all input or when it filled the output slice. If output is nil, then it gets
// created.
func toLower(input []byte, reuse []byte) []byte {
	output := reuse
	if cap(reuse) < len(input) {
		output = make([]byte, len(input))
	}

	for i := 0; i < len(input); i++ {
		r := rune(input[i])
		if input[i] <= unicode.MaxASCII {
			if 'A' <= r && r <= 'Z' {
				r += 'a' - 'A'
			}
		}
		output[i] = byte(r)
	}
	return output[:len(input)]
}

// WordConsumer defines a consumer for a word delimited by the [start,end) byte offsets in an input
// (start is inclusive, end is exclusive).
type WordConsumer func(start, end int)

// Words find word delimiters in an input based on its bytes' mappings to rune roles. The offset
// delimiters for each word are fed to the provided consumer function.
func Words(roles []RuneRole, consume WordConsumer) {
	var wordStart int
	for i, r := range roles {
		switch r {
		case RUCTail, RTail:
		case RHead, RNone, RSep:
			if i != wordStart {
				consume(wordStart, i)
			}
			wordStart = i
			if r != RHead {
				// Skip this character.
				wordStart = i + 1
			}
		}
	}
	if wordStart != len(roles) {
		consume(wordStart, len(roles))
	}
}

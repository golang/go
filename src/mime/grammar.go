// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

const tspecialsString = `()<>@,;:\"/[]?=`

var (
	tspecials  asciiSet
	tokenChars asciiSet
)

func init() {
	// tspecials :=  "(" / ")" / "<" / ">" / "@" /
	//               "," / ";" / ":" / "\" / <">
	//               "/" / "[" / "]" / "?" / "="
	tspecials.add(tspecialsString)

	// token := 1*<any (US-ASCII) CHAR except SPACE, CTLs,
	//             or tspecials>
	tokenChars.addRange('!', 0x7f)
	tokenChars.remove(tspecialsString)
}

// isTSpecial reports whether c is in 'tspecials' as defined by RFC
// 1521 and RFC 2045.
func isTSpecial(c byte) bool {
	return tspecials.contains(c)
}

// isTokenChar reports whether c is in 'token' as defined by RFC
// 1521 and RFC 2045.
func isTokenChar(c byte) bool {
	return tokenChars.contains(c)
}

// isToken reports whether s is a 'token' as defined by RFC 1521
// and RFC 2045.
func isToken(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range []byte(s) {
		if !tokenChars.contains(c) {
			return false
		}
	}
	return true
}

// asciiSet is a 32-byte value, where each bit represents the presence of a
// given ASCII character in the set. The 128-bits of the lower 16 bytes,
// starting with the least-significant bit of the lowest word to the
// most-significant bit of the highest word, map to the full range of all
// 128 ASCII characters. The 128-bits of the upper 16 bytes will be zeroed,
// ensuring that any non-ASCII character will be reported as not in the set.
// This allocates a total of 32 bytes even though the upper half
// is unused to avoid bounds checks in asciiSet.contains.
type asciiSet [8]uint32

// add adds all the characters in chars to the set.
// Precondition: all the characters in chars are ASCII.
func (as *asciiSet) add(chars string) {
	for _, c := range []byte(chars) {
		as[c/32] |= 1 << (c % 32)
	}
}

// addRange adds all the characters between lo (inclusive) and hi (exclusive) to the set.
// Precondition: hi <= utf8.RuneSelf (0x80)
func (as *asciiSet) addRange(lo, hi byte) {
	for c := lo; c < hi; c++ {
		as[c/32] |= 1 << (c % 32)
	}
}

// remove removes all the characters in chars from the set.
func (as *asciiSet) remove(chars string) {
	for _, c := range []byte(chars) {
		as[c/32] &^= 1 << (c % 32)
	}
}

// contains reports whether c is inside the set.
func (as *asciiSet) contains(c byte) bool {
	return (as[c/32] & (1 << (c % 32))) != 0
}

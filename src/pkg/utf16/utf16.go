// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package utf16 implements encoding and decoding of UTF-16 sequences.
package utf16

import "unicode"

const (
	// 0xd800-0xdc00 encodes the high 10 bits of a pair.
	// 0xdc00-0xe000 encodes the low 10 bits of a pair.
	// the value is those 20 bits plus 0x10000.
	surr1 = 0xd800
	surr2 = 0xdc00
	surr3 = 0xe000

	surrSelf = 0x10000
)

// IsSurrogate returns true if the specified Unicode code point
// can appear in a surrogate pair.
func IsSurrogate(rune int) bool {
	return surr1 <= rune && rune < surr3
}

// DecodeRune returns the UTF-16 decoding of a surrogate pair.
// If the pair is not a valid UTF-16 surrogate pair, DecodeRune returns
// the Unicode replacement code point U+FFFD.
func DecodeRune(r1, r2 int) int {
	if surr1 <= r1 && r1 < surr2 && surr2 <= r2 && r2 < surr3 {
		return (int(r1)-surr1)<<10 | (int(r2) - surr2) + 0x10000
	}
	return unicode.ReplacementChar
}

// EncodeRune returns the UTF-16 surrogate pair r1, r2 for the given rune.
// If the rune is not a valid Unicode code point or does not need encoding,
// EncodeRune returns U+FFFD, U+FFFD.
func EncodeRune(rune int) (r1, r2 int) {
	if rune < surrSelf || rune > unicode.MaxRune || IsSurrogate(rune) {
		return unicode.ReplacementChar, unicode.ReplacementChar
	}
	rune -= surrSelf
	return surr1 + (rune>>10)&0x3ff, surr2 + rune&0x3ff
}

// Encode returns the UTF-16 encoding of the Unicode code point sequence s.
func Encode(s []int) []uint16 {
	n := len(s)
	for _, v := range s {
		if v >= surrSelf {
			n++
		}
	}

	a := make([]uint16, n)
	n = 0
	for _, v := range s {
		switch {
		case v < 0, surr1 <= v && v < surr3, v > unicode.MaxRune:
			v = unicode.ReplacementChar
			fallthrough
		case v < surrSelf:
			a[n] = uint16(v)
			n++
		default:
			r1, r2 := EncodeRune(v)
			a[n] = uint16(r1)
			a[n+1] = uint16(r2)
			n += 2
		}
	}
	return a[0:n]
}

// Decode returns the Unicode code point sequence represented
// by the UTF-16 encoding s.
func Decode(s []uint16) []int {
	a := make([]int, len(s))
	n := 0
	for i := 0; i < len(s); i++ {
		switch r := s[i]; {
		case surr1 <= r && r < surr2 && i+1 < len(s) &&
			surr2 <= s[i+1] && s[i+1] < surr3:
			// valid surrogate sequence
			a[n] = DecodeRune(int(r), int(s[i+1]))
			i++
			n++
		case surr1 <= r && r < surr3:
			// invalid surrogate sequence
			a[n] = unicode.ReplacementChar
			n++
		default:
			// normal rune
			a[n] = int(r)
			n++
		}
	}
	return a[0:n]
}

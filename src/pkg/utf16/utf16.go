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
			v -= surrSelf
			a[n] = uint16(surr1 + (v>>10)&0x3ff)
			a[n+1] = uint16(surr2 + v&0x3ff)
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
			a[n] = (int(r)-surr1)<<10 | (int(s[i+1]) - surr2) + 0x10000
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

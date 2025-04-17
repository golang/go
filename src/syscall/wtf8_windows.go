// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows UTF-16 strings can contain unpaired surrogates, which can't be
// decoded into a valid UTF-8 string. This file defines a set of functions
// that can be used to encode and decode potentially ill-formed UTF-16 strings
// by using the [the WTF-8 encoding](https://simonsapin.github.io/wtf-8/).
//
// WTF-8 is a strict superset of UTF-8, i.e. any string that is
// well-formed in UTF-8 is also well-formed in WTF-8 and the content
// is unchanged. Also, the conversion never fails and is lossless.
//
// The benefit of using WTF-8 instead of UTF-8 when decoding a UTF-16 string
// is that the conversion is lossless even for ill-formed UTF-16 strings.
// This property allows to read an ill-formed UTF-16 string, convert it
// to a Go string, and convert it back to the same original UTF-16 string.
//
// See go.dev/issues/59971 for more info.

package syscall

import (
	"unicode/utf16"
	"unicode/utf8"
)

const (
	surr1 = 0xd800
	surr2 = 0xdc00
	surr3 = 0xe000

	tx    = 0b10000000
	t3    = 0b11100000
	maskx = 0b00111111
	mask3 = 0b00001111

	rune1Max = 1<<7 - 1
	rune2Max = 1<<11 - 1
)

// encodeWTF16 returns the potentially ill-formed
// UTF-16 encoding of s.
func encodeWTF16(s string, buf []uint16) []uint16 {
	for i := 0; i < len(s); {
		// Cannot use 'for range s' because it expects valid
		// UTF-8 runes.
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError {
			// Check if s[i:] contains a valid WTF-8 encoded surrogate.
			if sc := s[i:]; len(sc) >= 3 && sc[0] == 0xED && 0xA0 <= sc[1] && sc[1] <= 0xBF && 0x80 <= sc[2] && sc[2] <= 0xBF {
				r = rune(sc[0]&mask3)<<12 + rune(sc[1]&maskx)<<6 + rune(sc[2]&maskx)
				buf = append(buf, uint16(r))
				i += 3
				continue
			}
		}
		i += size
		buf = utf16.AppendRune(buf, r)
	}
	return buf
}

// decodeWTF16 returns the WTF-8 encoding of
// the potentially ill-formed UTF-16 s.
func decodeWTF16(s []uint16, buf []byte) []byte {
	for i := 0; i < len(s); i++ {
		var ar rune
		switch r := s[i]; {
		case r < surr1, surr3 <= r:
			// normal rune
			ar = rune(r)
		case surr1 <= r && r < surr2 && i+1 < len(s) &&
			surr2 <= s[i+1] && s[i+1] < surr3:
			// valid surrogate sequence
			ar = utf16.DecodeRune(rune(r), rune(s[i+1]))
			i++
		default:
			// WTF-8 fallback.
			// This only handles the 3-byte case of utf8.AppendRune,
			// as surrogates always fall in that case.
			ar = rune(r)
			if ar > utf8.MaxRune {
				ar = utf8.RuneError
			}
			buf = append(buf, t3|byte(ar>>12), tx|byte(ar>>6)&maskx, tx|byte(ar)&maskx)
			continue
		}
		buf = utf8.AppendRune(buf, ar)
	}
	return buf
}

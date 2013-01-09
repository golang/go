// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package utf8 implements functions and constants to support text encoded in
// UTF-8. It includes functions to translate between runes and UTF-8 byte sequences.
package utf8

// The conditions RuneError==unicode.ReplacementChar and
// MaxRune==unicode.MaxRune are verified in the tests.
// Defining them locally avoids this package depending on package unicode.

// Numbers fundamental to the encoding.
const (
	RuneError = '\uFFFD'     // the "error" Rune or "Unicode replacement character"
	RuneSelf  = 0x80         // characters below Runeself are represented as themselves in a single byte.
	MaxRune   = '\U0010FFFF' // Maximum valid Unicode code point.
	UTFMax    = 4            // maximum number of bytes of a UTF-8 encoded Unicode character.
)

// Code points in the surrogate range are not valid for UTF-8.
const (
	surrogateMin = 0xD800
	surrogateMax = 0xDFFF
)

const (
	t1 = 0x00 // 0000 0000
	tx = 0x80 // 1000 0000
	t2 = 0xC0 // 1100 0000
	t3 = 0xE0 // 1110 0000
	t4 = 0xF0 // 1111 0000
	t5 = 0xF8 // 1111 1000

	maskx = 0x3F // 0011 1111
	mask2 = 0x1F // 0001 1111
	mask3 = 0x0F // 0000 1111
	mask4 = 0x07 // 0000 0111

	rune1Max = 1<<7 - 1
	rune2Max = 1<<11 - 1
	rune3Max = 1<<16 - 1
)

func decodeRuneInternal(p []byte) (r rune, size int, short bool) {
	n := len(p)
	if n < 1 {
		return RuneError, 0, true
	}
	c0 := p[0]

	// 1-byte, 7-bit sequence?
	if c0 < tx {
		return rune(c0), 1, false
	}

	// unexpected continuation byte?
	if c0 < t2 {
		return RuneError, 1, false
	}

	// need first continuation byte
	if n < 2 {
		return RuneError, 1, true
	}
	c1 := p[1]
	if c1 < tx || t2 <= c1 {
		return RuneError, 1, false
	}

	// 2-byte, 11-bit sequence?
	if c0 < t3 {
		r = rune(c0&mask2)<<6 | rune(c1&maskx)
		if r <= rune1Max {
			return RuneError, 1, false
		}
		return r, 2, false
	}

	// need second continuation byte
	if n < 3 {
		return RuneError, 1, true
	}
	c2 := p[2]
	if c2 < tx || t2 <= c2 {
		return RuneError, 1, false
	}

	// 3-byte, 16-bit sequence?
	if c0 < t4 {
		r = rune(c0&mask3)<<12 | rune(c1&maskx)<<6 | rune(c2&maskx)
		if r <= rune2Max {
			return RuneError, 1, false
		}
		if surrogateMin <= r && r <= surrogateMax {
			return RuneError, 1, false
		}
		return r, 3, false
	}

	// need third continuation byte
	if n < 4 {
		return RuneError, 1, true
	}
	c3 := p[3]
	if c3 < tx || t2 <= c3 {
		return RuneError, 1, false
	}

	// 4-byte, 21-bit sequence?
	if c0 < t5 {
		r = rune(c0&mask4)<<18 | rune(c1&maskx)<<12 | rune(c2&maskx)<<6 | rune(c3&maskx)
		if r <= rune3Max || MaxRune < r {
			return RuneError, 1, false
		}
		return r, 4, false
	}

	// error
	return RuneError, 1, false
}

func decodeRuneInStringInternal(s string) (r rune, size int, short bool) {
	n := len(s)
	if n < 1 {
		return RuneError, 0, true
	}
	c0 := s[0]

	// 1-byte, 7-bit sequence?
	if c0 < tx {
		return rune(c0), 1, false
	}

	// unexpected continuation byte?
	if c0 < t2 {
		return RuneError, 1, false
	}

	// need first continuation byte
	if n < 2 {
		return RuneError, 1, true
	}
	c1 := s[1]
	if c1 < tx || t2 <= c1 {
		return RuneError, 1, false
	}

	// 2-byte, 11-bit sequence?
	if c0 < t3 {
		r = rune(c0&mask2)<<6 | rune(c1&maskx)
		if r <= rune1Max {
			return RuneError, 1, false
		}
		return r, 2, false
	}

	// need second continuation byte
	if n < 3 {
		return RuneError, 1, true
	}
	c2 := s[2]
	if c2 < tx || t2 <= c2 {
		return RuneError, 1, false
	}

	// 3-byte, 16-bit sequence?
	if c0 < t4 {
		r = rune(c0&mask3)<<12 | rune(c1&maskx)<<6 | rune(c2&maskx)
		if r <= rune2Max {
			return RuneError, 1, false
		}
		if surrogateMin <= r && r <= surrogateMax {
			return RuneError, 1, false
		}
		return r, 3, false
	}

	// need third continuation byte
	if n < 4 {
		return RuneError, 1, true
	}
	c3 := s[3]
	if c3 < tx || t2 <= c3 {
		return RuneError, 1, false
	}

	// 4-byte, 21-bit sequence?
	if c0 < t5 {
		r = rune(c0&mask4)<<18 | rune(c1&maskx)<<12 | rune(c2&maskx)<<6 | rune(c3&maskx)
		if r <= rune3Max || MaxRune < r {
			return RuneError, 1, false
		}
		return r, 4, false
	}

	// error
	return RuneError, 1, false
}

// FullRune reports whether the bytes in p begin with a full UTF-8 encoding of a rune.
// An invalid encoding is considered a full Rune since it will convert as a width-1 error rune.
func FullRune(p []byte) bool {
	_, _, short := decodeRuneInternal(p)
	return !short
}

// FullRuneInString is like FullRune but its input is a string.
func FullRuneInString(s string) bool {
	_, _, short := decodeRuneInStringInternal(s)
	return !short
}

// DecodeRune unpacks the first UTF-8 encoding in p and returns the rune and its width in bytes.
// If the encoding is invalid, it returns (RuneError, 1), an impossible result for correct UTF-8.
// An encoding is invalid if it is incorrect UTF-8, encodes a rune that is
// out of range, or is not the shortest possible UTF-8 encoding for the
// value. No other validation is performed.
func DecodeRune(p []byte) (r rune, size int) {
	r, size, _ = decodeRuneInternal(p)
	return
}

// DecodeRuneInString is like DecodeRune but its input is a string.
// If the encoding is invalid, it returns (RuneError, 1), an impossible result for correct UTF-8.
// An encoding is invalid if it is incorrect UTF-8, encodes a rune that is
// out of range, or is not the shortest possible UTF-8 encoding for the
// value. No other validation is performed.
func DecodeRuneInString(s string) (r rune, size int) {
	r, size, _ = decodeRuneInStringInternal(s)
	return
}

// DecodeLastRune unpacks the last UTF-8 encoding in p and returns the rune and its width in bytes.
// If the encoding is invalid, it returns (RuneError, 1), an impossible result for correct UTF-8.
// An encoding is invalid if it is incorrect UTF-8, encodes a rune that is
// out of range, or is not the shortest possible UTF-8 encoding for the
// value. No other validation is performed.
func DecodeLastRune(p []byte) (r rune, size int) {
	end := len(p)
	if end == 0 {
		return RuneError, 0
	}
	start := end - 1
	r = rune(p[start])
	if r < RuneSelf {
		return r, 1
	}
	// guard against O(n^2) behavior when traversing
	// backwards through strings with long sequences of
	// invalid UTF-8.
	lim := end - UTFMax
	if lim < 0 {
		lim = 0
	}
	for start--; start >= lim; start-- {
		if RuneStart(p[start]) {
			break
		}
	}
	if start < 0 {
		start = 0
	}
	r, size = DecodeRune(p[start:end])
	if start+size != end {
		return RuneError, 1
	}
	return r, size
}

// DecodeLastRuneInString is like DecodeLastRune but its input is a string.
// If the encoding is invalid, it returns (RuneError, 1), an impossible result for correct UTF-8.
// An encoding is invalid if it is incorrect UTF-8, encodes a rune that is
// out of range, or is not the shortest possible UTF-8 encoding for the
// value. No other validation is performed.
func DecodeLastRuneInString(s string) (r rune, size int) {
	end := len(s)
	if end == 0 {
		return RuneError, 0
	}
	start := end - 1
	r = rune(s[start])
	if r < RuneSelf {
		return r, 1
	}
	// guard against O(n^2) behavior when traversing
	// backwards through strings with long sequences of
	// invalid UTF-8.
	lim := end - UTFMax
	if lim < 0 {
		lim = 0
	}
	for start--; start >= lim; start-- {
		if RuneStart(s[start]) {
			break
		}
	}
	if start < 0 {
		start = 0
	}
	r, size = DecodeRuneInString(s[start:end])
	if start+size != end {
		return RuneError, 1
	}
	return r, size
}

// RuneLen returns the number of bytes required to encode the rune.
// It returns -1 if the rune is not a valid value to encode in UTF-8.
func RuneLen(r rune) int {
	switch {
	case r < 0:
		return -1
	case r <= rune1Max:
		return 1
	case r <= rune2Max:
		return 2
	case surrogateMin <= r && r <= surrogateMax:
		return -1
	case r <= rune3Max:
		return 3
	case r <= MaxRune:
		return 4
	}
	return -1
}

// EncodeRune writes into p (which must be large enough) the UTF-8 encoding of the rune.
// It returns the number of bytes written.
func EncodeRune(p []byte, r rune) int {
	// Negative values are erroneous.  Making it unsigned addresses the problem.
	if uint32(r) <= rune1Max {
		p[0] = byte(r)
		return 1
	}

	if uint32(r) <= rune2Max {
		p[0] = t2 | byte(r>>6)
		p[1] = tx | byte(r)&maskx
		return 2
	}

	if uint32(r) > MaxRune {
		r = RuneError
	}

	if surrogateMin <= r && r <= surrogateMax {
		r = RuneError
	}

	if uint32(r) <= rune3Max {
		p[0] = t3 | byte(r>>12)
		p[1] = tx | byte(r>>6)&maskx
		p[2] = tx | byte(r)&maskx
		return 3
	}

	p[0] = t4 | byte(r>>18)
	p[1] = tx | byte(r>>12)&maskx
	p[2] = tx | byte(r>>6)&maskx
	p[3] = tx | byte(r)&maskx
	return 4
}

// RuneCount returns the number of runes in p.  Erroneous and short
// encodings are treated as single runes of width 1 byte.
func RuneCount(p []byte) int {
	i := 0
	var n int
	for n = 0; i < len(p); n++ {
		if p[i] < RuneSelf {
			i++
		} else {
			_, size := DecodeRune(p[i:])
			i += size
		}
	}
	return n
}

// RuneCountInString is like RuneCount but its input is a string.
func RuneCountInString(s string) (n int) {
	for _ = range s {
		n++
	}
	return
}

// RuneStart reports whether the byte could be the first byte of
// an encoded rune.  Second and subsequent bytes always have the top
// two bits set to 10.
func RuneStart(b byte) bool { return b&0xC0 != 0x80 }

// Valid reports whether p consists entirely of valid UTF-8-encoded runes.
func Valid(p []byte) bool {
	i := 0
	for i < len(p) {
		if p[i] < RuneSelf {
			i++
		} else {
			_, size := DecodeRune(p[i:])
			if size == 1 {
				// All valid runes of size 1 (those
				// below RuneSelf) were handled above.
				// This must be a RuneError.
				return false
			}
			i += size
		}
	}
	return true
}

// ValidString reports whether s consists entirely of valid UTF-8-encoded runes.
func ValidString(s string) bool {
	for i, r := range s {
		if r == RuneError {
			// The RuneError value can be an error
			// sentinel value (if it's size 1) or the same
			// value encoded properly. Decode it to see if
			// it's the 1 byte sentinel value.
			_, size := DecodeRuneInString(s[i:])
			if size == 1 {
				return false
			}
		}
	}
	return true
}

// ValidRune reports whether r can be legally encoded as UTF-8.
// Code points that are out of range or a surrogate half are illegal.
func ValidRune(r rune) bool {
	switch {
	case r < 0:
		return false
	case surrogateMin <= r && r <= surrogateMax:
		return false
	case r > MaxRune:
		return false
	}
	return true
}

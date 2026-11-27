// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httpcommon

// The HTTP protocols are defined in terms of ASCII, not Unicode. This file
// contains helper functions which may use Unicode-aware functions which would
// otherwise be unsafe and could introduce vulnerabilities if used improperly.

// asciiEqualFold is strings.EqualFold, ASCII only. It reports whether s and t
// are equal, ASCII-case-insensitively.
func asciiEqualFold(s, t string) bool {
	if len(s) != len(t) {
		return false
	}
	for i := 0; i < len(s); i++ {
		if lower(s[i]) != lower(t[i]) {
			return false
		}
	}
	return true
}

// lower returns the ASCII lowercase version of b.
func lower(b byte) byte {
	if 'A' <= b && b <= 'Z' {
		return b + ('a' - 'A')
	}
	return b
}

// isASCIIPrint returns whether b is ASCII and printable according to
// https://tools.ietf.org/html/rfc20#section-4.2.
func isASCIIPrint(b byte) bool {
	return ' ' <= b && b <= '~'
}

// asciiToLower returns the lowercase version of s if s is ASCII and printable,
// and whether or not it was.
func asciiToLower(s string) (lower string, ok bool) {
	// Check s and find its first uppercase letter in a single pass.
	for i, c := range []byte(s) {
		if !isASCIIPrint(c) {
			return "", false
		}
		if 'A' <= c && c <= 'Z' {
			// Lowercase a copy of s. Most names fit in buf, so the copy does not allocate.
			var buf [64]byte
			b := append(buf[:0], s...)
			b[i] = c + ('a' - 'A')
			for i++; i < len(b); i++ {
				c = b[i]
				if !isASCIIPrint(c) {
					return "", false
				}
				if 'A' <= c && c <= 'Z' {
					b[i] = c + ('a' - 'A')
				}
			}
			return string(b), true
		}
	}
	return s, true
}

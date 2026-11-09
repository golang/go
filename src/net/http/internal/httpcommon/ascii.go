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

// asciiToLower returns the lowercase version of s and true if s is ASCII and
// printable according to https://tools.ietf.org/html/rfc20#section-4.2.
// Otherwise it returns "" and false. If s is already lowercase, it returns s.
func asciiToLower(s string) (lower string, ok bool) {
	// Check s and find its first uppercase letter in a single pass.
	i := 0
	for ; i < len(s); i++ {
		c := s[i]
		if c < ' ' || c > '~' {
			return "", false
		}
		if 'A' <= c && c <= 'Z' {
			break
		}
	}
	if i == len(s) {
		return s, true
	}
	// Lowercase a copy of s. Most names fit in buf, so the copy does not allocate.
	var buf [64]byte
	b := append(buf[:0], s...)
	for ; i < len(b); i++ {
		c := b[i]
		if c < ' ' || c > '~' {
			return "", false
		}
		if 'A' <= c && c <= 'Z' {
			b[i] = c + ('a' - 'A')
		}
	}
	return string(b), true
}

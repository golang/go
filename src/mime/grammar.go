// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

// isTSpecial reports whether c is in 'tspecials' as defined by RFC
// 1521 and RFC 2045.
func isTSpecial(c byte) bool {
	// tspecials :=  "(" / ")" / "<" / ">" / "@" /
	//               "," / ";" / ":" / "\" / <">
	//               "/" / "[" / "]" / "?" / "="
	//
	// mask is a 128-bit bitmap with 1s for allowed bytes,
	// so that the byte c can be tested with a shift and an and.
	// If c >= 128, then 1<<c and 1<<(c-64) will both be zero,
	// and this function will return false.
	const mask = 0 |
		1<<'(' |
		1<<')' |
		1<<'<' |
		1<<'>' |
		1<<'@' |
		1<<',' |
		1<<';' |
		1<<':' |
		1<<'\\' |
		1<<'"' |
		1<<'/' |
		1<<'[' |
		1<<']' |
		1<<'?' |
		1<<'='
	return ((uint64(1)<<c)&(mask&(1<<64-1)) |
		(uint64(1)<<(c-64))&(mask>>64)) != 0
}

// isTokenChar reports whether c is in 'token' as defined by RFC
// 1521 and RFC 2045.
func isTokenChar(c byte) bool {
	// token := 1*<any (US-ASCII) CHAR except SPACE, CTLs,
	//             or tspecials>
	//
	// mask is a 128-bit bitmap with 1s for allowed bytes,
	// so that the byte c can be tested with a shift and an and.
	// If c >= 128, then 1<<c and 1<<(c-64) will both be zero,
	// and this function will return false.
	const mask = 0 |
		(1<<(10)-1)<<'0' |
		(1<<(26)-1)<<'a' |
		(1<<(26)-1)<<'A' |
		1<<'!' |
		1<<'#' |
		1<<'$' |
		1<<'%' |
		1<<'&' |
		1<<'\'' |
		1<<'*' |
		1<<'+' |
		1<<'-' |
		1<<'.' |
		1<<'^' |
		1<<'_' |
		1<<'`' |
		1<<'{' |
		1<<'|' |
		1<<'}' |
		1<<'~'
	return ((uint64(1)<<c)&(mask&(1<<64-1)) |
		(uint64(1)<<(c-64))&(mask>>64)) != 0
}

// isToken reports whether s is a 'token' as defined by RFC 1521
// and RFC 2045.
func isToken(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range []byte(s) {
		if !isTokenChar(c) {
			return false
		}
	}
	return true
}

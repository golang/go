// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"strings"
)

// isTSpecial reports whether rune is in 'tspecials' as defined by RFC
// 1521 and RFC 2045.
func isTSpecial(r rune) bool {
	return strings.ContainsRune(`()<>@,;:\"/[]?=`, r)
}

// isTokenChar reports whether rune is in 'token' as defined by RFC
// 1521 and RFC 2045.
func isTokenChar(r rune) bool {
	// token := 1*<any (US-ASCII) CHAR except SPACE, CTLs,
	//             or tspecials>
	return r > 0x20 && r < 0x7f && !isTSpecial(r)
}

func isNot2616TokenChar(r rune) bool {
	// CHAR           = <any US-ASCII character (octets 0 - 127)>
	// CTL            = <any US-ASCII control character (octets 0 - 31) and DEL (127)>
	// token          = 1*<any CHAR except CTLs or separators>
	// separators     = "(" | ")" | "<" | ">" | "@"
	//                | "," | ";" | ":" | "\" | <">
	//                | "/" | "[" | "]" | "?" | "="
	//                | "{" | "}" | SP | HT
	return r <= 0x20 || r >= 0x7f || strings.ContainsRune(`()<>@,;:\"/[]?={}`, r)
}

// is2616Token reports whether s is a 'token' as defined by RFC 2616,
// which is more restrictive than RFC 1521 and RFC 2045.
func is2616Token(s string) bool {
	if s == "" {
		return false
	}
	return strings.IndexFunc(s, isNot2616TokenChar) < 0
}

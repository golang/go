// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"strings"
)

// isTSpecial returns true if rune is in 'tspecials' as defined by RFC
// 1521 and RFC 2045.
func isTSpecial(r rune) bool {
	return strings.IndexRune(`()<>@,;:\"/[]?=`, r) != -1
}

// isTokenChar returns true if rune is in 'token' as defined by RFC
// 1521 and RFC 2045.
func isTokenChar(r rune) bool {
	// token := 1*<any (US-ASCII) CHAR except SPACE, CTLs,
	//             or tspecials>
	return r > 0x20 && r < 0x7f && !isTSpecial(r)
}

// isToken returns true if s is a 'token' as defined by RFC 1521
// and RFC 2045.
func isToken(s string) bool {
	if s == "" {
		return false
	}
	return strings.IndexFunc(s, isNotTokenChar) < 0
}

// isQText returns true if rune is in 'qtext' as defined by RFC 822.
func isQText(r int) bool {
	// CHAR        =  <any ASCII character>        ; (  0-177,  0.-127.)
	// qtext       =  <any CHAR excepting <">,     ; => may be folded
	//                "\" & CR, and including
	//                linear-white-space>
	switch r {
	case '"', '\\', '\r':
		return false
	}
	return r < 0x80
}

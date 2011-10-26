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

// IsTokenChar returns true if rune is in 'token' as defined by RFC
// 1521 and RFC 2045.
func IsTokenChar(r rune) bool {
	// token := 1*<any (US-ASCII) CHAR except SPACE, CTLs,
	//             or tspecials>
	return r > 0x20 && r < 0x7f && !isTSpecial(r)
}

// IsToken returns true if s is a 'token' as as defined by RFC 1521
// and RFC 2045.
func IsToken(s string) bool {
	if s == "" {
		return false
	}
	return strings.IndexFunc(s, isNotTokenChar) < 0
}

// IsQText returns true if rune is in 'qtext' as defined by RFC 822.
func IsQText(r int) bool {
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

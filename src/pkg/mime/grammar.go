// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"strings"
)

// isTSpecial returns true if rune is in 'tspecials' as defined by RFC
// 1531 and RFC 2045.
func isTSpecial(rune int) bool {
	return strings.IndexRune(`()<>@,;:\"/[]?=`, rune) != -1
}

// IsTokenChar returns true if rune is in 'token' as defined by RFC
// 1531 and RFC 2045.
func IsTokenChar(rune int) bool {
	// token := 1*<any (US-ASCII) CHAR except SPACE, CTLs,
	//             or tspecials>
	return rune > 0x20 && rune < 0x7f && !isTSpecial(rune)
}

// IsQText returns true if rune is in 'qtext' as defined by RFC 822.
func IsQText(rune int) bool {
	// CHAR        =  <any ASCII character>        ; (  0-177,  0.-127.)
	// qtext       =  <any CHAR excepting <">,     ; => may be folded
	//                "\" & CR, and including
	//                linear-white-space>
	switch rune {
	case '"', '\\', '\r':
		return false
	}
	return rune < 0x80
}

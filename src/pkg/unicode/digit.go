// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode

// IsDigit reports whether the rune is a decimal digit.
func IsDigit(rune int) bool {
	if rune < 0x100 { // quick ASCII (Latin-1, really) check
		return '0' <= rune && rune <= '9'
	}
	return Is(Digit, rune)
}

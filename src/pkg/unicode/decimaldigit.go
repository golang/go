// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode

// IsDecimalDigit reports whether the rune is a decimal digit.
func IsDecimalDigit(rune int) bool {
	return Is(DecimalDigit, rune);
}

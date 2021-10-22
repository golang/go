// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import "unsafe"

func GenericShiftTest[DifferentSize ~int8|int16|int64, SameSize int8|byte]() {
	var d DifferentSize
	_ = d << 7
	_ = d << 8        // want "d .may be 8 bits. too small for shift of 8"
	_ = d << 15       // want "d .may be 8 bits. too small for shift of 15"
	_ = (d + 1) << 8  // want ".d . 1. .may be 8 bits. too small for shift of 8"
	_ = (d + 1) << 16 // want ".d . 1. .may be 8 bits. too small for shift of 16"
	_ = d << (7 + 1)  // want "d .may be 8 bits. too small for shift of 8"
	_ = d >> 8        // want "d .may be 8 bits. too small for shift of 8"
	d <<= 8           // want "d .may be 8 bits. too small for shift of 8"
	d >>= 8           // want "d .may be 8 bits. too small for shift of 8"

	// go/types does not compute constant sizes for type parameters, so we do not
	// report a diagnostic here.
	_ = d << (8 * DifferentSize(unsafe.Sizeof(d)))

	var s SameSize
	_ = s << 7
	_ = s << 8        // want "s .8 bits. too small for shift of 8"
	_ = s << (7 + 1)  // want "s .8 bits. too small for shift of 8"
	_ = s >> 8        // want "s .8 bits. too small for shift of 8"
	s <<= 8           // want "s .8 bits. too small for shift of 8"
	s >>= 8           // want "s .8 bits. too small for shift of 8"
}

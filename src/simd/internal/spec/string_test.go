// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

import "testing"

func TestString(t *testing.T) {
	assertFn1[Vec[int32, Width128], string](t, String, "String",
		{{1, -2, 3, -4}, "{1,-2,3,-4}"},
	)
	assertFn1[Vec[uint64, Width128], string](t, String, "String",
		{{1<<64 - 1, 0}, "{18446744073709551615,0}"},
	)
	assertFn1[Vec[float64, Width128], string](t, String, "String",
		{{1.5, -0.25}, "{1.5,-0.25}"},
	)
	assertFn1[Vec[Mask32, Width128], string](t, StringMask, "StringMask",
		{{^Mask32(0), 0, ^Mask32(0), 0}, "{1,0,1,0}"},
	)
	assertFn1[Vec[Mask64, Width128], string](t, StringMask, "StringMask",
		{{0, ^Mask64(0)}, "{0,1}"},
	)
}

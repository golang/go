// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && amd64

package archsimd

// These constants represent the source pattern for the four parameters
// (a, b, c, d) passed to ConcatPermuteScalars and ConcatPermuteScalarsGrouped.
// L means the element comes from the 'x' vector (Low), and
// H means it comes from the 'y' vector (High).
// The order of the letters corresponds to elements a, b, c, d.
// The underlying integer value is a bitmask where:
// Bit 0: Source of element 'a' (0 for x, 1 for y)
// Bit 1: Source of element 'b' (0 for x, 1 for y)
// Bit 2: Source of element 'c' (0 for x, 1 for y)
// Bit 3: Source of element 'd' (0 for x, 1 for y)
// Note that the least-significant bit is on the LEFT in this encoding.
const (
	_LLLL = iota // a:x, b:x, c:x, d:x
	_HLLL        // a:y, b:x, c:x, d:x
	_LHLL        // a:x, b:y, c:x, d:x
	_HHLL        // a:y, b:y, c:x, d:x
	_LLHL        // a:x, b:x, c:y, d:x
	_HLHL        // a:y, b:x, c:y, d:x
	_LHHL        // a:x, b:y, c:y, d:x
	_HHHL        // a:y, b:y, c:y, d:x
	_LLLH        // a:x, b:x, c:x, d:y
	_HLLH        // a:y, b:x, c:x, d:y
	_LHLH        // a:x, b:y, c:x, d:y
	_HHLH        // a:y, b:y, c:x, d:y
	_LLHH        // a:x, b:x, c:y, d:y
	_HLHH        // a:y, b:x, c:y, d:y
	_LHHH        // a:x, b:y, c:y, d:y
	_HHHH        // a:y, b:y, c:y, d:y
)

// cscimm4 converts the 4 vector element indices into a single
// uint8 for use as an immediate.
func cscimm4(a, b, c, d uint8) uint8 {
	return uint8(a + b<<2 + c<<4 + d<<6)
}

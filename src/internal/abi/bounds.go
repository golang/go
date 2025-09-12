// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

// This type and constants are for encoding different
// kinds of bounds check failures.
type BoundsErrorCode uint8

const (
	BoundsIndex      BoundsErrorCode = iota // s[x], 0 <= x < len(s) failed
	BoundsSliceAlen                         // s[?:x], 0 <= x <= len(s) failed
	BoundsSliceAcap                         // s[?:x], 0 <= x <= cap(s) failed
	BoundsSliceB                            // s[x:y], 0 <= x <= y failed (but boundsSliceA didn't happen)
	BoundsSlice3Alen                        // s[?:?:x], 0 <= x <= len(s) failed
	BoundsSlice3Acap                        // s[?:?:x], 0 <= x <= cap(s) failed
	BoundsSlice3B                           // s[?:x:y], 0 <= x <= y failed (but boundsSlice3A didn't happen)
	BoundsSlice3C                           // s[x:y:?], 0 <= x <= y failed (but boundsSlice3A/B didn't happen)
	BoundsConvert                           // (*[x]T)(s), 0 <= x <= len(s) failed
	numBoundsCodes
)

const (
	BoundsMaxReg   = 15
	BoundsMaxConst = 31
)

// Here's how we encode PCDATA_PanicBounds entries:

// We allow 16 registers (0-15) and 32 constants (0-31).
// Encode the following constant c:
//     bits    use
// -----------------------------
//       0     x is in a register
//       1     y is in a register
//
// if x is in a register
//       2     x is signed
//     [3:6]   x's register number
// else
//     [2:6]   x's constant value
//
// if y is in a register
//     [7:10]  y's register number
// else
//     [7:11]  y's constant value
//
// The final integer is c * numBoundsCode + code

// TODO: 32-bit

// Encode bounds failure information into an integer for PCDATA_PanicBounds.
// Register numbers must be in 0-15. Constants must be in 0-31.
func BoundsEncode(code BoundsErrorCode, signed, xIsReg, yIsReg bool, xVal, yVal int) int {
	c := int(0)
	if xIsReg {
		c |= 1 << 0
		if signed {
			c |= 1 << 2
		}
		if xVal < 0 || xVal > BoundsMaxReg {
			panic("bad xReg")
		}
		c |= xVal << 3
	} else {
		if xVal < 0 || xVal > BoundsMaxConst {
			panic("bad xConst")
		}
		c |= xVal << 2
	}
	if yIsReg {
		c |= 1 << 1
		if yVal < 0 || yVal > BoundsMaxReg {
			panic("bad yReg")
		}
		c |= yVal << 7
	} else {
		if yVal < 0 || yVal > BoundsMaxConst {
			panic("bad yConst")
		}
		c |= yVal << 7
	}
	return c*int(numBoundsCodes) + int(code)
}
func BoundsDecode(v int) (code BoundsErrorCode, signed, xIsReg, yIsReg bool, xVal, yVal int) {
	code = BoundsErrorCode(v % int(numBoundsCodes))
	c := v / int(numBoundsCodes)
	xIsReg = c&1 != 0
	c >>= 1
	yIsReg = c&1 != 0
	c >>= 1
	if xIsReg {
		signed = c&1 != 0
		c >>= 1
		xVal = c & 0xf
		c >>= 4
	} else {
		xVal = c & 0x1f
		c >>= 5
	}
	if yIsReg {
		yVal = c & 0xf
		c >>= 4
	} else {
		yVal = c & 0x1f
		c >>= 5
	}
	if c != 0 {
		panic("BoundsDecode decoding error")
	}
	return
}

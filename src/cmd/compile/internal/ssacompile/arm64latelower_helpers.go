// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

// isARM64addcon reports whether x can be encoded as the immediate value in an ADD or SUB instruction.
func isARM64addcon(v int64) bool {
	/* uimm12 or uimm24? */
	if v < 0 {
		return false
	}
	if (v & 0xFFF) == 0 {
		v >>= 12
	}
	return v <= 0xFFF
}

// isARM64bitcon reports whether a constant can be encoded into a logical instruction.
func isARM64bitcon(x uint64) bool {
	if x == 1<<64-1 || x == 0 {
		return false
	}
	// determine the period and sign-extend a unit to 64 bits
	switch {
	case x != x>>32|x<<32:
		// period is 64
		// nothing to do
	case x != x>>16|x<<48:
		// period is 32
		x = uint64(int64(int32(x)))
	case x != x>>8|x<<56:
		// period is 16
		x = uint64(int64(int16(x)))
	case x != x>>4|x<<60:
		// period is 8
		x = uint64(int64(int8(x)))
	default:
		// period is 4 or 2, always true
		// 0001, 0010, 0100, 1000 -- 0001 rotate
		// 0011, 0110, 1100, 1001 -- 0011 rotate
		// 0111, 1011, 1101, 1110 -- 0111 rotate
		// 0101, 1010             -- 01   rotate, repeat
		return true
	}
	return sequenceOfOnes(x) || sequenceOfOnes(^x)
}

// sequenceOfOnes tests whether a constant is a sequence of ones in binary, with leading and trailing zeros.
func sequenceOfOnes(x uint64) bool {
	y := x & -x // lowest set bit of x. x is good iff x+y is a power of 2
	y += x
	return (y-1)&y == 0
}

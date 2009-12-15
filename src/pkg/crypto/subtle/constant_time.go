// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements functions that are often useful in cryptographic
// code but require careful thought to use correctly.
package subtle

// ConstantTimeCompare returns 1 iff the two equal length slices, x
// and y, have equal contents. The time taken is a function of the length of
// the slices and is independent of the contents.
func ConstantTimeCompare(x, y []byte) int {
	var v byte

	for i := 0; i < len(x); i++ {
		v |= x[i] ^ y[i]
	}

	return ConstantTimeByteEq(v, 0)
}

// ConstantTimeSelect returns x if v is 1 and y if v is 0.
// Its behavior is undefined if v takes any other value.
func ConstantTimeSelect(v, x, y int) int { return ^(v-1)&x | (v-1)&y }

// ConstantTimeByteEq returns 1 if x == y and 0 otherwise.
func ConstantTimeByteEq(x, y uint8) int {
	z := ^(x ^ y)
	z &= z >> 4
	z &= z >> 2
	z &= z >> 1

	return int(z)
}

// ConstantTimeEq returns 1 if x == y and 0 otherwise.
func ConstantTimeEq(x, y int32) int {
	z := ^(x ^ y)
	z &= z >> 16
	z &= z >> 8
	z &= z >> 4
	z &= z >> 2
	z &= z >> 1

	return int(z & 1)
}

// ConstantTimeCopy copies the contents of y into x iff v == 1. If v == 0, x is left unchanged.
// Its behavior is undefined if v takes any other value.
func ConstantTimeCopy(v int, x, y []byte) {
	xmask := byte(v - 1)
	ymask := byte(^(v - 1))
	for i := 0; i < len(x); i++ {
		x[i] = x[i]&xmask | y[i]&ymask
	}
	return
}

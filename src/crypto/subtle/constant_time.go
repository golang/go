// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package subtle implements functions that are often useful in cryptographic
// code but require careful thought to use correctly.
package subtle

import "crypto/internal/fips140/subtle"

// ConstantTimeCompare returns 1 if the two slices, x and y, have equal contents
// and 0 otherwise. The time taken is a function of the length of the slices and
// is independent of the contents. If the lengths of x and y do not match it
// returns 0 immediately.
func ConstantTimeCompare(x, y []byte) int {
	if len(x) != len(y) {
		return 0
	}

	var v byte

	for i := 0; i < len(x); i++ {
		v |= x[i] ^ y[i]
	}

	return ConstantTimeByteEq(v, 0)
}

// ConstantTimeSelect returns x if v == 1 and y if v == 0.
// Its behavior is undefined if v takes any other value.
func ConstantTimeSelect(v, x, y int) int {
	// This is intrinsicified on arches with CMOV.
	// It implements the following superset behavior:
	// ConstantTimeSelect returns x if v != 0 and y if v == 0.
	// Do the same here to avoid non portable UB.
	v = int(constantTimeBoolToUint8(v != 0))
	return ^(v-1)&x | (v-1)&y
}

// ConstantTimeByteEq returns 1 if x == y and 0 otherwise.
func ConstantTimeByteEq(x, y uint8) int {
	return int(constantTimeBoolToUint8(x == y))
}

// ConstantTimeEq returns 1 if x == y and 0 otherwise.
func ConstantTimeEq(x, y int32) int {
	return int(constantTimeBoolToUint8(x == y))
}

// ConstantTimeCopy copies the contents of y into x (a slice of equal length)
// if v == 1. If v == 0, x is left unchanged. Its behavior is undefined if v
// takes any other value.
func ConstantTimeCopy(v int, x, y []byte) {
	// Forward this one since it gains nothing from compiler intrinsics.
	subtle.ConstantTimeCopy(v, x, y)
}

// ConstantTimeLessOrEq returns 1 if x <= y and 0 otherwise.
// Its behavior is undefined if x or y are negative or > 2**31 - 1.
func ConstantTimeLessOrEq(x, y int) int {
	return int(constantTimeBoolToUint8(x <= y))
}

// constantTimeBoolToUint8 is a compiler intrinsic.
// It returns 1 for true and 0 for false.
func constantTimeBoolToUint8(b bool) uint8 {
	panic("unreachable; must be intrinsicified")
}

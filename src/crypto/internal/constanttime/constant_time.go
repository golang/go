// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package constanttime

// The functions in this package are compiler intrinsics for constant-time
// operations. They are exposed by crypto/subtle and used directly by the
// FIPS 140-3 module.

// Select returns x if v == 1 and y if v == 0.
// Its behavior is undefined if v takes any other value.
func Select(v, x, y int) int {
	// This is intrinsicified on arches with CMOV.
	// It implements the following superset behavior:
	// ConstantTimeSelect returns x if v != 0 and y if v == 0.
	// Do the same here to avoid non portable UB.
	v = int(boolToUint8(v != 0))
	return ^(v-1)&x | (v-1)&y
}

// ByteEq returns 1 if x == y and 0 otherwise.
func ByteEq(x, y uint8) int {
	return int(boolToUint8(x == y))
}

// Eq returns 1 if x == y and 0 otherwise.
func Eq(x, y int32) int {
	return int(boolToUint8(x == y))
}

// LessOrEq returns 1 if x <= y and 0 otherwise.
// Its behavior is undefined if x or y are negative or > 2**31 - 1.
func LessOrEq(x, y int) int {
	return int(boolToUint8(x <= y))
}

// boolToUint8 is a compiler intrinsic.
// It returns 1 for true and 0 for false.
func boolToUint8(b bool) uint8 {
	panic("unreachable; must be intrinsicified")
}

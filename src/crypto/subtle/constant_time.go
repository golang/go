// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package subtle implements functions that are often useful in cryptographic
// code but require careful thought to use correctly.
package subtle

import (
	"crypto/internal/constanttime"
	"crypto/internal/fips140/subtle"
)

// These functions are forwarded to crypto/internal/constanttime for intrinsified
// operations, and to crypto/internal/fips140/subtle for byte slice operations.

// ConstantTimeCompare returns 1 if the two slices, x and y, have equal contents
// and 0 otherwise. The time taken is a function of the length of the slices and
// is independent of the contents. If the lengths of x and y do not match it
// returns 0 immediately.
func ConstantTimeCompare(x, y []byte) int {
	return subtle.ConstantTimeCompare(x, y)
}

// ConstantTimeSelect returns x if v == 1 and y if v == 0.
// Its behavior is undefined if v takes any other value.
func ConstantTimeSelect(v, x, y int) int {
	return constanttime.Select(v, x, y)
}

// ConstantTimeByteEq returns 1 if x == y and 0 otherwise.
func ConstantTimeByteEq(x, y uint8) int {
	return constanttime.ByteEq(x, y)
}

// ConstantTimeEq returns 1 if x == y and 0 otherwise.
func ConstantTimeEq(x, y int32) int {
	return constanttime.Eq(x, y)
}

// ConstantTimeCopy copies the contents of y into x (a slice of equal length)
// if v == 1. If v == 0, x is left unchanged. Its behavior is undefined if v
// takes any other value.
func ConstantTimeCopy(v int, x, y []byte) {
	subtle.ConstantTimeCopy(v, x, y)
}

// ConstantTimeLessOrEq returns 1 if x <= y and 0 otherwise.
// Its behavior is undefined if x or y are negative or > 2**31 - 1.
func ConstantTimeLessOrEq(x, y int) int {
	return constanttime.LessOrEq(x, y)
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package subtle

import (
	"crypto/internal/constanttime"
	"crypto/internal/fips140deps/byteorder"
	"math/bits"
)

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

	return constanttime.ByteEq(v, 0)
}

// ConstantTimeLessOrEqBytes returns 1 if x <= y and 0 otherwise. The comparison
// is lexigraphical, or big-endian. The time taken is a function of the length of
// the slices and is independent of the contents. If the lengths of x and y do not
// match it returns 0 immediately.
func ConstantTimeLessOrEqBytes(x, y []byte) int {
	if len(x) != len(y) {
		return 0
	}

	// Do a constant time subtraction chain y - x.
	// If there is no borrow at the end, then x <= y.
	var b uint64
	for len(x) > 8 {
		x0 := byteorder.BEUint64(x[len(x)-8:])
		y0 := byteorder.BEUint64(y[len(y)-8:])
		_, b = bits.Sub64(y0, x0, b)
		x = x[:len(x)-8]
		y = y[:len(y)-8]
	}
	if len(x) > 0 {
		xb := make([]byte, 8)
		yb := make([]byte, 8)
		copy(xb[8-len(x):], x)
		copy(yb[8-len(y):], y)
		x0 := byteorder.BEUint64(xb)
		y0 := byteorder.BEUint64(yb)
		_, b = bits.Sub64(y0, x0, b)
	}
	return int(b ^ 1)
}

// ConstantTimeCopy copies the contents of y into x (a slice of equal length)
// if v == 1. If v == 0, x is left unchanged. Its behavior is undefined if v
// takes any other value.
func ConstantTimeCopy(v int, x, y []byte) {
	if len(x) != len(y) {
		panic("subtle: slices have different lengths")
	}

	xmask := byte(v - 1)
	ymask := byte(^(v - 1))
	for i := 0; i < len(x); i++ {
		x[i] = x[i]&xmask | y[i]&ymask
	}
}

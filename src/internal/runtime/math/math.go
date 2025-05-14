// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "internal/goarch"

const (
	MaxUint32  = ^uint32(0)
	MaxUint64  = ^uint64(0)
	MaxUintptr = ^uintptr(0)

	MaxInt64 = int64(MaxUint64 >> 1)
)

// MulUintptr returns a * b and whether the multiplication overflowed.
// On supported platforms this is an intrinsic lowered by the compiler.
func MulUintptr(a, b uintptr) (uintptr, bool) {
	if a|b < 1<<(4*goarch.PtrSize) || a == 0 {
		return a * b, false
	}
	overflow := b > MaxUintptr/a
	return a * b, overflow
}

// Mul64 returns the 128-bit product of x and y: (hi, lo) = x * y
// with the product bits' upper half returned in hi and the lower
// half returned in lo.
// This is a copy from math/bits.Mul64
// On supported platforms this is an intrinsic lowered by the compiler.
func Mul64(x, y uint64) (hi, lo uint64) {
	const mask32 = 1<<32 - 1
	x0 := x & mask32
	x1 := x >> 32
	y0 := y & mask32
	y1 := y >> 32
	w0 := x0 * y0
	t := x1*y0 + w0>>32
	w1 := t & mask32
	w2 := t >> 32
	w1 += x0 * y1
	hi = x1*y1 + w2 + w1>>32
	lo = x * y
	return
}

// Add64 returns the sum with carry of x, y and carry: sum = x + y + carry.
// The carry input must be 0 or 1; otherwise the behavior is undefined.
// The carryOut output is guaranteed to be 0 or 1.
//
// This function's execution time does not depend on the inputs.
// On supported platforms this is an intrinsic lowered by the compiler.
func Add64(x, y, carry uint64) (sum, carryOut uint64) {
	sum = x + y + carry
	// The sum will overflow if both top bits are set (x & y) or if one of them
	// is (x | y), and a carry from the lower place happened. If such a carry
	// happens, the top bit will be 1 + 0 + 1 = 0 (&^ sum).
	carryOut = ((x & y) | ((x | y) &^ sum)) >> 63
	return
}

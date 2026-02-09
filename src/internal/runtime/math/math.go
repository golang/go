// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "internal/goarch"

const (
	MaxUint16  = ^uint16(0)
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

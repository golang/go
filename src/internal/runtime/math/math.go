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

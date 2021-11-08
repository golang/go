// +build amd64
// errorcheck -0 -m

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that inlining of math/bits.RotateLeft* treats those calls as intrinsics.

package p

import "math/bits"

var (
	x8  uint8
	x16 uint16
	x32 uint32
	x64 uint64
	x   uint
)

func f() { // ERROR "can inline f"
	x8 = bits.RotateLeft8(x8, 1)
	x16 = bits.RotateLeft16(x16, 1)
	x32 = bits.RotateLeft32(x32, 1)
	x64 = bits.RotateLeft64(x64, 1)
	x = bits.RotateLeft(x, 1)
}

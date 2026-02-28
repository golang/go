// errorcheck -0 -d=ssa/prove/debug=2

//go:build amd64 || arm64

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f0i(x int) int {
	if x == 20 {
		return x // ERROR "Proved.+is constant 20$"
	}

	if (x + 20) == 20 {
		return x + 5 // ERROR "Proved.+is constant 0$" "Proved.+is constant 5$" "x\+d >=? w"
	}

	return x + 1
}

func f0u(x uint) int {
	if x == 20 {
		return int(x) // ERROR "Proved.+is constant 20$"
	}

	if (x + 20) == 20 {
		return int(x + 5) // ERROR "Proved.+is constant 0$" "Proved.+is constant 5$" "x\+d >=? w"
	}

	if x < 1000 {
		return int(x) >> 31 // ERROR "(Proved.+is constant 0|Proved Rsh[0-9]+x[0-9]+ is unsigned)$"
	}
	if x := int32(x); x < -1000 {
		return int(x >> 31) // ERROR "Proved.+is constant -1$"
	}

	return int(x) + 1
}

// Check that prove is zeroing these right shifts of positive ints by bit-width - 1.
// e.g (Rsh64x64 <t> n (Const64 <typ.UInt64> [63])) && ft.isNonNegative(n) -> 0
func sh64(n int64) int64 {
	if n < 0 {
		return n
	}
	return n >> 63 // ERROR "(Proved .+ is constant 0|Proved Rsh[0-9]+x[0-9]+ is unsigned)$"
}

func sh32(n int32) int32 {
	if n < 0 {
		return n
	}
	return n >> 31 // ERROR "(Proved .+ is constant 0|Proved Rsh[0-9]+x[0-9]+ is unsigned)$"
}

func sh32x64(n int32) int32 {
	if n < 0 {
		return n
	}
	return n >> uint64(31) // ERROR "(Proved .+ is constant 0|Proved Rsh[0-9]+x[0-9]+ is unsigned)$"
}

func sh32x64n(n int32) int32 {
	if n >= 0 {
		return 0
	}
	return n >> 31 // ERROR "Proved .+ is constant -1$"
}

func sh16(n int16) int16 {
	if n < 0 {
		return n
	}
	return n >> 15 // ERROR "(Proved .+ is constant 0|Proved Rsh[0-9]+x[0-9]+ is unsigned)$"
}

func sh64noopt(n int64) int64 {
	return n >> 63 // not optimized; n could be negative
}

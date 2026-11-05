// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

// The unbounded shift lowering emits CMPQconst [width] count feeding the
// count mask (SBBQcarrymask). A comparison of the same count against the
// same width shares those flags; SBBQ writes them, so the SETcc must not
// read them after the mask. Regression test for the missing clobberFlags.

//go:noinline
func lshEq(x, y uint64) uint64 {
	r := x << y
	eq := uint64(0)
	if y == 64 {
		eq = 1
	}
	return r + eq
}

//go:noinline
func lshNe(x, y uint64) uint64 {
	r := x << y
	ne := uint64(0)
	if y != 64 {
		ne = 1
	}
	return r + ne
}

//go:noinline
func rshU(x, y uint64) uint64 {
	r := x >> y
	eq := uint64(0)
	if y == 64 {
		eq = 1
	}
	return r + eq
}

//go:noinline
func rshS(x int64, y uint64) int64 {
	r := x >> y
	eq := int64(0)
	if y == 64 {
		eq = 1
	}
	return r + eq
}

//go:noinline
func lsh32(x uint32, y uint64) uint32 {
	r := x << y
	eq := uint32(0)
	if y == 32 {
		eq = 1
	}
	return r + eq
}

// 386: the decomposed 64-bit comparison shares CMPLconst [1] hi with the
// shift's count saturation (SBBLcarrymask), so SETEQ reads clobbered flags.
//
//go:noinline
func cmpSat(x int16) int16 {
	r := int16(-2047) << uint64(uint64(x))
	if uint64(x) <= 8589934591 { // 0x1FFFFFFFF; false for negative x
		r = int16(x) &^ 3
	}
	return r
}

// The count comparison may also use <= or >: each of ==, !=, <= and > against the
// width lowers to the same CMP const [width] count as the shift's bounds mask.

//go:noinline
func rshLe16(x, y int16) int16 {
	r := ^int16(uint16(y)) >> uint64(int16(x))
	b := int16(0)
	if uint64(int16(x)) <= 16 {
		b = 1
	}
	return r + b
}

//go:noinline
func rshGt64(x int64) int64 {
	r := int64(36028797018963969) >> uint64(int64(x))
	b := int64(0)
	if uint64(int64(x)) > 64 {
		b = 1
	}
	return r + b
}

func main() {
	if got := lshEq(1, 65); got != 0 {
		fmt.Printf("lshEq(1, 65) = %d, want 0\n", got)
		panic("FAIL")
	}
	if got := lshEq(1, 63); got != 9223372036854775808 {
		fmt.Printf("lshEq(1, 63) = %d, want 9223372036854775808\n", got)
		panic("FAIL")
	}
	if got := lshEq(1, 64); got != 1 {
		fmt.Printf("lshEq(1, 64) = %d, want 1\n", got)
		panic("FAIL")
	}
	if got := lshNe(1, 65); got != 1 {
		fmt.Printf("lshNe(1, 65) = %d, want 1\n", got)
		panic("FAIL")
	}
	if got := rshU(1, 65); got != 0 {
		fmt.Printf("rshU(1, 65) = %d, want 0\n", got)
		panic("FAIL")
	}
	if got := rshS(-1, 65); got != -1 {
		fmt.Printf("rshS(-1, 65) = %d, want -1\n", got)
		panic("FAIL")
	}
	if got := lsh32(1, 33); got != 0 {
		fmt.Printf("lsh32(1, 33) = %d, want 0\n", got)
		panic("FAIL")
	}
	if got := cmpSat(-1); got != 0 {
		fmt.Printf("cmpSat(-1) = %d, want 0\n", got)
		panic("FAIL")
	}
	if got := cmpSat(5); got != 4 {
		fmt.Printf("cmpSat(5) = %d, want 4\n", got)
		panic("FAIL")
	}
	if got := rshLe16(-32768, -32768); got != 0 {
		fmt.Printf("rshLe16(-32768, -32768) = %d, want 0\n", got)
		panic("FAIL")
	}
	if got := rshGt64(-1 << 63); got != 1 {
		fmt.Printf("rshGt64(-1<<63) = %d, want 1\n", got)
		panic("FAIL")
	}
}

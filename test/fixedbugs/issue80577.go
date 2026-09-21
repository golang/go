// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 80577: on riscv64, sign extensions were elided after
// 32-bit instructions that architecturally sign-extend their
// result, but an unsigned-typed value spilled across a call is
// restored with a zero-extending load, losing the elided
// sign extension.

package main

import "math/bits"

var sink uint32

//go:noinline
func use(x uint32) { sink = x }

//go:noinline
func mul(a, b uint32) int64 {
	p := a * b
	use(p) // p live across the call, forcing a spill
	return int64(int32(p))
}

//go:noinline
func div(a, b uint32) int64 {
	p := a / b
	use(p)
	return int64(int32(p))
}

//go:noinline
func rem(a, b uint32) int64 {
	p := a % b
	use(p)
	return int64(int32(p))
}

//go:noinline
func rot(a uint32, k int) int64 {
	p := bits.RotateLeft32(a, k)
	use(p)
	return int64(int32(p))
}

func main() {
	const want = -2147483648
	if got := mul(0x8000, 0x10000); got != want {
		println("mul: got", got, "want", want)
		panic("bad mul")
	}
	if got := div(0x80000000, 1); got != want {
		println("div: got", got, "want", want)
		panic("bad div")
	}
	if got := rem(0x80000000, 0xffffffff); got != want {
		println("rem: got", got, "want", want)
		panic("bad rem")
	}
	if got := rot(1, 31); got != want {
		println("rot: got", got, "want", want)
		panic("bad rot")
	}
}

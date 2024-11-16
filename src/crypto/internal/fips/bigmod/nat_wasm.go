// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package bigmod

import "unsafe"

// The generic implementation relies on 64x64->128 bit multiplication and
// 64-bit add-with-carry, which are compiler intrinsics on many architectures.
// Wasm doesn't support those. Here we implement it with 32x32->64 bit
// operations, which is more efficient on Wasm.

func idx(x *uint, i uintptr) *uint {
	return (*uint)(unsafe.Pointer(uintptr(unsafe.Pointer(x)) + i*8))
}

func addMulVVWWasm(z, x *uint, y uint, n uintptr) (carry uint) {
	const mask32 = 1<<32 - 1
	y0 := y & mask32
	y1 := y >> 32
	for i := range n {
		xi := *idx(x, i)
		x0 := xi & mask32
		x1 := xi >> 32
		zi := *idx(z, i)
		z0 := zi & mask32
		z1 := zi >> 32
		c0 := carry & mask32
		c1 := carry >> 32

		w00 := x0*y0 + z0 + c0
		l00 := w00 & mask32
		h00 := w00 >> 32

		w01 := x0*y1 + z1 + h00
		l01 := w01 & mask32
		h01 := w01 >> 32

		w10 := x1*y0 + c1 + l01
		h10 := w10 >> 32

		carry = x1*y1 + h10 + h01
		*idx(z, i) = w10<<32 + l00
	}
	return carry
}

func addMulVVW1024(z, x *uint, y uint) (c uint) {
	return addMulVVWWasm(z, x, y, 1024/_W)
}

func addMulVVW1536(z, x *uint, y uint) (c uint) {
	return addMulVVWWasm(z, x, y, 1536/_W)
}

func addMulVVW2048(z, x *uint, y uint) (c uint) {
	return addMulVVWWasm(z, x, y, 2048/_W)
}

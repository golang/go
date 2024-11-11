// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bigmod

// The generic implementation relies on 64x64->128 bit multiplication and
// 64-bit add-with-carry, which are compiler intrinsics on many architectures.
// Wasm doesn't support those. Here we implement it with 32x32->64 bit
// operations, which is more efficient on Wasm.

// addMulVVW multiplies the multi-word value x by the single-word value y,
// adding the result to the multi-word value z and returning the final carry.
// It can be thought of as one row of a pen-and-paper column multiplication.
func addMulVVW(z, x []uint, y uint) (carry uint) {
	const mask32 = 1<<32 - 1
	y0 := y & mask32
	y1 := y >> 32
	_ = x[len(z)-1] // bounds check elimination hint
	for i, zi := range z {
		xi := x[i]
		x0 := xi & mask32
		x1 := xi >> 32
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
		z[i] = w10<<32 + l00
	}
	return carry
}

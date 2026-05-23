// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140deps/byteorder"
)

// GHASH is exposed to allow crypto/cipher to implement non-AES GCM modes.
// It is not allowed as a stand-alone operation in FIPS mode because it
// is not ACVP tested.
func GHASH(key *[16]byte, inputs ...[]byte) []byte {
	fips140.RecordNonApproved()
	var out [gcmBlockSize]byte
	ghash(&out, key, inputs...)
	return out[:]
}

// ghashMul does constant-time carry-less multiplication of two 32-bit integers,
// returning the 64-bit product.
func ghashMul(x, y uint32) uint64 {
	// This function implements carryless multiplication using a technique first
	// described by Thomas Pornin in the BearSSL documentation [0]. This
	// technique uses generic integer multiplication, but ignores the carrys by
	// masking all but 8 bits of the inputs, creating three bit holes between
	// each unmasked bit. If the multiplications of any of the unmasked bits
	// then cause a carry, the resulting carry bit spills into one of the three
	// bit holes.
	//
	// Each 32-bit input is split into four 32-bit masked values, each
	// containing 8 unmasked bits. The mask is shifted by one bit for each of
	// the four values, such that the four values cover the full 32 bits of the
	// input.
	//
	// In order to compute the bits at position z_k, z_k+4, z_k+8, ..., z_k+60
	// for k = 0, 1, 2, 3, we compute the sum of the products x_i*y_j for all i,
	// j such that i+j = k mod 4.
	//
	// We then mask the sum of each of the four products with the same mask used
	// for the input values, which zeros out any spilled carry bits, and OR the
	// masked values to get the final product.
	//
	// [0] https://www.bearssl.org/constanttime.html#ghash-for-gcm

	var xm, ym [4]uint32
	var z [4]uint64

	for i := range 4 {
		// Mask off the three bit holes in each input, creating four masked
		// values for each input.
		xm[i] = x & (0x11111111 << i)
		ym[i] = y & (0x11111111 << i)
	}

	for i := range 4 {
		// Compute the multiplication of x by the circulant matrix of y, using
		// XOR to get carryless addition of the products:
		//
		//  | z[0] |   | ym[0] ym[3] ym[2] ym[1] |   | xm[0] |
		//  | z[1] | = | ym[1] ym[0] ym[3] ym[2] | x | xm[1] |
		//  | z[2] |   | ym[2] ym[1] ym[0] ym[3] |   | xm[2] |
		//  | z[3] |   | ym[3] ym[2] ym[1] ym[0] |   | xm[3] |
		z[i] = (uint64(xm[0]) * uint64(ym[i])) ^ (uint64(xm[1]) * uint64(ym[(i+3)%4])) ^ (uint64(xm[2]) * uint64(ym[(i+2)%4])) ^ (uint64(xm[3]) * uint64(ym[(i+1)%4]))
		z[i] &= 0x1111111111111111 << i
	}

	return z[0] | z[1] | z[2] | z[3]
}

func ghash(out, H *[gcmBlockSize]byte, inputs ...[]byte) {
	// The GHASH algorithm computes the sum of the products of two 128 bit
	// integers Y and H (the input block and the key, respectively) in the field
	// GF(2^128), modulo the field polynomial.
	//
	// We use the Karatsuba algorithm to decompose the 128-bit multiplication
	// into three 64-bit multiplications, which we further decompose into 9
	// 32-bit multiplications with 64-bit products.

	// Make sure out is zeroed before we use it.
	clear(out[:])

	var y, h [4]uint32
	for i := range 4 {
		h[3-i] = byteorder.BEUint32(H[i*4 : (i*4)+4])
	}

	blockIterator := func(yield func([]byte) bool) {
		for _, input := range inputs {
			for len(input) >= 16 {
				if !yield(input[:16]) {
					return
				}
				input = input[16:]
			}
			if len(input) > 0 {
				var partialBlock [gcmBlockSize]byte
				copy(partialBlock[:], input)
				if !yield(partialBlock[:]) {
					return
				}
			}
		}
	}

	// Compute the GHASH of the inputs by iterating over 16-byte blocks of the
	// inputs, XORing each block into the current state, and multiplying the
	// result by the key.
	for block := range blockIterator {
		for i := range 4 {
			y[3-i] ^= byteorder.BEUint32(block[i*4 : (i*4)+4])
		}

		// Split y*h into nine products:
		//
		//  zLo = y0*h0, y2*h2, (y0^y2) * (h0^h2)
		//  zHi = y1*h1, y3*h3, (y1^y3) * (h1^h3)
		//  zSum = (y0^y1) * (h0^h1), (y2^y3) * (h2^h3), ((y0^y2) ^ (y1^y3)) * ((h0^h2) ^ (h1^h3))
		var zLo, zHi, zSum [3]uint64

		zLo[0] = ghashMul(y[0], h[0])
		zHi[0] = ghashMul(y[1], h[1])
		zSum[0] = ghashMul(y[0]^y[1], h[0]^h[1])

		zLo[1] = ghashMul(y[2], h[2])
		zHi[1] = ghashMul(y[3], h[3])
		zSum[1] = ghashMul(y[2]^y[3], h[2]^h[3])

		zLo[2] = ghashMul(y[0]^y[2], h[0]^h[2])
		zHi[2] = ghashMul(y[1]^y[3], h[1]^h[3])
		zSum[2] = ghashMul((y[0]^y[2])^(y[1]^y[3]), (h[0]^h[2])^(h[1]^h[3]))

		// Reconstruct the 128-bit terms zLo, zHi, and zSum from their constituent 64-bit products
		var result [3][2]uint64
		for i := range 3 {
			mid := zSum[i] ^ zLo[i] ^ zHi[i]
			// Add the lower 32 bits of the middle term to the low term
			result[i][0] = zLo[i] ^ (mid << 32)
			// Add the upper 32 bits of the middle term to the high term
			result[i][1] = zHi[i] ^ (mid >> 32)
		}

		// Compute the middle term by adding the high and low terms to the sum term
		result[2][0] ^= result[0][0] ^ result[1][0]
		result[2][1] ^= result[0][1] ^ result[1][1]

		// Add the lower bits of the middle term to the higher bits of the low term
		result[0][1] ^= result[2][0]
		// Add the higher bits of the middle term to the lower bits of the high term
		result[1][0] ^= result[2][1]

		// Reconstruct the 256-bit product from the low and high terms, shifted
		// by one bit to satisfy the GHASH construction.
		var z [4]uint64
		z[0] = result[0][0] << 1
		z[1] = (result[0][1] << 1) | (result[0][0] >> 63)
		z[2] = (result[1][0] << 1) | (result[0][1] >> 63)
		z[3] = (result[1][1] << 1) | (result[1][0] >> 63)

		// Reduce the 256-bit product modulo the field polynomial. z0 and z1 contain
		// the high-degree terms (255 to 128), and z2 and z3 contain the low-degree terms (127 to 0).
		for i := range 2 {
			lw := z[i]
			// Add the remainders of the high-degree terms to the low-degree terms
			z[i+2] ^= lw ^ (lw >> 1) ^ (lw >> 2) ^ (lw >> 7)
			// Add the carrys from the reduction
			z[i+1] ^= (lw << 63) ^ (lw << 62) ^ (lw << 57)
		}

		// Write the reduced 128-bit product back into y
		y[0], y[1], y[2], y[3] = uint32(z[2]), uint32(z[2]>>32), uint32(z[3]), uint32(z[3]>>32)
	}

	for i := range 4 {
		byteorder.BEPutUint32(out[i*4:(i*4)+4], y[3-i])
	}
}

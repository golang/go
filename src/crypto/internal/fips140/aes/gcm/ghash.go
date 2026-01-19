// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140deps/byteorder"
)

// gcmFieldElement represents a value in GF(2¹²⁸). In order to reflect the GCM
// standard and make binary.BigEndian suitable for marshaling these values, the
// bits are stored in big endian order. For example:
//
//	the coefficient of x⁰ can be obtained by v.low >> 63.
//	the coefficient of x⁶³ can be obtained by v.low & 1.
//	the coefficient of x⁶⁴ can be obtained by v.high >> 63.
//	the coefficient of x¹²⁷ can be obtained by v.high & 1.
type gcmFieldElement struct {
	low, high uint64
}

// GHASH is exposed to allow crypto/cipher to implement non-AES GCM modes.
// It is not allowed as a stand-alone operation in FIPS mode because it
// is not ACVP tested.
func GHASH(key *[16]byte, inputs ...[]byte) []byte {
	fips140.RecordNonApproved()
	var out [gcmBlockSize]byte
	ghash(&out, key, inputs...)
	return out[:]
}

// ghash is a variable-time generic implementation of GHASH, which shouldn't
// be used on any architecture with hardware support for AES-GCM.
//
// Each input is zero-padded to 128-bit before being absorbed.
func ghash(out, H *[gcmBlockSize]byte, inputs ...[]byte) {
	// productTable contains the first sixteen powers of the key, H.
	// However, they are in bit reversed order.
	var productTable [16]gcmFieldElement

	// We precompute 16 multiples of H. However, when we do lookups
	// into this table we'll be using bits from a field element and
	// therefore the bits will be in the reverse order. So normally one
	// would expect, say, 4*H to be in index 4 of the table but due to
	// this bit ordering it will actually be in index 0010 (base 2) = 2.
	x := gcmFieldElement{
		byteorder.BEUint64(H[:8]),
		byteorder.BEUint64(H[8:]),
	}
	productTable[reverseBits(1)] = x

	for i := 2; i < 16; i += 2 {
		productTable[reverseBits(i)] = ghashDouble(&productTable[reverseBits(i/2)])
		productTable[reverseBits(i+1)] = ghashAdd(&productTable[reverseBits(i)], &x)
	}

	var y gcmFieldElement
	for _, input := range inputs {
		ghashUpdate(&productTable, &y, input)
	}

	byteorder.BEPutUint64(out[:], y.low)
	byteorder.BEPutUint64(out[8:], y.high)
}

// reverseBits reverses the order of the bits of 4-bit number in i.
func reverseBits(i int) int {
	i = ((i << 2) & 0xc) | ((i >> 2) & 0x3)
	i = ((i << 1) & 0xa) | ((i >> 1) & 0x5)
	return i
}

// ghashAdd adds two elements of GF(2¹²⁸) and returns the sum.
func ghashAdd(x, y *gcmFieldElement) gcmFieldElement {
	// Addition in a characteristic 2 field is just XOR.
	return gcmFieldElement{x.low ^ y.low, x.high ^ y.high}
}

// ghashDouble returns the result of doubling an element of GF(2¹²⁸).
func ghashDouble(x *gcmFieldElement) (double gcmFieldElement) {
	msbSet := x.high&1 == 1

	// Because of the bit-ordering, doubling is actually a right shift.
	double.high = x.high >> 1
	double.high |= x.low << 63
	double.low = x.low >> 1

	// If the most-significant bit was set before shifting then it,
	// conceptually, becomes a term of x^128. This is greater than the
	// irreducible polynomial so the result has to be reduced. The
	// irreducible polynomial is 1+x+x^2+x^7+x^128. We can subtract that to
	// eliminate the term at x^128 which also means subtracting the other
	// four terms. In characteristic 2 fields, subtraction == addition ==
	// XOR.
	if msbSet {
		double.low ^= 0xe100000000000000
	}

	return
}

var ghashReductionTable = []uint16{
	0x0000, 0x1c20, 0x3840, 0x2460, 0x7080, 0x6ca0, 0x48c0, 0x54e0,
	0xe100, 0xfd20, 0xd940, 0xc560, 0x9180, 0x8da0, 0xa9c0, 0xb5e0,
}

// ghashMul sets y to y*H, where H is the GCM key, fixed during New.
func ghashMul(productTable *[16]gcmFieldElement, y *gcmFieldElement) {
	var z gcmFieldElement

	for i := 0; i < 2; i++ {
		word := y.high
		if i == 1 {
			word = y.low
		}

		// Multiplication works by multiplying z by 16 and adding in
		// one of the precomputed multiples of H.
		for j := 0; j < 64; j += 4 {
			msw := z.high & 0xf
			z.high >>= 4
			z.high |= z.low << 60
			z.low >>= 4
			z.low ^= uint64(ghashReductionTable[msw]) << 48

			// the values in |table| are ordered for little-endian bit
			// positions. See the comment in New.
			t := productTable[word&0xf]

			z.low ^= t.low
			z.high ^= t.high
			word >>= 4
		}
	}

	*y = z
}

// updateBlocks extends y with more polynomial terms from blocks, based on
// Horner's rule. There must be a multiple of gcmBlockSize bytes in blocks.
func updateBlocks(productTable *[16]gcmFieldElement, y *gcmFieldElement, blocks []byte) {
	for len(blocks) > 0 {
		y.low ^= byteorder.BEUint64(blocks)
		y.high ^= byteorder.BEUint64(blocks[8:])
		ghashMul(productTable, y)
		blocks = blocks[gcmBlockSize:]
	}
}

// ghashUpdate extends y with more polynomial terms from data. If data is not a
// multiple of gcmBlockSize bytes long then the remainder is zero padded.
func ghashUpdate(productTable *[16]gcmFieldElement, y *gcmFieldElement, data []byte) {
	fullBlocks := (len(data) >> 4) << 4
	updateBlocks(productTable, y, data[:fullBlocks])

	if len(data) != fullBlocks {
		var partialBlock [gcmBlockSize]byte
		copy(partialBlock[:], data[fullBlocks:])
		updateBlocks(productTable, y, partialBlock[:])
	}
}

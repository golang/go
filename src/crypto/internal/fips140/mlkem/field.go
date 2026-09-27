// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mlkem

import (
	"crypto/internal/constanttime"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140deps/byteorder"
	"errors"
)

// fieldElement is an integer modulo q, an element of ℤ_q. It is always reduced.
type fieldElement uint16

// fieldCheckReduced checks that a value a is < q.
func fieldCheckReduced(a uint16) (fieldElement, error) {
	if a >= q {
		return 0, errors.New("unreduced field element")
	}
	return fieldElement(a), nil
}

// fieldReduceOnce reduces a value a < 2q.
func fieldReduceOnce(a uint16) fieldElement {
	// a and q-1 are non-negative and fit int32, as LessOrEq requires.
	v := constanttime.LessOrEq(int(a), q-1)
	return fieldElement(constanttime.Select(v, int(a), int(a)-q))
}

func fieldAdd(a, b fieldElement) fieldElement {
	x := uint16(a + b)
	return fieldReduceOnce(x)
}

func fieldSub(a, b fieldElement) fieldElement {
	x := uint16(a - b + q)
	return fieldReduceOnce(x)
}

const (
	barrettMultiplier = 5039 // 2¹² * 2¹² / q
	barrettShift      = 24   // log₂(2¹² * 2¹²)
)

// fieldReduce reduces a value a < 2q² using Barrett reduction, to avoid
// potentially variable-time division.
func fieldReduce(a uint32) fieldElement {
	quotient := uint32((uint64(a) * barrettMultiplier) >> barrettShift)
	return fieldReduceOnce(uint16(a - quotient*q))
}

func fieldMul(a, b fieldElement) fieldElement {
	x := uint32(a) * uint32(b)
	return fieldReduce(x)
}

// fieldMulSub returns a * (b - c). This operation is fused to save a
// fieldReduceOnce after the subtraction.
func fieldMulSub(a, b, c fieldElement) fieldElement {
	x := uint32(a) * uint32(b-c+q)
	return fieldReduce(x)
}

// fieldAddMul returns a * b + c * d. This operation is fused to save a
// fieldReduceOnce and a fieldReduce.
func fieldAddMul(a, b, c, d fieldElement) fieldElement {
	x := uint32(a) * uint32(b)
	x += uint32(c) * uint32(d)
	return fieldReduce(x)
}

// compress maps a field element uniformly to the range 0 to 2ᵈ-1, according to
// FIPS 203, Definition 4.7.
func compress(x fieldElement, d uint8) uint16 {
	// We want to compute (x * 2ᵈ) / q, rounded to nearest integer, with 1/2
	// rounding up (see FIPS 203, Section 2.3).

	// Barrett reduction produces a quotient and a remainder in the range [0, 2q),
	// such that dividend = quotient * q + remainder.
	dividend := uint32(x) << d // x * 2ᵈ
	quotient := uint32(uint64(dividend) * barrettMultiplier >> barrettShift)
	remainder := dividend - quotient*q

	// Since the remainder is in the range [0, 2q), not [0, q), we need to
	// portion it into three spans for rounding.
	//
	//     [ 0,       q/2     ) -> round to 0
	//     [ q/2,     q + q/2 ) -> round to 1
	//     [ q + q/2, 2q      ) -> round to 2
	//
	// We can convert that to the following logic: add 1 if remainder > q/2,
	// then add 1 again if remainder > q + q/2.
	//
	// Note that if remainder > x, then ⌊x⌋ - remainder underflows, and the top
	// bit of the difference will be set.
	quotient += (q/2 - remainder) >> 31 & 1
	quotient += (q + q/2 - remainder) >> 31 & 1

	// quotient might have overflowed at this point, so reduce it by masking.
	var mask uint32 = (1 << d) - 1
	return uint16(quotient & mask)
}

// decompress maps a number x between 0 and 2ᵈ-1 uniformly to the full range of
// field elements, according to FIPS 203, Definition 4.8.
func decompress(y uint16, d uint8) fieldElement {
	// We want to compute (y * q) / 2ᵈ, rounded to nearest integer, with 1/2
	// rounding up (see FIPS 203, Section 2.3).

	dividend := uint32(y) * q
	quotient := dividend >> d // (y * q) / 2ᵈ

	// The d'th least-significant bit of the dividend (the most significant bit
	// of the remainder) is 1 for the top half of the values that divide to the
	// same quotient, which are the ones that round up.
	quotient += dividend >> (d - 1) & 1

	// quotient is at most (2¹¹-1) * q / 2¹¹ + 1 = 3328, so it didn't overflow.
	return fieldElement(quotient)
}

// ringElement is a polynomial, an element of R_q, represented as an array
// according to FIPS 203, Section 2.4.4.
type ringElement [n]fieldElement

// polyAdd adds two ringElements or nttElements.
func polyAdd[T ~[n]fieldElement](a, b T) (s T) {
	for i := range s {
		s[i] = fieldAdd(a[i], b[i])
	}
	return s
}

// polySub subtracts two ringElements or nttElements.
func polySub[T ~[n]fieldElement](a, b T) (s T) {
	for i := range s {
		s[i] = fieldSub(a[i], b[i])
	}
	return s
}

// polyByteEncode appends the 384-byte encoding of f to b.
//
// It implements ByteEncode₁₂, according to FIPS 203, Algorithm 5.
func polyByteEncode[T ~[n]fieldElement](b []byte, f T) []byte {
	out, B := sliceForAppend(b, encodingSize12)
	for i := 0; i < n; i += 2 {
		x := uint32(f[i]) | uint32(f[i+1])<<12
		B[0] = uint8(x)
		B[1] = uint8(x >> 8)
		B[2] = uint8(x >> 16)
		B = B[3:]
	}
	return out
}

// polyByteDecode decodes the 384-byte encoding of a polynomial, checking that
// all the coefficients are properly reduced. This fulfills the "Modulus check"
// step of ML-KEM Encapsulation.
//
// It implements ByteDecode₁₂, according to FIPS 203, Algorithm 6.
func polyByteDecode[T ~[n]fieldElement](b []byte) (T, error) {
	if len(b) != encodingSize12 {
		return T{}, errors.New("mlkem: invalid encoding length")
	}
	var f T
	for i := 0; i < n; i += 2 {
		d := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
		const mask12 = 0b1111_1111_1111
		var err error
		if f[i], err = fieldCheckReduced(uint16(d & mask12)); err != nil {
			return T{}, errors.New("mlkem: invalid polynomial encoding")
		}
		if f[i+1], err = fieldCheckReduced(uint16(d >> 12)); err != nil {
			return T{}, errors.New("mlkem: invalid polynomial encoding")
		}
		b = b[3:]
	}
	return f, nil
}

// sliceForAppend takes a slice and a requested number of bytes. It returns a
// slice with the contents of the given slice followed by that many bytes and a
// second slice that aliases into it and contains only the extra bytes. If the
// original slice has sufficient capacity then no allocation is performed.
func sliceForAppend(in []byte, n int) (head, tail []byte) {
	if total := len(in) + n; cap(in) >= total {
		head = in[:total]
	} else {
		head = make([]byte, total)
		copy(head, in)
	}
	tail = head[len(in):]
	return
}

// ringCompressAndEncode1 appends a 32-byte encoding of a ring element to s,
// compressing one coefficients per bit.
//
// It implements Compress₁, according to FIPS 203, Definition 4.7,
// followed by ByteEncode₁, according to FIPS 203, Algorithm 5.
func ringCompressAndEncode1(s []byte, f ringElement) []byte {
	s, b := sliceForAppend(s, encodingSize1)
	clear(b)
	for i := range f {
		b[i/8] |= uint8(compress(f[i], 1) << (i % 8))
	}
	return s
}

// ringDecodeAndDecompress1 decodes a 32-byte slice to a ring element where each
// bit is mapped to 0 or ⌈q/2⌋.
//
// It implements ByteDecode₁, according to FIPS 203, Algorithm 6,
// followed by Decompress₁, according to FIPS 203, Definition 4.8.
func ringDecodeAndDecompress1(b *[encodingSize1]byte) ringElement {
	var f ringElement
	for i := range f {
		b_i := b[i/8] >> (i % 8) & 1
		const halfQ = (q + 1) / 2        // ⌈q/2⌋, rounded up per FIPS 203, Section 2.3
		f[i] = fieldElement(b_i) * halfQ // 0 decompresses to 0, and 1 to ⌈q/2⌋
	}
	return f
}

// ringCompressAndEncode4 appends a 128-byte encoding of a ring element to s,
// compressing two coefficients per byte.
//
// It implements Compress₄, according to FIPS 203, Definition 4.7,
// followed by ByteEncode₄, according to FIPS 203, Algorithm 5.
func ringCompressAndEncode4(s []byte, f ringElement) []byte {
	s, b := sliceForAppend(s, encodingSize4)
	for i := 0; i < n; i += 2 {
		b[i/2] = uint8(compress(f[i], 4) | compress(f[i+1], 4)<<4)
	}
	return s
}

// ringDecodeAndDecompress4 decodes a 128-byte encoding of a ring element where
// each four bits are mapped to an equidistant distribution.
//
// It implements ByteDecode₄, according to FIPS 203, Algorithm 6,
// followed by Decompress₄, according to FIPS 203, Definition 4.8.
func ringDecodeAndDecompress4(b *[encodingSize4]byte) ringElement {
	var f ringElement
	for i := 0; i < n; i += 2 {
		f[i] = fieldElement(decompress(uint16(b[i/2]&0b1111), 4))
		f[i+1] = fieldElement(decompress(uint16(b[i/2]>>4), 4))
	}
	return f
}

// ringCompressAndEncode10 appends a 320-byte encoding of a ring element to s,
// compressing four coefficients per five bytes.
//
// It implements Compress₁₀, according to FIPS 203, Definition 4.7,
// followed by ByteEncode₁₀, according to FIPS 203, Algorithm 5.
func ringCompressAndEncode10(s []byte, f ringElement) []byte {
	s, b := sliceForAppend(s, encodingSize10)
	for i := 0; i < n; i += 4 {
		var x uint64
		x |= uint64(compress(f[i], 10))
		x |= uint64(compress(f[i+1], 10)) << 10
		x |= uint64(compress(f[i+2], 10)) << 20
		x |= uint64(compress(f[i+3], 10)) << 30
		b[0] = uint8(x)
		b[1] = uint8(x >> 8)
		b[2] = uint8(x >> 16)
		b[3] = uint8(x >> 24)
		b[4] = uint8(x >> 32)
		b = b[5:]
	}
	return s
}

// ringDecodeAndDecompress10 decodes a 320-byte encoding of a ring element where
// each ten bits are mapped to an equidistant distribution.
//
// It implements ByteDecode₁₀, according to FIPS 203, Algorithm 6,
// followed by Decompress₁₀, according to FIPS 203, Definition 4.8.
func ringDecodeAndDecompress10(bb *[encodingSize10]byte) ringElement {
	b := bb[:]
	var f ringElement
	for i := 0; i < n; i += 4 {
		x := uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32
		b = b[5:]
		f[i] = fieldElement(decompress(uint16(x>>0&0b11_1111_1111), 10))
		f[i+1] = fieldElement(decompress(uint16(x>>10&0b11_1111_1111), 10))
		f[i+2] = fieldElement(decompress(uint16(x>>20&0b11_1111_1111), 10))
		f[i+3] = fieldElement(decompress(uint16(x>>30&0b11_1111_1111), 10))
	}
	return f
}

// ringCompressAndEncode5 appends a 160-byte encoding of a ring element to s,
// compressing eight coefficients per five bytes.
//
// It implements Compress₅, according to FIPS 203, Definition 4.7,
// followed by ByteEncode₅, according to FIPS 203, Algorithm 5.
func ringCompressAndEncode5(s []byte, f ringElement) []byte {
	s, b := sliceForAppend(s, encodingSize5)
	for i := 0; i < n; i += 8 {
		var v uint64
		v |= uint64(compress(f[i+0], 5)) << 0
		v |= uint64(compress(f[i+1], 5)) << 5
		v |= uint64(compress(f[i+2], 5)) << 10
		v |= uint64(compress(f[i+3], 5)) << 15
		v |= uint64(compress(f[i+4], 5)) << 20
		v |= uint64(compress(f[i+5], 5)) << 25
		v |= uint64(compress(f[i+6], 5)) << 30
		v |= uint64(compress(f[i+7], 5)) << 35
		b[0] = byte(v >> 0)
		b[1] = byte(v >> 8)
		b[2] = byte(v >> 16)
		b[3] = byte(v >> 24)
		b[4] = byte(v >> 32)
		b = b[5:]
	}
	return s
}

// ringDecodeAndDecompress5 decodes a 160-byte encoding of a ring element where
// each five bits are mapped to an equidistant distribution.
//
// It implements ByteDecode₅, according to FIPS 203, Algorithm 6,
// followed by Decompress₅, according to FIPS 203, Definition 4.8.
func ringDecodeAndDecompress5(bb *[encodingSize5]byte) ringElement {
	b := bb[:]
	var f ringElement
	for i := 0; i < n; i += 8 {
		v := uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32
		f[i+0] = decompress(uint16(v>>0)&31, 5)
		f[i+1] = decompress(uint16(v>>5)&31, 5)
		f[i+2] = decompress(uint16(v>>10)&31, 5)
		f[i+3] = decompress(uint16(v>>15)&31, 5)
		f[i+4] = decompress(uint16(v>>20)&31, 5)
		f[i+5] = decompress(uint16(v>>25)&31, 5)
		f[i+6] = decompress(uint16(v>>30)&31, 5)
		f[i+7] = decompress(uint16(v>>35)&31, 5)
		b = b[5:]
	}
	return f
}

// ringCompressAndEncode11 appends a 352-byte encoding of a ring element to s,
// compressing eight coefficients per eleven bytes.
//
// It implements Compress₁₁, according to FIPS 203, Definition 4.7,
// followed by ByteEncode₁₁, according to FIPS 203, Algorithm 5.
func ringCompressAndEncode11(s []byte, f ringElement) []byte {
	s, b := sliceForAppend(s, encodingSize11)
	for i := 0; i < n; i += 8 {
		t0 := compress(f[i+0], 11)
		t1 := compress(f[i+1], 11)
		t2 := compress(f[i+2], 11)
		t3 := compress(f[i+3], 11)
		t4 := compress(f[i+4], 11)
		t5 := compress(f[i+5], 11)
		t6 := compress(f[i+6], 11)
		t7 := compress(f[i+7], 11)
		b[0] = byte(t0)
		b[1] = byte(t0>>8 | t1<<3)
		b[2] = byte(t1>>5 | t2<<6)
		b[3] = byte(t2 >> 2)
		b[4] = byte(t2>>10 | t3<<1)
		b[5] = byte(t3>>7 | t4<<4)
		b[6] = byte(t4>>4 | t5<<7)
		b[7] = byte(t5 >> 1)
		b[8] = byte(t5>>9 | t6<<2)
		b[9] = byte(t6>>6 | t7<<5)
		b[10] = byte(t7 >> 3)
		b = b[11:]
	}
	return s
}

// ringDecodeAndDecompress11 decodes a 352-byte encoding of a ring element where
// each eleven bits are mapped to an equidistant distribution.
//
// It implements ByteDecode₁₁, according to FIPS 203, Algorithm 6,
// followed by Decompress₁₁, according to FIPS 203, Definition 4.8.
func ringDecodeAndDecompress11(bb *[encodingSize11]byte) ringElement {
	b := bb[:]
	var f ringElement
	for i := 0; i < n; i += 8 {
		t0 := uint16(b[0]) | uint16(b[1])<<8
		f[i+0] = decompress(t0&2047, 11)
		t1 := uint16(b[1])>>3 | uint16(b[2])<<5
		f[i+1] = decompress(t1&2047, 11)
		t2 := uint16(b[2])>>6 | uint16(b[3])<<2 | uint16(b[4])<<10
		f[i+2] = decompress(t2&2047, 11)
		t3 := uint16(b[4])>>1 | uint16(b[5])<<7
		f[i+3] = decompress(t3&2047, 11)
		t4 := uint16(b[5])>>4 | uint16(b[6])<<4
		f[i+4] = decompress(t4&2047, 11)
		t5 := uint16(b[6])>>7 | uint16(b[7])<<1 | uint16(b[8])<<9
		f[i+5] = decompress(t5&2047, 11)
		t6 := uint16(b[8])>>2 | uint16(b[9])<<6
		f[i+6] = decompress(t6&2047, 11)
		t7 := uint16(b[9])>>5 | uint16(b[10])<<3
		f[i+7] = decompress(t7&2047, 11)
		b = b[11:]
	}
	return f
}

// samplePolyCBD draws a ringElement from the special Dη distribution given a
// stream of random bytes generated by the PRF function, according to FIPS 203,
// Algorithm 8 and Definition 4.3.
func samplePolyCBD(s []byte, b byte) ringElement {
	prf := sha3.NewShake256()
	prf.Write(s)
	prf.Write([]byte{b})
	B := make([]byte, 64*2) // η = 2
	prf.Read(B)

	// SamplePolyCBD simply draws four (2η) bits for each coefficient, and adds
	// the first two and subtracts the last two.

	var f ringElement
	for i := 0; i < n; i += 2 {
		b := B[i/2]
		b_7, b_6, b_5, b_4 := b>>7, b>>6&1, b>>5&1, b>>4&1
		b_3, b_2, b_1, b_0 := b>>3&1, b>>2&1, b>>1&1, b&1
		f[i] = fieldSub(fieldElement(b_0+b_1), fieldElement(b_2+b_3))
		f[i+1] = fieldSub(fieldElement(b_4+b_5), fieldElement(b_6+b_7))
	}
	return f
}

// nttElement is an NTT representation, an element of T_q, represented as an
// array according to FIPS 203, Section 2.4.4.
type nttElement [n]fieldElement

// gammas are the values ζ^2BitRev7(i)+1 mod q for each index i, according to
// FIPS 203, Appendix A (with negative values reduced to positive).
var gammas = [128]fieldElement{17, 3312, 2761, 568, 583, 2746, 2649, 680, 1637, 1692, 723, 2606, 2288, 1041, 1100, 2229, 1409, 1920, 2662, 667, 3281, 48, 233, 3096, 756, 2573, 2156, 1173, 3015, 314, 3050, 279, 1703, 1626, 1651, 1678, 2789, 540, 1789, 1540, 1847, 1482, 952, 2377, 1461, 1868, 2687, 642, 939, 2390, 2308, 1021, 2437, 892, 2388, 941, 733, 2596, 2337, 992, 268, 3061, 641, 2688, 1584, 1745, 2298, 1031, 2037, 1292, 3220, 109, 375, 2954, 2549, 780, 2090, 1239, 1645, 1684, 1063, 2266, 319, 3010, 2773, 556, 757, 2572, 2099, 1230, 561, 2768, 2466, 863, 2594, 735, 2804, 525, 1092, 2237, 403, 2926, 1026, 2303, 1143, 2186, 2150, 1179, 2775, 554, 886, 2443, 1722, 1607, 1212, 2117, 1874, 1455, 1029, 2300, 2110, 1219, 2935, 394, 885, 2444, 2154, 1175}

// nttMulAdd adds f*g to h without materializing the product.
// All inputs and outputs are reduced. h may alias f or g.
func nttMulAdd(h, f, g *nttElement) {
	for i := 0; i < n; i += 2 {
		a0, a1 := f[i], f[i+1]
		b0, b1 := g[i], g[i+1]
		// Both sums are at most 2(q-1)^2 + (q-1) < 2q².
		h[i] = fieldReduce(uint32(h[i]) + uint32(a0)*uint32(b0) + uint32(fieldMul(a1, b1))*uint32(gammas[i/2]))
		h[i+1] = fieldReduce(uint32(h[i+1]) + uint32(a0)*uint32(b1) + uint32(a1)*uint32(b0))
	}
}

// nttMulPrecompute computes the gamma-scaled odd coefficients of g.
// The result is local to one matrix/vector multiplication.
func nttMulPrecompute(p *[n / 2]fieldElement, g *nttElement) {
	for i := range p {
		p[i] = fieldMul(g[2*i+1], gammas[i])
	}
}

// nttMulAddPrecomputed adds f*g to h, given nttMulPrecompute(g).
// All inputs and outputs are reduced. h may alias f or g.
func nttMulAddPrecomputed(h, f, g *nttElement, gamma *[n / 2]fieldElement) {
	for i := 0; i < n; i += 2 {
		a0, a1 := f[i], f[i+1]
		b0, b1 := g[i], g[i+1]
		// Both sums are at most 2(q-1)^2 + (q-1) < 2q².
		h[i] = fieldReduce(uint32(h[i]) + uint32(a0)*uint32(b0) + uint32(a1)*uint32(gamma[i/2]))
		h[i+1] = fieldReduce(uint32(h[i+1]) + uint32(a0)*uint32(b1) + uint32(a1)*uint32(b0))
	}
}

// zetas are the values ζ^BitRev7(k) mod q for each index k, according to FIPS
// 203, Appendix A.
var zetas = [128]fieldElement{1, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797, 2786, 3260, 569, 1746, 296, 2447, 1339, 1476, 3046, 56, 2240, 1333, 1426, 2094, 535, 2882, 2393, 2879, 1974, 821, 289, 331, 3253, 1756, 1197, 2304, 2277, 2055, 650, 1977, 2513, 632, 2865, 33, 1320, 1915, 2319, 1435, 807, 452, 1438, 2868, 1534, 2402, 2647, 2617, 1481, 648, 2474, 3110, 1227, 910, 17, 2761, 583, 2649, 1637, 723, 2288, 1100, 1409, 2662, 3281, 233, 756, 2156, 3015, 3050, 1703, 1651, 2789, 1789, 1847, 952, 1461, 2687, 939, 2308, 2437, 2388, 733, 2337, 268, 641, 1584, 2298, 2037, 3220, 375, 2549, 2090, 1645, 1063, 319, 2773, 757, 2099, 561, 2466, 2594, 2804, 1092, 403, 1026, 1143, 2150, 2775, 886, 1722, 1212, 1874, 1029, 2110, 2935, 885, 2154}

// ntt maps a ringElement to its nttElement representation.
//
// It implements NTT, according to FIPS 203, Algorithm 9.
func ntt(f ringElement) nttElement {
	k := 1
	for len := 128; len >= 2; len /= 2 {
		for start := 0; start < 256; start += 2 * len {
			zeta := zetas[k]
			k++
			// Bounds check elimination hint.
			f, flen := f[start:start+len], f[start+len:start+len+len]
			for j := 0; j < len; j++ {
				t := fieldMul(zeta, flen[j])
				flen[j] = fieldSub(f[j], t)
				f[j] = fieldAdd(f[j], t)
			}
		}
	}
	return nttElement(f)
}

// inverseNTT maps a nttElement back to the ringElement it represents.
//
// It implements NTT⁻¹, according to FIPS 203, Algorithm 10.
func inverseNTT(f nttElement) ringElement {
	k := 127
	for len := 2; len < 128; len *= 2 {
		for start := 0; start < 256; start += 2 * len {
			zeta := zetas[k]
			k--
			// Bounds check elimination hint.
			f, flen := f[start:start+len], f[start+len:start+len+len]
			for j := 0; j < len; j++ {
				t := f[j]
				f[j] = fieldAdd(t, flen[j])
				flen[j] = fieldMulSub(zeta, flen[j], t)
			}
		}
	}
	// The first six stages consume zetas[127] through zetas[2].
	// Fuse the final stage (zeta = zetas[1]) with scaling by 128⁻¹.
	const scale = 3303
	const zetaScaled = 1729 * scale % q // zetas[1] * 128⁻¹ mod q
	for i := 0; i < 128; i++ {
		a, b := f[i], f[i+128]
		// a+b < 2q, so (a+b)*scale < 2q², within fieldReduce's bound.
		f[i] = fieldReduce(uint32(a+b) * scale)
		f[i+128] = fieldMulSub(zetaScaled, b, a)
	}
	return ringElement(f)
}

// sampleNTTMatrix fills A so that A[i*k+j] is drawn from the XOF seeded with
// ρ, j, and i, sampling the underlying streams in batches with
// sha3.ReadMulti.
//
// It implements the matrix generation loops of K-PKE.KeyGen and K-PKE.Encrypt
// according to FIPS 203, Algorithms 13 and 14, with SampleNTT according to
// FIPS 203, Algorithm 7.
func sampleNTTMatrix(A []nttElement, rho []byte, k int) {
	// Compute up to four elements at a time in parallel.
	for len(A) > 0 {
		lanesCount := min(4, len(A))
		lanes := [4]*sha3.SHAKE{
			sha3.NewShake128(), sha3.NewShake128(),
			sha3.NewShake128(), sha3.NewShake128(),
		}
		// Squeeze 3 blocks (each of 168 bytes, the rate of SHAKE-128) at once
		// from each lane, which is enough to fill a full nttElement more than
		// 99% of the time.
		var bufs [4][3 * 168]byte
		var outs [4][]byte
		for i := range lanesCount {
			// The index of A[i] in the full k×k matrix.
			m := k*k - len(A) + i
			lanes[i].Write(rho)
			lanes[i].Write([]byte{byte(m % k), byte(m / k)})
			outs[i] = bufs[i][:]
		}
		sha3.ReadMulti(lanes[:lanesCount], outs[:lanesCount])
		for i := range lanesCount {
			a := &A[i]
			j := parseSampleNTT(a, 0, bufs[i][:])
			// If the eager read above wasn't enough, draw a block at a time
			// from this lane until the element is filled.
			for j < n {
				lanes[i].Read(bufs[i][:168])
				j = parseSampleNTT(a, j, bufs[i][:168])
			}
		}
		A = A[lanesCount:]
	}
}

// parseSampleNTT consumes buf, whose length must be a multiple of three,
// filling a from index j on with accepted values, and returns the updated
// index.
//
// SampleNTT essentially draws 12 bits at a time from buf, interprets them
// in little-endian, and rejects values higher than q, until it drew 256
// values. (The rejection rate is approximately 19%.)
func parseSampleNTT(a *nttElement, j int, buf []byte) int {
	// For as long as there are enough bytes in buf and enough remaining slots
	// in a, consume 6 bytes (48 bits) at a time as four 12-bit values,
	// advancing j branchlessly. (The uint conversion lets the compiler prove
	// j is non-negative and drop the bounds checks on the stores below.)
	_ = a[0] // hoist a's nil check out of the loops below
	const mask = 0b1111_1111_1111
	for len(buf) >= 8 && uint(j) <= n-4 {
		w := byteorder.LEUint64(buf)
		d1 := uint32(w) & mask
		d2 := uint32(w>>12) & mask
		d3 := uint32(w>>24) & mask
		d4 := uint32(w>>36) & mask
		a[j] = fieldElement(d1)
		j += int((d1 - q) >> 31)
		a[j] = fieldElement(d2)
		j += int((d2 - q) >> 31)
		a[j] = fieldElement(d3)
		j += int((d3 - q) >> 31)
		a[j] = fieldElement(d4)
		j += int((d4 - q) >> 31)
		buf = buf[6:]
	}
	// Consume the rest of the buffer 3 bytes (two values) at a time. This
	// handles both the last six bytes (which the loop above can't load, as it
	// would read past the end of buf) and the last three elements (which the
	// loop above can't fill, as it would write past the end of a).
	for len(buf) > 0 && j < n {
		//
		//          r₀              r₁              r₂
		//  |- - - - - - - -|- - - - - - - -|- - - - - - - -|
		//
		//             Uint16(buf)
		//  |- - - - - - - - - - - - - - - -|
		//  |- - - - - - - - - - - -|
		//              d₁
		//
		//                            Uint16(buf[1:])
		//                  |- - - - - - - - - - - - - - - -|
		//                          |- - - - - - - - - - - -|
		//                                      d₂
		//
		d1 := byteorder.LEUint16(buf) & mask
		d2 := byteorder.LEUint16(buf[1:]) >> 4
		if d1 < q {
			a[j] = fieldElement(d1)
			j++
		}
		if d2 < q && j < n {
			a[j] = fieldElement(d2)
			j++
		}
		buf = buf[3:]
	}
	return j
}

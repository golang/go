// Copyright (c) 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package field implements fast arithmetic modulo 2^255-19.
package field

import (
	_ "crypto/internal/fips140/check"
	"crypto/internal/fips140/subtle"
	"crypto/internal/fips140deps/byteorder"
	"errors"
	"math/bits"
)

// Element represents an element of the field GF(2^255-19). Note that this
// is not a cryptographically secure group, and should only be used to interact
// with edwards25519.Point coordinates.
//
// This type works similarly to math/big.Int, and all arguments and receivers
// are allowed to alias.
//
// The zero value is a valid zero element.
type Element struct {
	// An element t represents the integer
	//     t.l0 + t.l1*2^51 + t.l2*2^102 + t.l3*2^153 + t.l4*2^204
	//
	// Between operations, all limbs are expected to be lower than 2^52.
	l0 uint64
	l1 uint64
	l2 uint64
	l3 uint64
	l4 uint64
}

const maskLow51Bits uint64 = (1 << 51) - 1

var feZero = &Element{0, 0, 0, 0, 0}

// Zero sets v = 0, and returns v.
func (v *Element) Zero() *Element {
	*v = *feZero
	return v
}

var feOne = &Element{1, 0, 0, 0, 0}

// One sets v = 1, and returns v.
func (v *Element) One() *Element {
	*v = *feOne
	return v
}

// reduce reduces v modulo 2^255 - 19 and returns it.
func (v *Element) reduce() *Element {
	v.carryPropagate()

	// After the light reduction we now have a field element representation
	// v < 2^255 + 2^13 * 19, but need v < 2^255 - 19.

	// If v >= 2^255 - 19, then v + 19 >= 2^255, which would overflow 2^255 - 1,
	// generating a carry. That is, c will be 0 if v < 2^255 - 19, and 1 otherwise.
	c := (v.l0 + 19) >> 51
	c = (v.l1 + c) >> 51
	c = (v.l2 + c) >> 51
	c = (v.l3 + c) >> 51
	c = (v.l4 + c) >> 51

	// If v < 2^255 - 19 and c = 0, this will be a no-op. Otherwise, it's
	// effectively applying the reduction identity to the carry.
	v.l0 += 19 * c

	v.l1 += v.l0 >> 51
	v.l0 = v.l0 & maskLow51Bits
	v.l2 += v.l1 >> 51
	v.l1 = v.l1 & maskLow51Bits
	v.l3 += v.l2 >> 51
	v.l2 = v.l2 & maskLow51Bits
	v.l4 += v.l3 >> 51
	v.l3 = v.l3 & maskLow51Bits
	// no additional carry
	v.l4 = v.l4 & maskLow51Bits

	return v
}

// Add sets v = a + b, and returns v.
func (v *Element) Add(a, b *Element) *Element {
	v.l0 = a.l0 + b.l0
	v.l1 = a.l1 + b.l1
	v.l2 = a.l2 + b.l2
	v.l3 = a.l3 + b.l3
	v.l4 = a.l4 + b.l4
	// Using the generic implementation here is actually faster than the
	// assembly. Probably because the body of this function is so simple that
	// the compiler can figure out better optimizations by inlining the carry
	// propagation.
	return v.carryPropagateGeneric()
}

// Subtract sets v = a - b, and returns v.
func (v *Element) Subtract(a, b *Element) *Element {
	// We first add 2 * p, to guarantee the subtraction won't underflow, and
	// then subtract b (which can be up to 2^255 + 2^13 * 19).
	v.l0 = (a.l0 + 0xFFFFFFFFFFFDA) - b.l0
	v.l1 = (a.l1 + 0xFFFFFFFFFFFFE) - b.l1
	v.l2 = (a.l2 + 0xFFFFFFFFFFFFE) - b.l2
	v.l3 = (a.l3 + 0xFFFFFFFFFFFFE) - b.l3
	v.l4 = (a.l4 + 0xFFFFFFFFFFFFE) - b.l4
	return v.carryPropagate()
}

// Negate sets v = -a, and returns v.
func (v *Element) Negate(a *Element) *Element {
	return v.Subtract(feZero, a)
}

// Invert sets v = 1/z mod p, and returns v.
//
// If z == 0, Invert returns v = 0.
func (v *Element) Invert(z *Element) *Element {
	// Inversion is implemented as exponentiation with exponent p − 2. It uses the
	// same sequence of 255 squarings and 11 multiplications as [Curve25519].
	var z2, z9, z11, z2_5_0, z2_10_0, z2_20_0, z2_50_0, z2_100_0, t Element

	z2.Square(z)             // 2
	t.Square(&z2)            // 4
	t.Square(&t)             // 8
	z9.Multiply(&t, z)       // 9
	z11.Multiply(&z9, &z2)   // 11
	t.Square(&z11)           // 22
	z2_5_0.Multiply(&t, &z9) // 31 = 2^5 - 2^0

	t.Square(&z2_5_0) // 2^6 - 2^1
	for i := 0; i < 4; i++ {
		t.Square(&t) // 2^10 - 2^5
	}
	z2_10_0.Multiply(&t, &z2_5_0) // 2^10 - 2^0

	t.Square(&z2_10_0) // 2^11 - 2^1
	for i := 0; i < 9; i++ {
		t.Square(&t) // 2^20 - 2^10
	}
	z2_20_0.Multiply(&t, &z2_10_0) // 2^20 - 2^0

	t.Square(&z2_20_0) // 2^21 - 2^1
	for i := 0; i < 19; i++ {
		t.Square(&t) // 2^40 - 2^20
	}
	t.Multiply(&t, &z2_20_0) // 2^40 - 2^0

	t.Square(&t) // 2^41 - 2^1
	for i := 0; i < 9; i++ {
		t.Square(&t) // 2^50 - 2^10
	}
	z2_50_0.Multiply(&t, &z2_10_0) // 2^50 - 2^0

	t.Square(&z2_50_0) // 2^51 - 2^1
	for i := 0; i < 49; i++ {
		t.Square(&t) // 2^100 - 2^50
	}
	z2_100_0.Multiply(&t, &z2_50_0) // 2^100 - 2^0

	t.Square(&z2_100_0) // 2^101 - 2^1
	for i := 0; i < 99; i++ {
		t.Square(&t) // 2^200 - 2^100
	}
	t.Multiply(&t, &z2_100_0) // 2^200 - 2^0

	t.Square(&t) // 2^201 - 2^1
	for i := 0; i < 49; i++ {
		t.Square(&t) // 2^250 - 2^50
	}
	t.Multiply(&t, &z2_50_0) // 2^250 - 2^0

	t.Square(&t) // 2^251 - 2^1
	t.Square(&t) // 2^252 - 2^2
	t.Square(&t) // 2^253 - 2^3
	t.Square(&t) // 2^254 - 2^4
	t.Square(&t) // 2^255 - 2^5

	return v.Multiply(&t, &z11) // 2^255 - 21
}

// Set sets v = a, and returns v.
func (v *Element) Set(a *Element) *Element {
	*v = *a
	return v
}

// SetBytes sets v to x, where x is a 32-byte little-endian encoding. If x is
// not of the right length, SetBytes returns nil and an error, and the
// receiver is unchanged.
//
// Consistent with RFC 7748, the most significant bit (the high bit of the
// last byte) is ignored, and non-canonical values (2^255-19 through 2^255-1)
// are accepted. Note that this is laxer than specified by RFC 8032, but
// consistent with most Ed25519 implementations.
func (v *Element) SetBytes(x []byte) (*Element, error) {
	if len(x) != 32 {
		return nil, errors.New("edwards25519: invalid field element input size")
	}

	// Bits 0:51 (bytes 0:8, bits 0:64, shift 0, mask 51).
	v.l0 = byteorder.LEUint64(x[0:8])
	v.l0 &= maskLow51Bits
	// Bits 51:102 (bytes 6:14, bits 48:112, shift 3, mask 51).
	v.l1 = byteorder.LEUint64(x[6:14]) >> 3
	v.l1 &= maskLow51Bits
	// Bits 102:153 (bytes 12:20, bits 96:160, shift 6, mask 51).
	v.l2 = byteorder.LEUint64(x[12:20]) >> 6
	v.l2 &= maskLow51Bits
	// Bits 153:204 (bytes 19:27, bits 152:216, shift 1, mask 51).
	v.l3 = byteorder.LEUint64(x[19:27]) >> 1
	v.l3 &= maskLow51Bits
	// Bits 204:255 (bytes 24:32, bits 192:256, shift 12, mask 51).
	// Note: not bytes 25:33, shift 4, to avoid overread.
	v.l4 = byteorder.LEUint64(x[24:32]) >> 12
	v.l4 &= maskLow51Bits

	return v, nil
}

// Bytes returns the canonical 32-byte little-endian encoding of v.
func (v *Element) Bytes() []byte {
	// This function is outlined to make the allocations inline in the caller
	// rather than happen on the heap.
	var out [32]byte
	return v.bytes(&out)
}

func (v *Element) bytes(out *[32]byte) []byte {
	t := *v
	t.reduce()

	// Pack five 51-bit limbs into four 64-bit words:
	//
	//  255    204    153    102     51      0
	//    ├──l4──┼──l3──┼──l2──┼──l1──┼──l0──┤
	//   ├───u3───┼───u2───┼───u1───┼───u0───┤
	// 256      192      128       64        0

	u0 := t.l1<<51 | t.l0
	u1 := t.l2<<(102-64) | t.l1>>(64-51)
	u2 := t.l3<<(153-128) | t.l2>>(128-102)
	u3 := t.l4<<(204-192) | t.l3>>(192-153)

	byteorder.LEPutUint64(out[0*8:], u0)
	byteorder.LEPutUint64(out[1*8:], u1)
	byteorder.LEPutUint64(out[2*8:], u2)
	byteorder.LEPutUint64(out[3*8:], u3)

	return out[:]
}

// Equal returns 1 if v and u are equal, and 0 otherwise.
func (v *Element) Equal(u *Element) int {
	sa, sv := u.Bytes(), v.Bytes()
	return subtle.ConstantTimeCompare(sa, sv)
}

// mask64Bits returns 0xffffffff if cond is 1, and 0 otherwise.
func mask64Bits(cond int) uint64 { return ^(uint64(cond) - 1) }

// Select sets v to a if cond == 1, and to b if cond == 0.
func (v *Element) Select(a, b *Element, cond int) *Element {
	m := mask64Bits(cond)
	v.l0 = (m & a.l0) | (^m & b.l0)
	v.l1 = (m & a.l1) | (^m & b.l1)
	v.l2 = (m & a.l2) | (^m & b.l2)
	v.l3 = (m & a.l3) | (^m & b.l3)
	v.l4 = (m & a.l4) | (^m & b.l4)
	return v
}

// Swap swaps v and u if cond == 1 or leaves them unchanged if cond == 0, and returns v.
func (v *Element) Swap(u *Element, cond int) {
	m := mask64Bits(cond)
	t := m & (v.l0 ^ u.l0)
	v.l0 ^= t
	u.l0 ^= t
	t = m & (v.l1 ^ u.l1)
	v.l1 ^= t
	u.l1 ^= t
	t = m & (v.l2 ^ u.l2)
	v.l2 ^= t
	u.l2 ^= t
	t = m & (v.l3 ^ u.l3)
	v.l3 ^= t
	u.l3 ^= t
	t = m & (v.l4 ^ u.l4)
	v.l4 ^= t
	u.l4 ^= t
}

// IsNegative returns 1 if v is negative, and 0 otherwise.
func (v *Element) IsNegative() int {
	return int(v.Bytes()[0] & 1)
}

// Absolute sets v to |u|, and returns v.
func (v *Element) Absolute(u *Element) *Element {
	return v.Select(new(Element).Negate(u), u, u.IsNegative())
}

// Multiply sets v = x * y, and returns v.
func (v *Element) Multiply(x, y *Element) *Element {
	feMul(v, x, y)
	return v
}

// Square sets v = x * x, and returns v.
func (v *Element) Square(x *Element) *Element {
	feSquare(v, x)
	return v
}

// Mult32 sets v = x * y, and returns v.
func (v *Element) Mult32(x *Element, y uint32) *Element {
	x0lo, x0hi := mul51(x.l0, y)
	x1lo, x1hi := mul51(x.l1, y)
	x2lo, x2hi := mul51(x.l2, y)
	x3lo, x3hi := mul51(x.l3, y)
	x4lo, x4hi := mul51(x.l4, y)
	v.l0 = x0lo + 19*x4hi // carried over per the reduction identity
	v.l1 = x1lo + x0hi
	v.l2 = x2lo + x1hi
	v.l3 = x3lo + x2hi
	v.l4 = x4lo + x3hi
	// The hi portions are going to be only 32 bits, plus any previous excess,
	// so we can skip the carry propagation.
	return v
}

// mul51 returns lo + hi * 2⁵¹ = a * b.
func mul51(a uint64, b uint32) (lo uint64, hi uint64) {
	mh, ml := bits.Mul64(a, uint64(b))
	lo = ml & maskLow51Bits
	hi = (mh << 13) | (ml >> 51)
	return
}

// Pow22523 set v = x^((p-5)/8), and returns v. (p-5)/8 is 2^252-3.
func (v *Element) Pow22523(x *Element) *Element {
	var t0, t1, t2 Element

	t0.Square(x)             // x^2
	t1.Square(&t0)           // x^4
	t1.Square(&t1)           // x^8
	t1.Multiply(x, &t1)      // x^9
	t0.Multiply(&t0, &t1)    // x^11
	t0.Square(&t0)           // x^22
	t0.Multiply(&t1, &t0)    // x^31
	t1.Square(&t0)           // x^62
	for i := 1; i < 5; i++ { // x^992
		t1.Square(&t1)
	}
	t0.Multiply(&t1, &t0)     // x^1023 -> 1023 = 2^10 - 1
	t1.Square(&t0)            // 2^11 - 2
	for i := 1; i < 10; i++ { // 2^20 - 2^10
		t1.Square(&t1)
	}
	t1.Multiply(&t1, &t0)     // 2^20 - 1
	t2.Square(&t1)            // 2^21 - 2
	for i := 1; i < 20; i++ { // 2^40 - 2^20
		t2.Square(&t2)
	}
	t1.Multiply(&t2, &t1)     // 2^40 - 1
	t1.Square(&t1)            // 2^41 - 2
	for i := 1; i < 10; i++ { // 2^50 - 2^10
		t1.Square(&t1)
	}
	t0.Multiply(&t1, &t0)     // 2^50 - 1
	t1.Square(&t0)            // 2^51 - 2
	for i := 1; i < 50; i++ { // 2^100 - 2^50
		t1.Square(&t1)
	}
	t1.Multiply(&t1, &t0)      // 2^100 - 1
	t2.Square(&t1)             // 2^101 - 2
	for i := 1; i < 100; i++ { // 2^200 - 2^100
		t2.Square(&t2)
	}
	t1.Multiply(&t2, &t1)     // 2^200 - 1
	t1.Square(&t1)            // 2^201 - 2
	for i := 1; i < 50; i++ { // 2^250 - 2^50
		t1.Square(&t1)
	}
	t0.Multiply(&t1, &t0)     // 2^250 - 1
	t0.Square(&t0)            // 2^251 - 2
	t0.Square(&t0)            // 2^252 - 4
	return v.Multiply(&t0, x) // 2^252 - 3 -> x^(2^252-3)
}

// sqrtM1 is 2^((p-1)/4), which squared is equal to -1 by Euler's Criterion.
var sqrtM1 = &Element{1718705420411056, 234908883556509,
	2233514472574048, 2117202627021982, 765476049583133}

// SqrtRatio sets r to the non-negative square root of the ratio of u and v.
//
// If u/v is square, SqrtRatio returns r and 1. If u/v is not square, SqrtRatio
// sets r according to Section 4.3 of draft-irtf-cfrg-ristretto255-decaf448-00,
// and returns r and 0.
func (r *Element) SqrtRatio(u, v *Element) (R *Element, wasSquare int) {
	t0 := new(Element)

	// r = (u * v3) * (u * v7)^((p-5)/8)
	v2 := new(Element).Square(v)
	uv3 := new(Element).Multiply(u, t0.Multiply(v2, v))
	uv7 := new(Element).Multiply(uv3, t0.Square(v2))
	rr := new(Element).Multiply(uv3, t0.Pow22523(uv7))

	check := new(Element).Multiply(v, t0.Square(rr)) // check = v * r^2

	uNeg := new(Element).Negate(u)
	correctSignSqrt := check.Equal(u)
	flippedSignSqrt := check.Equal(uNeg)
	flippedSignSqrtI := check.Equal(t0.Multiply(uNeg, sqrtM1))

	rPrime := new(Element).Multiply(rr, sqrtM1) // r_prime = SQRT_M1 * r
	// r = CT_SELECT(r_prime IF flipped_sign_sqrt | flipped_sign_sqrt_i ELSE r)
	rr.Select(rPrime, rr, flippedSignSqrt|flippedSignSqrtI)

	r.Absolute(rr) // Choose the nonnegative square root.
	return r, correctSignSqrt | flippedSignSqrt
}

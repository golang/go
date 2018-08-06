// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

// This is a constant-time, 32-bit implementation of P224. See FIPS 186-3,
// section D.2.2.
//
// See https://www.imperialviolet.org/2010/12/04/ecc.html ([1]) for background.

import (
	"math/big"
)

var p224 p224Curve

type p224Curve struct {
	*CurveParams
	gx, gy, b p224FieldElement
}

func initP224() {
	// See FIPS 186-3, section D.2.2
	p224.CurveParams = &CurveParams{Name: "P-224"}
	p224.P, _ = new(big.Int).SetString("26959946667150639794667015087019630673557916260026308143510066298881", 10)
	p224.N, _ = new(big.Int).SetString("26959946667150639794667015087019625940457807714424391721682722368061", 10)
	p224.A, _ = new(big.Int).SetString("fffffffffffffffffffffffffffffffefffffffffffffffffffffffe", 16)
	p224.B, _ = new(big.Int).SetString("b4050a850c04b3abf54132565044b0b7d7bfd8ba270b39432355ffb4", 16)
	p224.Gx, _ = new(big.Int).SetString("b70e0cbd6bb4bf7f321390b94a03c1d356c21122343280d6115c1d21", 16)
	p224.Gy, _ = new(big.Int).SetString("bd376388b5f723fb4c22dfe6cd4375a05a07476444d5819985007e34", 16)
	p224.BitSize = 224

	p224FromBig(&p224.gx, p224.Gx)
	p224FromBig(&p224.gy, p224.Gy)
	p224FromBig(&p224.b, p224.B)
}

// P224 returns a Curve which implements P-224 (see FIPS 186-3, section D.2.2).
//
// The cryptographic operations are implemented using constant-time algorithms.
func P224() Curve {
	initonce.Do(initAll)
	return p224
}

func (curve p224Curve) Params() *CurveParams {
	return curve.CurveParams
}

func (curve p224Curve) IsOnCurve(bigX, bigY *big.Int) bool {
	var x, y p224FieldElement
	p224FromBig(&x, bigX)
	p224FromBig(&y, bigY)

	// y² = x³ - 3x + b
	var tmp p224LargeFieldElement
	var x3 p224FieldElement
	p224Square(&x3, &x, &tmp)
	p224Mul(&x3, &x3, &x, &tmp)

	for i := 0; i < 8; i++ {
		x[i] *= 3
	}
	p224Sub(&x3, &x3, &x)
	p224Reduce(&x3)
	p224Add(&x3, &x3, &curve.b)
	p224Contract(&x3, &x3)

	p224Square(&y, &y, &tmp)
	p224Contract(&y, &y)

	for i := 0; i < 8; i++ {
		if y[i] != x3[i] {
			return false
		}
	}
	return true
}

func (p224Curve) Add(bigX1, bigY1, bigX2, bigY2 *big.Int) (x, y *big.Int) {
	var x1, y1, z1, x2, y2, z2, x3, y3, z3 p224FieldElement

	p224FromBig(&x1, bigX1)
	p224FromBig(&y1, bigY1)
	if bigX1.Sign() != 0 || bigY1.Sign() != 0 {
		z1[0] = 1
	}
	p224FromBig(&x2, bigX2)
	p224FromBig(&y2, bigY2)
	if bigX2.Sign() != 0 || bigY2.Sign() != 0 {
		z2[0] = 1
	}

	p224AddJacobian(&x3, &y3, &z3, &x1, &y1, &z1, &x2, &y2, &z2)
	return p224ToAffine(&x3, &y3, &z3)
}

func (p224Curve) Double(bigX1, bigY1 *big.Int) (x, y *big.Int) {
	var x1, y1, z1, x2, y2, z2 p224FieldElement

	p224FromBig(&x1, bigX1)
	p224FromBig(&y1, bigY1)
	z1[0] = 1

	p224DoubleJacobian(&x2, &y2, &z2, &x1, &y1, &z1)
	return p224ToAffine(&x2, &y2, &z2)
}

func (p224Curve) ScalarMult(bigX1, bigY1 *big.Int, scalar []byte) (x, y *big.Int) {
	var x1, y1, z1, x2, y2, z2 p224FieldElement

	p224FromBig(&x1, bigX1)
	p224FromBig(&y1, bigY1)
	z1[0] = 1

	p224ScalarMult(&x2, &y2, &z2, &x1, &y1, &z1, scalar)
	return p224ToAffine(&x2, &y2, &z2)
}

func (curve p224Curve) ScalarBaseMult(scalar []byte) (x, y *big.Int) {
	var z1, x2, y2, z2 p224FieldElement

	z1[0] = 1
	p224ScalarMult(&x2, &y2, &z2, &curve.gx, &curve.gy, &z1, scalar)
	return p224ToAffine(&x2, &y2, &z2)
}

// Field element functions.
//
// The field that we're dealing with is ℤ/pℤ where p = 2**224 - 2**96 + 1.
//
// Field elements are represented by a FieldElement, which is a typedef to an
// array of 8 uint32's. The value of a FieldElement, a, is:
//   a[0] + 2**28·a[1] + 2**56·a[1] + ... + 2**196·a[7]
//
// Using 28-bit limbs means that there's only 4 bits of headroom, which is less
// than we would really like. But it has the useful feature that we hit 2**224
// exactly, making the reflections during a reduce much nicer.
type p224FieldElement [8]uint32

// p224P is the order of the field, represented as a p224FieldElement.
var p224P = [8]uint32{1, 0, 0, 0xffff000, 0xfffffff, 0xfffffff, 0xfffffff, 0xfffffff}

// p224IsZero returns 1 if a == 0 mod p and 0 otherwise.
//
// a[i] < 2**29
func p224IsZero(a *p224FieldElement) uint32 {
	// Since a p224FieldElement contains 224 bits there are two possible
	// representations of 0: 0 and p.
	var minimal p224FieldElement
	p224Contract(&minimal, a)

	var isZero, isP uint32
	for i, v := range minimal {
		isZero |= v
		isP |= v - p224P[i]
	}

	// If either isZero or isP is 0, then we should return 1.
	isZero |= isZero >> 16
	isZero |= isZero >> 8
	isZero |= isZero >> 4
	isZero |= isZero >> 2
	isZero |= isZero >> 1

	isP |= isP >> 16
	isP |= isP >> 8
	isP |= isP >> 4
	isP |= isP >> 2
	isP |= isP >> 1

	// For isZero and isP, the LSB is 0 iff all the bits are zero.
	result := isZero & isP
	result = (^result) & 1

	return result
}

// p224Add computes *out = a+b
//
// a[i] + b[i] < 2**32
func p224Add(out, a, b *p224FieldElement) {
	for i := 0; i < 8; i++ {
		out[i] = a[i] + b[i]
	}
}

const two31p3 = 1<<31 + 1<<3
const two31m3 = 1<<31 - 1<<3
const two31m15m3 = 1<<31 - 1<<15 - 1<<3

// p224ZeroModP31 is 0 mod p where bit 31 is set in all limbs so that we can
// subtract smaller amounts without underflow. See the section "Subtraction" in
// [1] for reasoning.
var p224ZeroModP31 = []uint32{two31p3, two31m3, two31m3, two31m15m3, two31m3, two31m3, two31m3, two31m3}

// p224Sub computes *out = a-b
//
// a[i], b[i] < 2**30
// out[i] < 2**32
func p224Sub(out, a, b *p224FieldElement) {
	for i := 0; i < 8; i++ {
		out[i] = a[i] + p224ZeroModP31[i] - b[i]
	}
}

// LargeFieldElement also represents an element of the field. The limbs are
// still spaced 28-bits apart and in little-endian order. So the limbs are at
// 0, 28, 56, ..., 392 bits, each 64-bits wide.
type p224LargeFieldElement [15]uint64

const two63p35 = 1<<63 + 1<<35
const two63m35 = 1<<63 - 1<<35
const two63m35m19 = 1<<63 - 1<<35 - 1<<19

// p224ZeroModP63 is 0 mod p where bit 63 is set in all limbs. See the section
// "Subtraction" in [1] for why.
var p224ZeroModP63 = [8]uint64{two63p35, two63m35, two63m35, two63m35, two63m35m19, two63m35, two63m35, two63m35}

const bottom12Bits = 0xfff
const bottom28Bits = 0xfffffff

// p224Mul computes *out = a*b
//
// a[i] < 2**29, b[i] < 2**30 (or vice versa)
// out[i] < 2**29
func p224Mul(out, a, b *p224FieldElement, tmp *p224LargeFieldElement) {
	for i := 0; i < 15; i++ {
		tmp[i] = 0
	}

	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			tmp[i+j] += uint64(a[i]) * uint64(b[j])
		}
	}

	p224ReduceLarge(out, tmp)
}

// Square computes *out = a*a
//
// a[i] < 2**29
// out[i] < 2**29
func p224Square(out, a *p224FieldElement, tmp *p224LargeFieldElement) {
	for i := 0; i < 15; i++ {
		tmp[i] = 0
	}

	for i := 0; i < 8; i++ {
		for j := 0; j <= i; j++ {
			r := uint64(a[i]) * uint64(a[j])
			if i == j {
				tmp[i+j] += r
			} else {
				tmp[i+j] += r << 1
			}
		}
	}

	p224ReduceLarge(out, tmp)
}

// ReduceLarge converts a p224LargeFieldElement to a p224FieldElement.
//
// in[i] < 2**62
func p224ReduceLarge(out *p224FieldElement, in *p224LargeFieldElement) {
	for i := 0; i < 8; i++ {
		in[i] += p224ZeroModP63[i]
	}

	// Eliminate the coefficients at 2**224 and greater.
	for i := 14; i >= 8; i-- {
		in[i-8] -= in[i]
		in[i-5] += (in[i] & 0xffff) << 12
		in[i-4] += in[i] >> 16
	}
	in[8] = 0
	// in[0..8] < 2**64

	// As the values become small enough, we start to store them in |out|
	// and use 32-bit operations.
	for i := 1; i < 8; i++ {
		in[i+1] += in[i] >> 28
		out[i] = uint32(in[i] & bottom28Bits)
	}
	in[0] -= in[8]
	out[3] += uint32(in[8]&0xffff) << 12
	out[4] += uint32(in[8] >> 16)
	// in[0] < 2**64
	// out[3] < 2**29
	// out[4] < 2**29
	// out[1,2,5..7] < 2**28

	out[0] = uint32(in[0] & bottom28Bits)
	out[1] += uint32((in[0] >> 28) & bottom28Bits)
	out[2] += uint32(in[0] >> 56)
	// out[0] < 2**28
	// out[1..4] < 2**29
	// out[5..7] < 2**28
}

// Reduce reduces the coefficients of a to smaller bounds.
//
// On entry: a[i] < 2**31 + 2**30
// On exit: a[i] < 2**29
func p224Reduce(a *p224FieldElement) {
	for i := 0; i < 7; i++ {
		a[i+1] += a[i] >> 28
		a[i] &= bottom28Bits
	}
	top := a[7] >> 28
	a[7] &= bottom28Bits

	// top < 2**4
	mask := top
	mask |= mask >> 2
	mask |= mask >> 1
	mask <<= 31
	mask = uint32(int32(mask) >> 31)
	// Mask is all ones if top != 0, all zero otherwise

	a[0] -= top
	a[3] += top << 12

	// We may have just made a[0] negative but, if we did, then we must
	// have added something to a[3], this it's > 2**12. Therefore we can
	// carry down to a[0].
	a[3] -= 1 & mask
	a[2] += mask & (1<<28 - 1)
	a[1] += mask & (1<<28 - 1)
	a[0] += mask & (1 << 28)
}

// p224Invert calculates *out = in**-1 by computing in**(2**224 - 2**96 - 1),
// i.e. Fermat's little theorem.
func p224Invert(out, in *p224FieldElement) {
	var f1, f2, f3, f4 p224FieldElement
	var c p224LargeFieldElement

	p224Square(&f1, in, &c)    // 2
	p224Mul(&f1, &f1, in, &c)  // 2**2 - 1
	p224Square(&f1, &f1, &c)   // 2**3 - 2
	p224Mul(&f1, &f1, in, &c)  // 2**3 - 1
	p224Square(&f2, &f1, &c)   // 2**4 - 2
	p224Square(&f2, &f2, &c)   // 2**5 - 4
	p224Square(&f2, &f2, &c)   // 2**6 - 8
	p224Mul(&f1, &f1, &f2, &c) // 2**6 - 1
	p224Square(&f2, &f1, &c)   // 2**7 - 2
	for i := 0; i < 5; i++ {   // 2**12 - 2**6
		p224Square(&f2, &f2, &c)
	}
	p224Mul(&f2, &f2, &f1, &c) // 2**12 - 1
	p224Square(&f3, &f2, &c)   // 2**13 - 2
	for i := 0; i < 11; i++ {  // 2**24 - 2**12
		p224Square(&f3, &f3, &c)
	}
	p224Mul(&f2, &f3, &f2, &c) // 2**24 - 1
	p224Square(&f3, &f2, &c)   // 2**25 - 2
	for i := 0; i < 23; i++ {  // 2**48 - 2**24
		p224Square(&f3, &f3, &c)
	}
	p224Mul(&f3, &f3, &f2, &c) // 2**48 - 1
	p224Square(&f4, &f3, &c)   // 2**49 - 2
	for i := 0; i < 47; i++ {  // 2**96 - 2**48
		p224Square(&f4, &f4, &c)
	}
	p224Mul(&f3, &f3, &f4, &c) // 2**96 - 1
	p224Square(&f4, &f3, &c)   // 2**97 - 2
	for i := 0; i < 23; i++ {  // 2**120 - 2**24
		p224Square(&f4, &f4, &c)
	}
	p224Mul(&f2, &f4, &f2, &c) // 2**120 - 1
	for i := 0; i < 6; i++ {   // 2**126 - 2**6
		p224Square(&f2, &f2, &c)
	}
	p224Mul(&f1, &f1, &f2, &c) // 2**126 - 1
	p224Square(&f1, &f1, &c)   // 2**127 - 2
	p224Mul(&f1, &f1, in, &c)  // 2**127 - 1
	for i := 0; i < 97; i++ {  // 2**224 - 2**97
		p224Square(&f1, &f1, &c)
	}
	p224Mul(out, &f1, &f3, &c) // 2**224 - 2**96 - 1
}

// p224Contract converts a FieldElement to its unique, minimal form.
//
// On entry, in[i] < 2**29
// On exit, in[i] < 2**28
func p224Contract(out, in *p224FieldElement) {
	copy(out[:], in[:])

	for i := 0; i < 7; i++ {
		out[i+1] += out[i] >> 28
		out[i] &= bottom28Bits
	}
	top := out[7] >> 28
	out[7] &= bottom28Bits

	out[0] -= top
	out[3] += top << 12

	// We may just have made out[i] negative. So we carry down. If we made
	// out[0] negative then we know that out[3] is sufficiently positive
	// because we just added to it.
	for i := 0; i < 3; i++ {
		mask := uint32(int32(out[i]) >> 31)
		out[i] += (1 << 28) & mask
		out[i+1] -= 1 & mask
	}

	// We might have pushed out[3] over 2**28 so we perform another, partial,
	// carry chain.
	for i := 3; i < 7; i++ {
		out[i+1] += out[i] >> 28
		out[i] &= bottom28Bits
	}
	top = out[7] >> 28
	out[7] &= bottom28Bits

	// Eliminate top while maintaining the same value mod p.
	out[0] -= top
	out[3] += top << 12

	// There are two cases to consider for out[3]:
	//   1) The first time that we eliminated top, we didn't push out[3] over
	//      2**28. In this case, the partial carry chain didn't change any values
	//      and top is zero.
	//   2) We did push out[3] over 2**28 the first time that we eliminated top.
	//      The first value of top was in [0..16), therefore, prior to eliminating
	//      the first top, 0xfff1000 <= out[3] <= 0xfffffff. Therefore, after
	//      overflowing and being reduced by the second carry chain, out[3] <=
	//      0xf000. Thus it cannot have overflowed when we eliminated top for the
	//      second time.

	// Again, we may just have made out[0] negative, so do the same carry down.
	// As before, if we made out[0] negative then we know that out[3] is
	// sufficiently positive.
	for i := 0; i < 3; i++ {
		mask := uint32(int32(out[i]) >> 31)
		out[i] += (1 << 28) & mask
		out[i+1] -= 1 & mask
	}

	// Now we see if the value is >= p and, if so, subtract p.

	// First we build a mask from the top four limbs, which must all be
	// equal to bottom28Bits if the whole value is >= p. If top4AllOnes
	// ends up with any zero bits in the bottom 28 bits, then this wasn't
	// true.
	top4AllOnes := uint32(0xffffffff)
	for i := 4; i < 8; i++ {
		top4AllOnes &= out[i]
	}
	top4AllOnes |= 0xf0000000
	// Now we replicate any zero bits to all the bits in top4AllOnes.
	top4AllOnes &= top4AllOnes >> 16
	top4AllOnes &= top4AllOnes >> 8
	top4AllOnes &= top4AllOnes >> 4
	top4AllOnes &= top4AllOnes >> 2
	top4AllOnes &= top4AllOnes >> 1
	top4AllOnes = uint32(int32(top4AllOnes<<31) >> 31)

	// Now we test whether the bottom three limbs are non-zero.
	bottom3NonZero := out[0] | out[1] | out[2]
	bottom3NonZero |= bottom3NonZero >> 16
	bottom3NonZero |= bottom3NonZero >> 8
	bottom3NonZero |= bottom3NonZero >> 4
	bottom3NonZero |= bottom3NonZero >> 2
	bottom3NonZero |= bottom3NonZero >> 1
	bottom3NonZero = uint32(int32(bottom3NonZero<<31) >> 31)

	// Everything depends on the value of out[3].
	//    If it's > 0xffff000 and top4AllOnes != 0 then the whole value is >= p
	//    If it's = 0xffff000 and top4AllOnes != 0 and bottom3NonZero != 0,
	//      then the whole value is >= p
	//    If it's < 0xffff000, then the whole value is < p
	n := out[3] - 0xffff000
	out3Equal := n
	out3Equal |= out3Equal >> 16
	out3Equal |= out3Equal >> 8
	out3Equal |= out3Equal >> 4
	out3Equal |= out3Equal >> 2
	out3Equal |= out3Equal >> 1
	out3Equal = ^uint32(int32(out3Equal<<31) >> 31)

	// If out[3] > 0xffff000 then n's MSB will be zero.
	out3GT := ^uint32(int32(n) >> 31)

	mask := top4AllOnes & ((out3Equal & bottom3NonZero) | out3GT)
	out[0] -= 1 & mask
	out[3] -= 0xffff000 & mask
	out[4] -= 0xfffffff & mask
	out[5] -= 0xfffffff & mask
	out[6] -= 0xfffffff & mask
	out[7] -= 0xfffffff & mask
}

// Group element functions.
//
// These functions deal with group elements. The group is an elliptic curve
// group with a = -3 defined in FIPS 186-3, section D.2.2.

// p224AddJacobian computes *out = a+b where a != b.
func p224AddJacobian(x3, y3, z3, x1, y1, z1, x2, y2, z2 *p224FieldElement) {
	// See https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#addition-p224Add-2007-bl
	var z1z1, z2z2, u1, u2, s1, s2, h, i, j, r, v p224FieldElement
	var c p224LargeFieldElement

	z1IsZero := p224IsZero(z1)
	z2IsZero := p224IsZero(z2)

	// Z1Z1 = Z1²
	p224Square(&z1z1, z1, &c)
	// Z2Z2 = Z2²
	p224Square(&z2z2, z2, &c)
	// U1 = X1*Z2Z2
	p224Mul(&u1, x1, &z2z2, &c)
	// U2 = X2*Z1Z1
	p224Mul(&u2, x2, &z1z1, &c)
	// S1 = Y1*Z2*Z2Z2
	p224Mul(&s1, z2, &z2z2, &c)
	p224Mul(&s1, y1, &s1, &c)
	// S2 = Y2*Z1*Z1Z1
	p224Mul(&s2, z1, &z1z1, &c)
	p224Mul(&s2, y2, &s2, &c)
	// H = U2-U1
	p224Sub(&h, &u2, &u1)
	p224Reduce(&h)
	xEqual := p224IsZero(&h)
	// I = (2*H)²
	for j := 0; j < 8; j++ {
		i[j] = h[j] << 1
	}
	p224Reduce(&i)
	p224Square(&i, &i, &c)
	// J = H*I
	p224Mul(&j, &h, &i, &c)
	// r = 2*(S2-S1)
	p224Sub(&r, &s2, &s1)
	p224Reduce(&r)
	yEqual := p224IsZero(&r)
	if xEqual == 1 && yEqual == 1 && z1IsZero == 0 && z2IsZero == 0 {
		p224DoubleJacobian(x3, y3, z3, x1, y1, z1)
		return
	}
	for i := 0; i < 8; i++ {
		r[i] <<= 1
	}
	p224Reduce(&r)
	// V = U1*I
	p224Mul(&v, &u1, &i, &c)
	// Z3 = ((Z1+Z2)²-Z1Z1-Z2Z2)*H
	p224Add(&z1z1, &z1z1, &z2z2)
	p224Add(&z2z2, z1, z2)
	p224Reduce(&z2z2)
	p224Square(&z2z2, &z2z2, &c)
	p224Sub(z3, &z2z2, &z1z1)
	p224Reduce(z3)
	p224Mul(z3, z3, &h, &c)
	// X3 = r²-J-2*V
	for i := 0; i < 8; i++ {
		z1z1[i] = v[i] << 1
	}
	p224Add(&z1z1, &j, &z1z1)
	p224Reduce(&z1z1)
	p224Square(x3, &r, &c)
	p224Sub(x3, x3, &z1z1)
	p224Reduce(x3)
	// Y3 = r*(V-X3)-2*S1*J
	for i := 0; i < 8; i++ {
		s1[i] <<= 1
	}
	p224Mul(&s1, &s1, &j, &c)
	p224Sub(&z1z1, &v, x3)
	p224Reduce(&z1z1)
	p224Mul(&z1z1, &z1z1, &r, &c)
	p224Sub(y3, &z1z1, &s1)
	p224Reduce(y3)

	p224CopyConditional(x3, x2, z1IsZero)
	p224CopyConditional(x3, x1, z2IsZero)
	p224CopyConditional(y3, y2, z1IsZero)
	p224CopyConditional(y3, y1, z2IsZero)
	p224CopyConditional(z3, z2, z1IsZero)
	p224CopyConditional(z3, z1, z2IsZero)
}

// p224DoubleJacobian computes *out = a+a.
func p224DoubleJacobian(x3, y3, z3, x1, y1, z1 *p224FieldElement) {
	var delta, gamma, beta, alpha, t p224FieldElement
	var c p224LargeFieldElement

	p224Square(&delta, z1, &c)
	p224Square(&gamma, y1, &c)
	p224Mul(&beta, x1, &gamma, &c)

	// alpha = 3*(X1-delta)*(X1+delta)
	p224Add(&t, x1, &delta)
	for i := 0; i < 8; i++ {
		t[i] += t[i] << 1
	}
	p224Reduce(&t)
	p224Sub(&alpha, x1, &delta)
	p224Reduce(&alpha)
	p224Mul(&alpha, &alpha, &t, &c)

	// Z3 = (Y1+Z1)²-gamma-delta
	p224Add(z3, y1, z1)
	p224Reduce(z3)
	p224Square(z3, z3, &c)
	p224Sub(z3, z3, &gamma)
	p224Reduce(z3)
	p224Sub(z3, z3, &delta)
	p224Reduce(z3)

	// X3 = alpha²-8*beta
	for i := 0; i < 8; i++ {
		delta[i] = beta[i] << 3
	}
	p224Reduce(&delta)
	p224Square(x3, &alpha, &c)
	p224Sub(x3, x3, &delta)
	p224Reduce(x3)

	// Y3 = alpha*(4*beta-X3)-8*gamma²
	for i := 0; i < 8; i++ {
		beta[i] <<= 2
	}
	p224Sub(&beta, &beta, x3)
	p224Reduce(&beta)
	p224Square(&gamma, &gamma, &c)
	for i := 0; i < 8; i++ {
		gamma[i] <<= 3
	}
	p224Reduce(&gamma)
	p224Mul(y3, &alpha, &beta, &c)
	p224Sub(y3, y3, &gamma)
	p224Reduce(y3)
}

// p224CopyConditional sets *out = *in iff the least-significant-bit of control
// is true, and it runs in constant time.
func p224CopyConditional(out, in *p224FieldElement, control uint32) {
	control <<= 31
	control = uint32(int32(control) >> 31)

	for i := 0; i < 8; i++ {
		out[i] ^= (out[i] ^ in[i]) & control
	}
}

func p224ScalarMult(outX, outY, outZ, inX, inY, inZ *p224FieldElement, scalar []byte) {
	var xx, yy, zz p224FieldElement
	for i := 0; i < 8; i++ {
		outX[i] = 0
		outY[i] = 0
		outZ[i] = 0
	}

	for _, byte := range scalar {
		for bitNum := uint(0); bitNum < 8; bitNum++ {
			p224DoubleJacobian(outX, outY, outZ, outX, outY, outZ)
			bit := uint32((byte >> (7 - bitNum)) & 1)
			p224AddJacobian(&xx, &yy, &zz, inX, inY, inZ, outX, outY, outZ)
			p224CopyConditional(outX, &xx, bit)
			p224CopyConditional(outY, &yy, bit)
			p224CopyConditional(outZ, &zz, bit)
		}
	}
}

// p224ToAffine converts from Jacobian to affine form.
func p224ToAffine(x, y, z *p224FieldElement) (*big.Int, *big.Int) {
	var zinv, zinvsq, outx, outy p224FieldElement
	var tmp p224LargeFieldElement

	if isPointAtInfinity := p224IsZero(z); isPointAtInfinity == 1 {
		return new(big.Int), new(big.Int)
	}

	p224Invert(&zinv, z)
	p224Square(&zinvsq, &zinv, &tmp)
	p224Mul(x, x, &zinvsq, &tmp)
	p224Mul(&zinvsq, &zinvsq, &zinv, &tmp)
	p224Mul(y, y, &zinvsq, &tmp)

	p224Contract(&outx, x)
	p224Contract(&outy, y)
	return p224ToBig(&outx), p224ToBig(&outy)
}

// get28BitsFromEnd returns the least-significant 28 bits from buf>>shift,
// where buf is interpreted as a big-endian number.
func get28BitsFromEnd(buf []byte, shift uint) (uint32, []byte) {
	var ret uint32

	for i := uint(0); i < 4; i++ {
		var b byte
		if l := len(buf); l > 0 {
			b = buf[l-1]
			// We don't remove the byte if we're about to return and we're not
			// reading all of it.
			if i != 3 || shift == 4 {
				buf = buf[:l-1]
			}
		}
		ret |= uint32(b) << (8 * i) >> shift
	}
	ret &= bottom28Bits
	return ret, buf
}

// p224FromBig sets *out = *in.
func p224FromBig(out *p224FieldElement, in *big.Int) {
	bytes := in.Bytes()
	out[0], bytes = get28BitsFromEnd(bytes, 0)
	out[1], bytes = get28BitsFromEnd(bytes, 4)
	out[2], bytes = get28BitsFromEnd(bytes, 0)
	out[3], bytes = get28BitsFromEnd(bytes, 4)
	out[4], bytes = get28BitsFromEnd(bytes, 0)
	out[5], bytes = get28BitsFromEnd(bytes, 4)
	out[6], bytes = get28BitsFromEnd(bytes, 0)
	out[7], bytes = get28BitsFromEnd(bytes, 4)
}

// p224ToBig returns in as a big.Int.
func p224ToBig(in *p224FieldElement) *big.Int {
	var buf [28]byte
	buf[27] = byte(in[0])
	buf[26] = byte(in[0] >> 8)
	buf[25] = byte(in[0] >> 16)
	buf[24] = byte(((in[0] >> 24) & 0x0f) | (in[1]<<4)&0xf0)

	buf[23] = byte(in[1] >> 4)
	buf[22] = byte(in[1] >> 12)
	buf[21] = byte(in[1] >> 20)

	buf[20] = byte(in[2])
	buf[19] = byte(in[2] >> 8)
	buf[18] = byte(in[2] >> 16)
	buf[17] = byte(((in[2] >> 24) & 0x0f) | (in[3]<<4)&0xf0)

	buf[16] = byte(in[3] >> 4)
	buf[15] = byte(in[3] >> 12)
	buf[14] = byte(in[3] >> 20)

	buf[13] = byte(in[4])
	buf[12] = byte(in[4] >> 8)
	buf[11] = byte(in[4] >> 16)
	buf[10] = byte(((in[4] >> 24) & 0x0f) | (in[5]<<4)&0xf0)

	buf[9] = byte(in[5] >> 4)
	buf[8] = byte(in[5] >> 12)
	buf[7] = byte(in[5] >> 20)

	buf[6] = byte(in[6])
	buf[5] = byte(in[6] >> 8)
	buf[4] = byte(in[6] >> 16)
	buf[3] = byte(((in[6] >> 24) & 0x0f) | (in[7]<<4)&0xf0)

	buf[2] = byte(in[7] >> 4)
	buf[1] = byte(in[7] >> 12)
	buf[0] = byte(in[7] >> 20)

	return new(big.Int).SetBytes(buf[:])
}

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x

package elliptic

import (
	"math/big"
)

type p256CurveFast struct {
	*CurveParams
}

type p256Point struct {
	x [32]byte
	y [32]byte
	z [32]byte
}

var (
	p256        Curve
	p256PreFast *[37][64]p256Point
)

// hasVectorFacility reports whether the machine has the z/Architecture
// vector facility installed and enabled.
func hasVectorFacility() bool

var hasVX = hasVectorFacility()

func initP256Arch() {
	if hasVX {
		p256 = p256CurveFast{p256Params}
		initTable()
		return
	}

	// No vector support, use pure Go implementation.
	p256 = p256Curve{p256Params}
	return
}

func (curve p256CurveFast) Params() *CurveParams {
	return curve.CurveParams
}

// Functions implemented in p256_asm_s390x.s
// Montgomery multiplication modulo P256
func p256MulAsm(res, in1, in2 []byte)

// Montgomery square modulo P256
func p256Sqr(res, in []byte) {
	p256MulAsm(res, in, in)
}

// Montgomery multiplication by 1
func p256FromMont(res, in []byte)

// iff cond == 1  val <- -val
func p256NegCond(val *p256Point, cond int)

// if cond == 0 res <- b; else res <- a
func p256MovCond(res, a, b *p256Point, cond int)

// Constant time table access
func p256Select(point *p256Point, table []p256Point, idx int)
func p256SelectBase(point *p256Point, table []p256Point, idx int)

// Montgomery multiplication modulo Ord(G)
func p256OrdMul(res, in1, in2 []byte)

// Montgomery square modulo Ord(G), repeated n times
func p256OrdSqr(res, in []byte, n int) {
	copy(res, in)
	for i := 0; i < n; i += 1 {
		p256OrdMul(res, res, res)
	}
}

// Point add with P2 being affine point
// If sign == 1 -> P2 = -P2
// If sel == 0 -> P3 = P1
// if zero == 0 -> P3 = P2
func p256PointAddAffineAsm(P3, P1, P2 *p256Point, sign, sel, zero int)

// Point add
func p256PointAddAsm(P3, P1, P2 *p256Point)
func p256PointDoubleAsm(P3, P1 *p256Point)

func (curve p256CurveFast) Inverse(k *big.Int) *big.Int {
	if k.Cmp(p256Params.N) >= 0 {
		// This should never happen.
		reducedK := new(big.Int).Mod(k, p256Params.N)
		k = reducedK
	}

	// table will store precomputed powers of x. The 32 bytes at index
	// i store x^(i+1).
	var table [15][32]byte

	x := fromBig(k)
	// This code operates in the Montgomery domain where R = 2^256 mod n
	// and n is the order of the scalar field. (See initP256 for the
	// value.) Elements in the Montgomery domain take the form a×R and
	// multiplication of x and y in the calculates (x × y × R^-1) mod n. RR
	// is R×R mod n thus the Montgomery multiplication x and RR gives x×R,
	// i.e. converts x into the Montgomery domain. Stored in BigEndian form
	RR := []byte{0x66, 0xe1, 0x2d, 0x94, 0xf3, 0xd9, 0x56, 0x20, 0x28, 0x45, 0xb2, 0x39, 0x2b, 0x6b, 0xec, 0x59,
		0x46, 0x99, 0x79, 0x9c, 0x49, 0xbd, 0x6f, 0xa6, 0x83, 0x24, 0x4c, 0x95, 0xbe, 0x79, 0xee, 0xa2}

	p256OrdMul(table[0][:], x, RR)

	// Prepare the table, no need in constant time access, because the
	// power is not a secret. (Entry 0 is never used.)
	for i := 2; i < 16; i += 2 {
		p256OrdSqr(table[i-1][:], table[(i/2)-1][:], 1)
		p256OrdMul(table[i][:], table[i-1][:], table[0][:])
	}

	copy(x, table[14][:]) // f

	p256OrdSqr(x[0:32], x[0:32], 4)
	p256OrdMul(x[0:32], x[0:32], table[14][:]) // ff
	t := make([]byte, 32)
	copy(t, x)

	p256OrdSqr(x, x, 8)
	p256OrdMul(x, x, t) // ffff
	copy(t, x)

	p256OrdSqr(x, x, 16)
	p256OrdMul(x, x, t) // ffffffff
	copy(t, x)

	p256OrdSqr(x, x, 64) // ffffffff0000000000000000
	p256OrdMul(x, x, t)  // ffffffff00000000ffffffff
	p256OrdSqr(x, x, 32) // ffffffff00000000ffffffff00000000
	p256OrdMul(x, x, t)  // ffffffff00000000ffffffffffffffff

	// Remaining 32 windows
	expLo := [32]byte{0xb, 0xc, 0xe, 0x6, 0xf, 0xa, 0xa, 0xd, 0xa, 0x7, 0x1, 0x7, 0x9, 0xe, 0x8, 0x4,
		0xf, 0x3, 0xb, 0x9, 0xc, 0xa, 0xc, 0x2, 0xf, 0xc, 0x6, 0x3, 0x2, 0x5, 0x4, 0xf}
	for i := 0; i < 32; i++ {
		p256OrdSqr(x, x, 4)
		p256OrdMul(x, x, table[expLo[i]-1][:])
	}

	// Multiplying by one in the Montgomery domain converts a Montgomery
	// value out of the domain.
	one := []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
	p256OrdMul(x, x, one)

	return new(big.Int).SetBytes(x)
}

// fromBig converts a *big.Int into a format used by this code.
func fromBig(big *big.Int) []byte {
	// This could be done a lot more efficiently...
	res := big.Bytes()
	if 32 == len(res) {
		return res
	}
	t := make([]byte, 32)
	offset := 32 - len(res)
	for i := len(res) - 1; i >= 0; i-- {
		t[i+offset] = res[i]
	}
	return t
}

// p256GetMultiplier makes sure byte array will have 32 byte elements, If the scalar
// is equal or greater than the order of the group, it's reduced modulo that order.
func p256GetMultiplier(in []byte) []byte {
	n := new(big.Int).SetBytes(in)

	if n.Cmp(p256Params.N) >= 0 {
		n.Mod(n, p256Params.N)
	}
	return fromBig(n)
}

// p256MulAsm operates in a Montgomery domain with R = 2^256 mod p, where p is the
// underlying field of the curve. (See initP256 for the value.) Thus rr here is
// R×R mod p. See comment in Inverse about how this is used.
var rr = []byte{0x00, 0x00, 0x00, 0x04, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe,
	0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03}

// (This is one, in the Montgomery domain.)
var one = []byte{0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01}

func maybeReduceModP(in *big.Int) *big.Int {
	if in.Cmp(p256Params.P) < 0 {
		return in
	}
	return new(big.Int).Mod(in, p256Params.P)
}

func (curve p256CurveFast) CombinedMult(bigX, bigY *big.Int, baseScalar, scalar []byte) (x, y *big.Int) {
	var r1, r2 p256Point
	r1.p256BaseMult(p256GetMultiplier(baseScalar))

	copy(r2.x[:], fromBig(maybeReduceModP(bigX)))
	copy(r2.y[:], fromBig(maybeReduceModP(bigY)))
	copy(r2.z[:], one)
	p256MulAsm(r2.x[:], r2.x[:], rr[:])
	p256MulAsm(r2.y[:], r2.y[:], rr[:])

	r2.p256ScalarMult(p256GetMultiplier(scalar))
	p256PointAddAsm(&r1, &r1, &r2)
	return r1.p256PointToAffine()
}

func (curve p256CurveFast) ScalarBaseMult(scalar []byte) (x, y *big.Int) {
	var r p256Point
	r.p256BaseMult(p256GetMultiplier(scalar))
	return r.p256PointToAffine()
}

func (curve p256CurveFast) ScalarMult(bigX, bigY *big.Int, scalar []byte) (x, y *big.Int) {
	var r p256Point
	copy(r.x[:], fromBig(maybeReduceModP(bigX)))
	copy(r.y[:], fromBig(maybeReduceModP(bigY)))
	copy(r.z[:], one)
	p256MulAsm(r.x[:], r.x[:], rr[:])
	p256MulAsm(r.y[:], r.y[:], rr[:])
	r.p256ScalarMult(p256GetMultiplier(scalar))
	return r.p256PointToAffine()
}

func (p *p256Point) p256PointToAffine() (x, y *big.Int) {
	zInv := make([]byte, 32)
	zInvSq := make([]byte, 32)

	p256Inverse(zInv, p.z[:])
	p256Sqr(zInvSq, zInv)
	p256MulAsm(zInv, zInv, zInvSq)

	p256MulAsm(zInvSq, p.x[:], zInvSq)
	p256MulAsm(zInv, p.y[:], zInv)

	p256FromMont(zInvSq, zInvSq)
	p256FromMont(zInv, zInv)

	return new(big.Int).SetBytes(zInvSq), new(big.Int).SetBytes(zInv)
}

// p256Inverse sets out to in^-1 mod p.
func p256Inverse(out, in []byte) {
	var stack [6 * 32]byte
	p2 := stack[32*0 : 32*0+32]
	p4 := stack[32*1 : 32*1+32]
	p8 := stack[32*2 : 32*2+32]
	p16 := stack[32*3 : 32*3+32]
	p32 := stack[32*4 : 32*4+32]

	p256Sqr(out, in)
	p256MulAsm(p2, out, in) // 3*p

	p256Sqr(out, p2)
	p256Sqr(out, out)
	p256MulAsm(p4, out, p2) // f*p

	p256Sqr(out, p4)
	p256Sqr(out, out)
	p256Sqr(out, out)
	p256Sqr(out, out)
	p256MulAsm(p8, out, p4) // ff*p

	p256Sqr(out, p8)

	for i := 0; i < 7; i++ {
		p256Sqr(out, out)
	}
	p256MulAsm(p16, out, p8) // ffff*p

	p256Sqr(out, p16)
	for i := 0; i < 15; i++ {
		p256Sqr(out, out)
	}
	p256MulAsm(p32, out, p16) // ffffffff*p

	p256Sqr(out, p32)

	for i := 0; i < 31; i++ {
		p256Sqr(out, out)
	}
	p256MulAsm(out, out, in)

	for i := 0; i < 32*4; i++ {
		p256Sqr(out, out)
	}
	p256MulAsm(out, out, p32)

	for i := 0; i < 32; i++ {
		p256Sqr(out, out)
	}
	p256MulAsm(out, out, p32)

	for i := 0; i < 16; i++ {
		p256Sqr(out, out)
	}
	p256MulAsm(out, out, p16)

	for i := 0; i < 8; i++ {
		p256Sqr(out, out)
	}
	p256MulAsm(out, out, p8)

	p256Sqr(out, out)
	p256Sqr(out, out)
	p256Sqr(out, out)
	p256Sqr(out, out)
	p256MulAsm(out, out, p4)

	p256Sqr(out, out)
	p256Sqr(out, out)
	p256MulAsm(out, out, p2)

	p256Sqr(out, out)
	p256Sqr(out, out)
	p256MulAsm(out, out, in)
}

func boothW5(in uint) (int, int) {
	var s uint = ^((in >> 5) - 1)
	var d uint = (1 << 6) - in - 1
	d = (d & s) | (in & (^s))
	d = (d >> 1) + (d & 1)
	return int(d), int(s & 1)
}

func boothW7(in uint) (int, int) {
	var s uint = ^((in >> 7) - 1)
	var d uint = (1 << 8) - in - 1
	d = (d & s) | (in & (^s))
	d = (d >> 1) + (d & 1)
	return int(d), int(s & 1)
}

func initTable() {
	p256PreFast = new([37][64]p256Point) //z coordinate not used
	basePoint := p256Point{
		x: [32]byte{0x18, 0x90, 0x5f, 0x76, 0xa5, 0x37, 0x55, 0xc6, 0x79, 0xfb, 0x73, 0x2b, 0x77, 0x62, 0x25, 0x10,
			0x75, 0xba, 0x95, 0xfc, 0x5f, 0xed, 0xb6, 0x01, 0x79, 0xe7, 0x30, 0xd4, 0x18, 0xa9, 0x14, 0x3c}, //(p256.x*2^256)%p
		y: [32]byte{0x85, 0x71, 0xff, 0x18, 0x25, 0x88, 0x5d, 0x85, 0xd2, 0xe8, 0x86, 0x88, 0xdd, 0x21, 0xf3, 0x25,
			0x8b, 0x4a, 0xb8, 0xe4, 0xba, 0x19, 0xe4, 0x5c, 0xdd, 0xf2, 0x53, 0x57, 0xce, 0x95, 0x56, 0x0a}, //(p256.y*2^256)%p
		z: [32]byte{0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
			0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01}, //(p256.z*2^256)%p
	}

	t1 := new(p256Point)
	t2 := new(p256Point)
	*t2 = basePoint

	zInv := make([]byte, 32)
	zInvSq := make([]byte, 32)
	for j := 0; j < 64; j++ {
		*t1 = *t2
		for i := 0; i < 37; i++ {
			// The window size is 7 so we need to double 7 times.
			if i != 0 {
				for k := 0; k < 7; k++ {
					p256PointDoubleAsm(t1, t1)
				}
			}
			// Convert the point to affine form. (Its values are
			// still in Montgomery form however.)
			p256Inverse(zInv, t1.z[:])
			p256Sqr(zInvSq, zInv)
			p256MulAsm(zInv, zInv, zInvSq)

			p256MulAsm(t1.x[:], t1.x[:], zInvSq)
			p256MulAsm(t1.y[:], t1.y[:], zInv)

			copy(t1.z[:], basePoint.z[:])
			// Update the table entry
			copy(p256PreFast[i][j].x[:], t1.x[:])
			copy(p256PreFast[i][j].y[:], t1.y[:])
		}
		if j == 0 {
			p256PointDoubleAsm(t2, &basePoint)
		} else {
			p256PointAddAsm(t2, t2, &basePoint)
		}
	}
}

func (p *p256Point) p256BaseMult(scalar []byte) {
	wvalue := (uint(scalar[31]) << 1) & 0xff
	sel, sign := boothW7(uint(wvalue))
	p256SelectBase(p, p256PreFast[0][:], sel)
	p256NegCond(p, sign)

	copy(p.z[:], one[:])
	var t0 p256Point

	copy(t0.z[:], one[:])

	index := uint(6)
	zero := sel

	for i := 1; i < 37; i++ {
		if index < 247 {
			wvalue = ((uint(scalar[31-index/8]) >> (index % 8)) + (uint(scalar[31-index/8-1]) << (8 - (index % 8)))) & 0xff
		} else {
			wvalue = (uint(scalar[31-index/8]) >> (index % 8)) & 0xff
		}
		index += 7
		sel, sign = boothW7(uint(wvalue))
		p256SelectBase(&t0, p256PreFast[i][:], sel)
		p256PointAddAffineAsm(p, p, &t0, sign, sel, zero)
		zero |= sel
	}
}

func (p *p256Point) p256ScalarMult(scalar []byte) {
	// precomp is a table of precomputed points that stores powers of p
	// from p^1 to p^16.
	var precomp [16]p256Point
	var t0, t1, t2, t3 p256Point

	// Prepare the table
	*&precomp[0] = *p

	p256PointDoubleAsm(&t0, p)
	p256PointDoubleAsm(&t1, &t0)
	p256PointDoubleAsm(&t2, &t1)
	p256PointDoubleAsm(&t3, &t2)
	*&precomp[1] = t0  // 2
	*&precomp[3] = t1  // 4
	*&precomp[7] = t2  // 8
	*&precomp[15] = t3 // 16

	p256PointAddAsm(&t0, &t0, p)
	p256PointAddAsm(&t1, &t1, p)
	p256PointAddAsm(&t2, &t2, p)
	*&precomp[2] = t0 // 3
	*&precomp[4] = t1 // 5
	*&precomp[8] = t2 // 9

	p256PointDoubleAsm(&t0, &t0)
	p256PointDoubleAsm(&t1, &t1)
	*&precomp[5] = t0 // 6
	*&precomp[9] = t1 // 10

	p256PointAddAsm(&t2, &t0, p)
	p256PointAddAsm(&t1, &t1, p)
	*&precomp[6] = t2  // 7
	*&precomp[10] = t1 // 11

	p256PointDoubleAsm(&t0, &t0)
	p256PointDoubleAsm(&t2, &t2)
	*&precomp[11] = t0 // 12
	*&precomp[13] = t2 // 14

	p256PointAddAsm(&t0, &t0, p)
	p256PointAddAsm(&t2, &t2, p)
	*&precomp[12] = t0 // 13
	*&precomp[14] = t2 // 15

	// Start scanning the window from top bit
	index := uint(254)
	var sel, sign int

	wvalue := (uint(scalar[31-index/8]) >> (index % 8)) & 0x3f
	sel, _ = boothW5(uint(wvalue))
	p256Select(p, precomp[:], sel)
	zero := sel

	for index > 4 {
		index -= 5
		p256PointDoubleAsm(p, p)
		p256PointDoubleAsm(p, p)
		p256PointDoubleAsm(p, p)
		p256PointDoubleAsm(p, p)
		p256PointDoubleAsm(p, p)

		if index < 247 {
			wvalue = ((uint(scalar[31-index/8]) >> (index % 8)) + (uint(scalar[31-index/8-1]) << (8 - (index % 8)))) & 0x3f
		} else {
			wvalue = (uint(scalar[31-index/8]) >> (index % 8)) & 0x3f
		}

		sel, sign = boothW5(uint(wvalue))

		p256Select(&t0, precomp[:], sel)
		p256NegCond(&t0, sign)
		p256PointAddAsm(&t1, p, &t0)
		p256MovCond(&t1, &t1, p, sel)
		p256MovCond(p, &t1, &t0, zero)
		zero |= sel
	}

	p256PointDoubleAsm(p, p)
	p256PointDoubleAsm(p, p)
	p256PointDoubleAsm(p, p)
	p256PointDoubleAsm(p, p)
	p256PointDoubleAsm(p, p)

	wvalue = (uint(scalar[31]) << 1) & 0x3f
	sel, sign = boothW5(uint(wvalue))

	p256Select(&t0, precomp[:], sel)
	p256NegCond(&t0, sign)
	p256PointAddAsm(&t1, p, &t0)
	p256MovCond(&t1, &t1, p, sel)
	p256MovCond(p, &t1, &t0, zero)
}

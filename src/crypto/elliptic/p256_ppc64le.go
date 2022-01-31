// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64le

package elliptic

import (
	"crypto/subtle"
	"encoding/binary"
	"math/big"
)

// This was ported from the s390x implementation for ppc64le.
// Some hints are included here for changes that should be
// in the big endian ppc64 implementation, however more
// investigation and testing is needed for the ppc64 big
// endian version to work.
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

func initP256Arch() {
	p256 = p256CurveFast{p256Params}
	initTable()
	return
}

func (curve p256CurveFast) Params() *CurveParams {
	return curve.CurveParams
}

// Functions implemented in p256_asm_ppc64le.s
// Montgomery multiplication modulo P256
//
//go:noescape
func p256MulAsm(res, in1, in2 []byte)

// Montgomery square modulo P256
func p256Sqr(res, in []byte) {
	p256MulAsm(res, in, in)
}

// Montgomery multiplication by 1
//
//go:noescape
func p256FromMont(res, in []byte)

// iff cond == 1  val <- -val
//
//go:noescape
func p256NegCond(val *p256Point, cond int)

// if cond == 0 res <- b; else res <- a
//
//go:noescape
func p256MovCond(res, a, b *p256Point, cond int)

// Constant time table access
//
//go:noescape
func p256Select(point *p256Point, table []p256Point, idx int)

//
//go:noescape
func p256SelectBase(point *p256Point, table []p256Point, idx int)

// Point add with P2 being affine point
// If sign == 1 -> P2 = -P2
// If sel == 0 -> P3 = P1
// if zero == 0 -> P3 = P2
//
//go:noescape
func p256PointAddAffineAsm(res, in1, in2 *p256Point, sign, sel, zero int)

// Point add
//
//go:noescape
func p256PointAddAsm(res, in1, in2 *p256Point) int

//
//go:noescape
func p256PointDoubleAsm(res, in *p256Point)

// The result should be a slice in LE order, but the slice
// from big.Bytes is in BE order.
// TODO: For big endian implementation, do not reverse bytes.
func fromBig(big *big.Int) []byte {
	// This could be done a lot more efficiently...
	res := big.Bytes()
	t := make([]byte, 32)
	if len(res) < 32 {
		copy(t[32-len(res):], res)
	} else if len(res) == 32 {
		copy(t, res)
	} else {
		copy(t, res[len(res)-32:])
	}
	p256ReverseBytes(t, t)
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
// TODO: For big endian implementation, the bytes in these slices should be in reverse order,
// as found in the s390x implementation.
var rr = []byte{0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0, 0xff, 0xff, 0xff, 0xff, 0xfb, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00}

// (This is one, in the Montgomery domain.)
var one = []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00}

func maybeReduceModP(in *big.Int) *big.Int {
	if in.Cmp(p256Params.P) < 0 {
		return in
	}
	return new(big.Int).Mod(in, p256Params.P)
}

// p256ReverseBytes copies the first 32 bytes from in to res in reverse order.
func p256ReverseBytes(res, in []byte) {
	// remove bounds check
	in = in[:32]
	res = res[:32]

	// Load in reverse order
	a := binary.BigEndian.Uint64(in[0:])
	b := binary.BigEndian.Uint64(in[8:])
	c := binary.BigEndian.Uint64(in[16:])
	d := binary.BigEndian.Uint64(in[24:])

	// Store in normal order
	binary.LittleEndian.PutUint64(res[0:], d)
	binary.LittleEndian.PutUint64(res[8:], c)
	binary.LittleEndian.PutUint64(res[16:], b)
	binary.LittleEndian.PutUint64(res[24:], a)
}

func (curve p256CurveFast) CombinedMult(bigX, bigY *big.Int, baseScalar, scalar []byte) (x, y *big.Int) {
	var r1, r2 p256Point

	scalarReduced := p256GetMultiplier(baseScalar)
	r1IsInfinity := scalarIsZero(scalarReduced)
	r1.p256BaseMult(scalarReduced)

	copy(r2.x[:], fromBig(maybeReduceModP(bigX)))
	copy(r2.y[:], fromBig(maybeReduceModP(bigY)))
	copy(r2.z[:], one)
	p256MulAsm(r2.x[:], r2.x[:], rr[:])
	p256MulAsm(r2.y[:], r2.y[:], rr[:])

	scalarReduced = p256GetMultiplier(scalar)
	r2IsInfinity := scalarIsZero(scalarReduced)
	r2.p256ScalarMult(scalarReduced)

	var sum, double p256Point
	pointsEqual := p256PointAddAsm(&sum, &r1, &r2)
	p256PointDoubleAsm(&double, &r1)
	p256MovCond(&sum, &double, &sum, pointsEqual)
	p256MovCond(&sum, &r1, &sum, r2IsInfinity)
	p256MovCond(&sum, &r2, &sum, r1IsInfinity)
	return sum.p256PointToAffine()
}

func (curve p256CurveFast) ScalarBaseMult(scalar []byte) (x, y *big.Int) {
	var r p256Point
	reducedScalar := p256GetMultiplier(scalar)
	r.p256BaseMult(reducedScalar)
	return r.p256PointToAffine()
}

func (curve p256CurveFast) ScalarMult(bigX, bigY *big.Int, scalar []byte) (x, y *big.Int) {
	scalarReduced := p256GetMultiplier(scalar)
	var r p256Point
	copy(r.x[:], fromBig(maybeReduceModP(bigX)))
	copy(r.y[:], fromBig(maybeReduceModP(bigY)))
	copy(r.z[:], one)
	p256MulAsm(r.x[:], r.x[:], rr[:])
	p256MulAsm(r.y[:], r.y[:], rr[:])
	r.p256ScalarMult(scalarReduced)
	return r.p256PointToAffine()
}

func scalarIsZero(scalar []byte) int {
	// If any byte is not zero, return 0.
	// Check for -0.... since that appears to compare to 0.
	b := byte(0)
	for _, s := range scalar {
		b |= s
	}
	return subtle.ConstantTimeByteEq(b, 0)
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

	// SetBytes expects a slice in big endian order,
	// since ppc64le is little endian, reverse the bytes.
	// TODO: For big endian, bytes don't need to be reversed.
	p256ReverseBytes(zInvSq, zInvSq)
	p256ReverseBytes(zInv, zInv)
	rx := new(big.Int).SetBytes(zInvSq)
	ry := new(big.Int).SetBytes(zInv)
	return rx, ry
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

func boothW6(in uint) (int, int) {
	var s uint = ^((in >> 6) - 1)
	var d uint = (1 << 7) - in - 1
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

	p256PreFast = new([37][64]p256Point)

	// TODO: For big endian, these slices should be in reverse byte order,
	// as found in the s390x implementation.
	basePoint := p256Point{
		x: [32]byte{0x3c, 0x14, 0xa9, 0x18, 0xd4, 0x30, 0xe7, 0x79, 0x01, 0xb6, 0xed, 0x5f, 0xfc, 0x95, 0xba, 0x75,
			0x10, 0x25, 0x62, 0x77, 0x2b, 0x73, 0xfb, 0x79, 0xc6, 0x55, 0x37, 0xa5, 0x76, 0x5f, 0x90, 0x18}, //(p256.x*2^256)%p
		y: [32]byte{0x0a, 0x56, 0x95, 0xce, 0x57, 0x53, 0xf2, 0xdd, 0x5c, 0xe4, 0x19, 0xba, 0xe4, 0xb8, 0x4a, 0x8b,
			0x25, 0xf3, 0x21, 0xdd, 0x88, 0x86, 0xe8, 0xd2, 0x85, 0x5d, 0x88, 0x25, 0x18, 0xff, 0x71, 0x85}, //(p256.y*2^256)%p
		z: [32]byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
			0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00}, //(p256.z*2^256)%p

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
	// TODO: For big endian, the index should be 31 not 0.
	wvalue := (uint(scalar[0]) << 1) & 0xff
	sel, sign := boothW7(uint(wvalue))
	p256SelectBase(p, p256PreFast[0][:], sel)
	p256NegCond(p, sign)

	copy(p.z[:], one[:])
	var t0 p256Point

	copy(t0.z[:], one[:])

	index := uint(6)
	zero := sel
	for i := 1; i < 37; i++ {
		// TODO: For big endian, use the same index values as found
		// in the  s390x implementation.
		if index < 247 {
			wvalue = ((uint(scalar[index/8]) >> (index % 8)) + (uint(scalar[index/8+1]) << (8 - (index % 8)))) & 0xff
		} else {
			wvalue = (uint(scalar[index/8]) >> (index % 8)) & 0xff
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

	*&precomp[0] = *p
	p256PointDoubleAsm(&t0, p)
	p256PointDoubleAsm(&t1, &t0)
	p256PointDoubleAsm(&t2, &t1)
	p256PointDoubleAsm(&t3, &t2)
	*&precomp[1] = t0
	*&precomp[3] = t1
	*&precomp[7] = t2
	*&precomp[15] = t3

	p256PointAddAsm(&t0, &t0, p)
	p256PointAddAsm(&t1, &t1, p)
	p256PointAddAsm(&t2, &t2, p)

	*&precomp[2] = t0
	*&precomp[4] = t1
	*&precomp[8] = t2

	p256PointDoubleAsm(&t0, &t0)
	p256PointDoubleAsm(&t1, &t1)
	*&precomp[5] = t0
	*&precomp[9] = t1

	p256PointAddAsm(&t2, &t0, p)
	p256PointAddAsm(&t1, &t1, p)
	*&precomp[6] = t2
	*&precomp[10] = t1

	p256PointDoubleAsm(&t0, &t0)
	p256PointDoubleAsm(&t2, &t2)
	*&precomp[11] = t0
	*&precomp[13] = t2

	p256PointAddAsm(&t0, &t0, p)
	p256PointAddAsm(&t2, &t2, p)
	*&precomp[12] = t0
	*&precomp[14] = t2

	// Start scanning the window from top bit
	index := uint(254)
	var sel, sign int

	// TODO: For big endian, use index found in s390x implementation.
	wvalue := (uint(scalar[index/8]) >> (index % 8)) & 0x3f
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

		// TODO: For big endian, use index values as found in s390x implementation.
		if index < 247 {
			wvalue = ((uint(scalar[index/8]) >> (index % 8)) + (uint(scalar[index/8+1]) << (8 - (index % 8)))) & 0x3f
		} else {
			wvalue = (uint(scalar[index/8]) >> (index % 8)) & 0x3f
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

	// TODO: Use index for big endian as found in s390x implementation.
	wvalue = (uint(scalar[0]) << 1) & 0x3f
	sel, sign = boothW5(uint(wvalue))

	p256Select(&t0, precomp[:], sel)
	p256NegCond(&t0, sign)
	p256PointAddAsm(&t1, p, &t0)
	p256MovCond(&t1, &t1, p, sel)
	p256MovCond(p, &t1, &t0, zero)
}

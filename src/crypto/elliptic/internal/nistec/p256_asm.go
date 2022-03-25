// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the Go wrapper for the constant-time, 64-bit assembly
// implementation of P256. The optimizations performed here are described in
// detail in:
// S.Gueron and V.Krasnov, "Fast prime field elliptic-curve cryptography with
//                          256-bit primes"
// https://link.springer.com/article/10.1007%2Fs13389-014-0090-x
// https://eprint.iacr.org/2013/816.pdf

//go:build amd64 || arm64

package nistec

import (
	_ "embed"
	"errors"
	"math/bits"
)

//go:embed p256_asm_table.bin
var p256Precomputed string

// P256Point is a P-256 point. The zero value is NOT valid.
type P256Point struct {
	xyz [12]uint64
}

// NewP256Point returns a new P256Point representing the point at infinity point.
func NewP256Point() *P256Point {
	return &P256Point{[12]uint64{
		0x0000000000000001, 0xffffffff00000000, 0xffffffffffffffff, 0x00000000fffffffe,
		0x0000000000000001, 0xffffffff00000000, 0xffffffffffffffff, 0x00000000fffffffe,
		0, 0, 0, 0,
	}}
}

// NewP256Generator returns a new P256Point set to the canonical generator.
func NewP256Generator() *P256Point {
	return &P256Point{[12]uint64{
		0x79e730d418a9143c, 0x75ba95fc5fedb601, 0x79fb732b77622510, 0x18905f76a53755c6,
		0xddf25357ce95560a, 0x8b4ab8e4ba19e45c, 0xd2e88688dd21f325, 0x8571ff1825885d85,
		0x0000000000000001, 0xffffffff00000000, 0xffffffffffffffff, 0x00000000fffffffe,
	}}
}

// Set sets p = q and returns p.
func (p *P256Point) Set(q *P256Point) *P256Point {
	p.xyz = q.xyz
	return p
}

const p256ElementLength = 32
const p256UncompressedLength = 1 + 2*p256ElementLength
const p256CompressedLength = 1 + p256ElementLength

// SetBytes sets p to the compressed, uncompressed, or infinity value encoded in
// b, as specified in SEC 1, Version 2.0, Section 2.3.4. If the point is not on
// the curve, it returns nil and an error, and the receiver is unchanged.
// Otherwise, it returns p.
func (p *P256Point) SetBytes(b []byte) (*P256Point, error) {
	switch {
	// Point at infinity.
	case len(b) == 1 && b[0] == 0:
		return p.Set(NewP256Point()), nil

	// Uncompressed form.
	case len(b) == p256UncompressedLength && b[0] == 4:
		var r P256Point
		p256BigToLittle(r.xyz[0:4], b[1:33])
		p256BigToLittle(r.xyz[4:8], b[33:65])
		if p256LessThanP(r.xyz[0:4]) == 0 || p256LessThanP(r.xyz[4:8]) == 0 {
			return nil, errors.New("invalid P256 element encoding")
		}
		p256Mul(r.xyz[0:4], r.xyz[0:4], rr[:])
		p256Mul(r.xyz[4:8], r.xyz[4:8], rr[:])
		if err := p256CheckOnCurve(r.xyz[0:4], r.xyz[4:8]); err != nil {
			return nil, err
		}
		// This sets r's Z value to 1, in the Montgomery domain.
		r.xyz[8] = 0x0000000000000001
		r.xyz[9] = 0xffffffff00000000
		r.xyz[10] = 0xffffffffffffffff
		r.xyz[11] = 0x00000000fffffffe
		return p.Set(&r), nil

	// Compressed form.
	case len(b) == p256CompressedLength && (b[0] == 2 || b[0] == 3):
		return nil, errors.New("unimplemented") // TODO(filippo)

	default:
		return nil, errors.New("invalid P256 point encoding")
	}
}

func p256CheckOnCurve(x, y []uint64) error {
	// x³ - 3x + b
	x3 := make([]uint64, 4)
	p256Sqr(x3, x, 1)
	p256Mul(x3, x3, x)

	threeX := make([]uint64, 4)
	p256Add(threeX, x, x)
	p256Add(threeX, threeX, x)
	p256NegCond(threeX, 1)

	p256B := []uint64{0xd89cdf6229c4bddf, 0xacf005cd78843090,
		0xe5a220abf7212ed6, 0xdc30061d04874834}

	p256Add(x3, x3, threeX)
	p256Add(x3, x3, p256B)

	// y² = x³ - 3x + b
	y2 := make([]uint64, 4)
	p256Sqr(y2, y, 1)

	diff := (x3[0] ^ y2[0]) | (x3[1] ^ y2[1]) |
		(x3[2] ^ y2[2]) | (x3[3] ^ y2[3])
	if uint64IsZero(diff) != 1 {
		return errors.New("P256 point not on curve")
	}
	return nil
}

var p256P = []uint64{0xffffffffffffffff, 0x00000000ffffffff,
	0x0000000000000000, 0xffffffff00000001}

// p256LessThanP returns 1 if x < p, and 0 otherwise.
func p256LessThanP(x []uint64) int {
	var b uint64
	_, b = bits.Sub64(x[0], p256P[0], b)
	_, b = bits.Sub64(x[1], p256P[1], b)
	_, b = bits.Sub64(x[2], p256P[2], b)
	_, b = bits.Sub64(x[3], p256P[3], b)
	return int(b)
}

func p256Add(res, x, y []uint64) {
	var c, b uint64
	t1 := make([]uint64, 4)
	t1[0], c = bits.Add64(x[0], y[0], 0)
	t1[1], c = bits.Add64(x[1], y[1], c)
	t1[2], c = bits.Add64(x[2], y[2], c)
	t1[3], c = bits.Add64(x[3], y[3], c)
	t2 := make([]uint64, 4)
	t2[0], b = bits.Sub64(t1[0], p256P[0], 0)
	t2[1], b = bits.Sub64(t1[1], p256P[1], b)
	t2[2], b = bits.Sub64(t1[2], p256P[2], b)
	t2[3], b = bits.Sub64(t1[3], p256P[3], b)
	// Three options:
	//   - a+b < p
	//     then c is 0, b is 1, and t1 is correct
	//   - p <= a+b < 2^256
	//     then c is 0, b is 0, and t2 is correct
	//   - 2^256 <= a+b
	//     then c is 1, b is 1, and t2 is correct
	t2Mask := (c ^ b) - 1
	res[0] = (t1[0] & ^t2Mask) | (t2[0] & t2Mask)
	res[1] = (t1[1] & ^t2Mask) | (t2[1] & t2Mask)
	res[2] = (t1[2] & ^t2Mask) | (t2[2] & t2Mask)
	res[3] = (t1[3] & ^t2Mask) | (t2[3] & t2Mask)
}

// Functions implemented in p256_asm_*64.s
// Montgomery multiplication modulo P256
//
//go:noescape
func p256Mul(res, in1, in2 []uint64)

// Montgomery square modulo P256, repeated n times (n >= 1)
//
//go:noescape
func p256Sqr(res, in []uint64, n int)

// Montgomery multiplication by 1
//
//go:noescape
func p256FromMont(res, in []uint64)

// iff cond == 1  val <- -val
//
//go:noescape
func p256NegCond(val []uint64, cond int)

// if cond == 0 res <- b; else res <- a
//
//go:noescape
func p256MovCond(res, a, b []uint64, cond int)

// Endianness swap
//
//go:noescape
func p256BigToLittle(res []uint64, in []byte)

//go:noescape
func p256LittleToBig(res []byte, in []uint64)

// Constant time table access
//
//go:noescape
func p256Select(point, table []uint64, idx int)

//go:noescape
func p256SelectBase(point *[12]uint64, table string, idx int)

// Montgomery multiplication modulo Ord(G)
//
//go:noescape
func p256OrdMul(res, in1, in2 []uint64)

// Montgomery square modulo Ord(G), repeated n times
//
//go:noescape
func p256OrdSqr(res, in []uint64, n int)

// Point add with in2 being affine point
// If sign == 1 -> in2 = -in2
// If sel == 0 -> res = in1
// if zero == 0 -> res = in2
//
//go:noescape
func p256PointAddAffineAsm(res, in1, in2 []uint64, sign, sel, zero int)

// Point add. Returns one if the two input points were equal and zero
// otherwise. (Note that, due to the way that the equations work out, some
// representations of ∞ are considered equal to everything by this function.)
//
//go:noescape
func p256PointAddAsm(res, in1, in2 []uint64) int

// Point double
//
//go:noescape
func p256PointDoubleAsm(res, in []uint64)

func P256OrdInverse(k []byte) ([]byte, error) {
	// TODO: test for values p <= x < 2^256.
	if len(k) != 32 {
		return nil, errors.New("invalid scalar length")
	}

	// table will store precomputed powers of x.
	var table [4 * 9]uint64
	var (
		_1      = table[4*0 : 4*1]
		_11     = table[4*1 : 4*2]
		_101    = table[4*2 : 4*3]
		_111    = table[4*3 : 4*4]
		_1111   = table[4*4 : 4*5]
		_10101  = table[4*5 : 4*6]
		_101111 = table[4*6 : 4*7]
		x       = table[4*7 : 4*8]
		t       = table[4*8 : 4*9]
	)

	p256BigToLittle(x, k)
	// This code operates in the Montgomery domain where R = 2^256 mod n
	// and n is the order of the scalar field. (See initP256 for the
	// value.) Elements in the Montgomery domain take the form a×R and
	// multiplication of x and y in the calculates (x × y × R^-1) mod n. RR
	// is R×R mod n thus the Montgomery multiplication x and RR gives x×R,
	// i.e. converts x into the Montgomery domain.
	// Window values borrowed from https://briansmith.org/ecc-inversion-addition-chains-01#p256_scalar_inversion
	RR := []uint64{0x83244c95be79eea2, 0x4699799c49bd6fa6, 0x2845b2392b6bec59, 0x66e12d94f3d95620}
	p256OrdMul(_1, x, RR)      // _1
	p256OrdSqr(x, _1, 1)       // _10
	p256OrdMul(_11, x, _1)     // _11
	p256OrdMul(_101, x, _11)   // _101
	p256OrdMul(_111, x, _101)  // _111
	p256OrdSqr(x, _101, 1)     // _1010
	p256OrdMul(_1111, _101, x) // _1111

	p256OrdSqr(t, x, 1)          // _10100
	p256OrdMul(_10101, t, _1)    // _10101
	p256OrdSqr(x, _10101, 1)     // _101010
	p256OrdMul(_101111, _101, x) // _101111
	p256OrdMul(x, _10101, x)     // _111111 = x6
	p256OrdSqr(t, x, 2)          // _11111100
	p256OrdMul(t, t, _11)        // _11111111 = x8
	p256OrdSqr(x, t, 8)          // _ff00
	p256OrdMul(x, x, t)          // _ffff = x16
	p256OrdSqr(t, x, 16)         // _ffff0000
	p256OrdMul(t, t, x)          // _ffffffff = x32

	p256OrdSqr(x, t, 64)
	p256OrdMul(x, x, t)
	p256OrdSqr(x, x, 32)
	p256OrdMul(x, x, t)

	sqrs := []uint8{
		6, 5, 4, 5, 5,
		4, 3, 3, 5, 9,
		6, 2, 5, 6, 5,
		4, 5, 5, 3, 10,
		2, 5, 5, 3, 7, 6}
	muls := [][]uint64{
		_101111, _111, _11, _1111, _10101,
		_101, _101, _101, _111, _101111,
		_1111, _1, _1, _1111, _111,
		_111, _111, _101, _11, _101111,
		_11, _11, _11, _1, _10101, _1111}

	for i, s := range sqrs {
		p256OrdSqr(x, x, int(s))
		p256OrdMul(x, x, muls[i])
	}

	// Multiplying by one in the Montgomery domain converts a Montgomery
	// value out of the domain.
	one := []uint64{1, 0, 0, 0}
	p256OrdMul(x, x, one)

	xOut := make([]byte, 32)
	p256LittleToBig(xOut, x)
	return xOut, nil
}

// p256Mul operates in a Montgomery domain with R = 2^256 mod p, where p is the
// underlying field of the curve. (See initP256 for the value.) Thus rr here is
// R×R mod p. See comment in Inverse about how this is used.
var rr = []uint64{0x0000000000000003, 0xfffffffbffffffff, 0xfffffffffffffffe, 0x00000004fffffffd}

// Add sets q = p1 + p2, and returns q. The points may overlap.
func (q *P256Point) Add(r1, r2 *P256Point) *P256Point {
	var sum, double P256Point
	r1IsInfinity := r1.isInfinity()
	r2IsInfinity := r2.isInfinity()
	pointsEqual := p256PointAddAsm(sum.xyz[:], r1.xyz[:], r2.xyz[:])
	p256PointDoubleAsm(double.xyz[:], r1.xyz[:])
	sum.Select(&double, &sum, pointsEqual)
	sum.Select(r1, &sum, r2IsInfinity)
	sum.Select(r2, &sum, r1IsInfinity)
	return q.Set(&sum)
}

// Double sets q = p + p, and returns q. The points may overlap.
func (q *P256Point) Double(p *P256Point) *P256Point {
	var double P256Point
	p256PointDoubleAsm(double.xyz[:], p.xyz[:])
	return q.Set(&double)
}

// ScalarBaseMult sets r = scalar * generator, where scalar is a 32-byte big
// endian value, and returns r. If scalar is not 32 bytes long, ScalarBaseMult
// returns an error and the receiver is unchanged.
func (r *P256Point) ScalarBaseMult(scalar []byte) (*P256Point, error) {
	// TODO: test for values p <= x < 2^256.
	if len(scalar) != 32 {
		return nil, errors.New("invalid scalar length")
	}
	scalarReversed := make([]uint64, 4)
	p256BigToLittle(scalarReversed, scalar)

	r.p256BaseMult(scalarReversed)
	return r, nil
}

// ScalarMult sets r = scalar * q, where scalar is a 32-byte big endian value,
// and returns r. If scalar is not 32 bytes long, ScalarBaseMult returns an
// error and the receiver is unchanged.
func (r *P256Point) ScalarMult(q *P256Point, scalar []byte) (*P256Point, error) {
	// TODO: test for values p <= x < 2^256.
	if len(scalar) != 32 {
		return nil, errors.New("invalid scalar length")
	}
	scalarReversed := make([]uint64, 4)
	p256BigToLittle(scalarReversed, scalar)

	r.Set(q).p256ScalarMult(scalarReversed)
	return r, nil
}

// uint64IsZero returns 1 if x is zero and zero otherwise.
func uint64IsZero(x uint64) int {
	x = ^x
	x &= x >> 32
	x &= x >> 16
	x &= x >> 8
	x &= x >> 4
	x &= x >> 2
	x &= x >> 1
	return int(x & 1)
}

// isInfinity returns 1 if p is the point at infinity and 0 otherwise.
func (p *P256Point) isInfinity() int {
	return uint64IsZero(p.xyz[8] | p.xyz[9] | p.xyz[10] | p.xyz[11])
}

// Bytes returns the uncompressed or infinity encoding of p, as specified in
// SEC 1, Version 2.0, Section 2.3.3. Note that the encoding of the point at
// infinity is shorter than all other encodings.
func (p *P256Point) Bytes() []byte {
	// This function is outlined to make the allocations inline in the caller
	// rather than happen on the heap.
	var out [p256UncompressedLength]byte
	return p.bytes(&out)
}

func (p *P256Point) bytes(out *[p256UncompressedLength]byte) []byte {
	// The proper representation of the point at infinity is a single zero byte.
	if p.isInfinity() == 1 {
		return out[:1]
	}

	zInv := make([]uint64, 4)
	zInvSq := make([]uint64, 4)
	p256Inverse(zInv, p.xyz[8:12])
	p256Sqr(zInvSq, zInv, 1)
	p256Mul(zInv, zInv, zInvSq)

	p256Mul(zInvSq, p.xyz[0:4], zInvSq)
	p256Mul(zInv, p.xyz[4:8], zInv)

	p256FromMont(zInvSq, zInvSq)
	p256FromMont(zInv, zInv)

	out[0] = 4 // Uncompressed form.
	p256LittleToBig(out[1:33], zInvSq)
	p256LittleToBig(out[33:65], zInv)

	return out[:]
}

// Select sets q to p1 if cond == 1, and to p2 if cond == 0.
func (q *P256Point) Select(p1, p2 *P256Point, cond int) *P256Point {
	p256MovCond(q.xyz[:], p1.xyz[:], p2.xyz[:], cond)
	return q
}

// p256Inverse sets out to in^-1 mod p.
func p256Inverse(out, in []uint64) {
	var stack [6 * 4]uint64
	p2 := stack[4*0 : 4*0+4]
	p4 := stack[4*1 : 4*1+4]
	p8 := stack[4*2 : 4*2+4]
	p16 := stack[4*3 : 4*3+4]
	p32 := stack[4*4 : 4*4+4]

	p256Sqr(out, in, 1)
	p256Mul(p2, out, in) // 3*p

	p256Sqr(out, p2, 2)
	p256Mul(p4, out, p2) // f*p

	p256Sqr(out, p4, 4)
	p256Mul(p8, out, p4) // ff*p

	p256Sqr(out, p8, 8)
	p256Mul(p16, out, p8) // ffff*p

	p256Sqr(out, p16, 16)
	p256Mul(p32, out, p16) // ffffffff*p

	p256Sqr(out, p32, 32)
	p256Mul(out, out, in)

	p256Sqr(out, out, 128)
	p256Mul(out, out, p32)

	p256Sqr(out, out, 32)
	p256Mul(out, out, p32)

	p256Sqr(out, out, 16)
	p256Mul(out, out, p16)

	p256Sqr(out, out, 8)
	p256Mul(out, out, p8)

	p256Sqr(out, out, 4)
	p256Mul(out, out, p4)

	p256Sqr(out, out, 2)
	p256Mul(out, out, p2)

	p256Sqr(out, out, 2)
	p256Mul(out, out, in)
}

func (p *P256Point) p256StorePoint(r *[16 * 4 * 3]uint64, index int) {
	copy(r[index*12:], p.xyz[:])
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

func (p *P256Point) p256BaseMult(scalar []uint64) {
	wvalue := (scalar[0] << 1) & 0x7f
	sel, sign := boothW6(uint(wvalue))
	p256SelectBase(&p.xyz, p256Precomputed, sel)
	p256NegCond(p.xyz[4:8], sign)

	// (This is one, in the Montgomery domain.)
	p.xyz[8] = 0x0000000000000001
	p.xyz[9] = 0xffffffff00000000
	p.xyz[10] = 0xffffffffffffffff
	p.xyz[11] = 0x00000000fffffffe

	var t0 P256Point
	// (This is one, in the Montgomery domain.)
	t0.xyz[8] = 0x0000000000000001
	t0.xyz[9] = 0xffffffff00000000
	t0.xyz[10] = 0xffffffffffffffff
	t0.xyz[11] = 0x00000000fffffffe

	index := uint(5)
	zero := sel

	for i := 1; i < 43; i++ {
		if index < 192 {
			wvalue = ((scalar[index/64] >> (index % 64)) + (scalar[index/64+1] << (64 - (index % 64)))) & 0x7f
		} else {
			wvalue = (scalar[index/64] >> (index % 64)) & 0x7f
		}
		index += 6
		sel, sign = boothW6(uint(wvalue))
		p256SelectBase(&t0.xyz, p256Precomputed[i*32*8*8:], sel)
		p256PointAddAffineAsm(p.xyz[0:12], p.xyz[0:12], t0.xyz[0:8], sign, sel, zero)
		zero |= sel
	}

	// If the whole scalar was zero, set to the point at infinity.
	p256MovCond(p.xyz[:], NewP256Point().xyz[:], p.xyz[:], uint64IsZero(uint64(zero)))
}

func (p *P256Point) p256ScalarMult(scalar []uint64) {
	// precomp is a table of precomputed points that stores powers of p
	// from p^1 to p^16.
	var precomp [16 * 4 * 3]uint64
	var t0, t1, t2, t3 P256Point

	// Prepare the table
	p.p256StorePoint(&precomp, 0) // 1

	p256PointDoubleAsm(t0.xyz[:], p.xyz[:])
	p256PointDoubleAsm(t1.xyz[:], t0.xyz[:])
	p256PointDoubleAsm(t2.xyz[:], t1.xyz[:])
	p256PointDoubleAsm(t3.xyz[:], t2.xyz[:])
	t0.p256StorePoint(&precomp, 1)  // 2
	t1.p256StorePoint(&precomp, 3)  // 4
	t2.p256StorePoint(&precomp, 7)  // 8
	t3.p256StorePoint(&precomp, 15) // 16

	p256PointAddAsm(t0.xyz[:], t0.xyz[:], p.xyz[:])
	p256PointAddAsm(t1.xyz[:], t1.xyz[:], p.xyz[:])
	p256PointAddAsm(t2.xyz[:], t2.xyz[:], p.xyz[:])
	t0.p256StorePoint(&precomp, 2) // 3
	t1.p256StorePoint(&precomp, 4) // 5
	t2.p256StorePoint(&precomp, 8) // 9

	p256PointDoubleAsm(t0.xyz[:], t0.xyz[:])
	p256PointDoubleAsm(t1.xyz[:], t1.xyz[:])
	t0.p256StorePoint(&precomp, 5) // 6
	t1.p256StorePoint(&precomp, 9) // 10

	p256PointAddAsm(t2.xyz[:], t0.xyz[:], p.xyz[:])
	p256PointAddAsm(t1.xyz[:], t1.xyz[:], p.xyz[:])
	t2.p256StorePoint(&precomp, 6)  // 7
	t1.p256StorePoint(&precomp, 10) // 11

	p256PointDoubleAsm(t0.xyz[:], t0.xyz[:])
	p256PointDoubleAsm(t2.xyz[:], t2.xyz[:])
	t0.p256StorePoint(&precomp, 11) // 12
	t2.p256StorePoint(&precomp, 13) // 14

	p256PointAddAsm(t0.xyz[:], t0.xyz[:], p.xyz[:])
	p256PointAddAsm(t2.xyz[:], t2.xyz[:], p.xyz[:])
	t0.p256StorePoint(&precomp, 12) // 13
	t2.p256StorePoint(&precomp, 14) // 15

	// Start scanning the window from top bit
	index := uint(254)
	var sel, sign int

	wvalue := (scalar[index/64] >> (index % 64)) & 0x3f
	sel, _ = boothW5(uint(wvalue))

	p256Select(p.xyz[0:12], precomp[0:], sel)
	zero := sel

	for index > 4 {
		index -= 5
		p256PointDoubleAsm(p.xyz[:], p.xyz[:])
		p256PointDoubleAsm(p.xyz[:], p.xyz[:])
		p256PointDoubleAsm(p.xyz[:], p.xyz[:])
		p256PointDoubleAsm(p.xyz[:], p.xyz[:])
		p256PointDoubleAsm(p.xyz[:], p.xyz[:])

		if index < 192 {
			wvalue = ((scalar[index/64] >> (index % 64)) + (scalar[index/64+1] << (64 - (index % 64)))) & 0x3f
		} else {
			wvalue = (scalar[index/64] >> (index % 64)) & 0x3f
		}

		sel, sign = boothW5(uint(wvalue))

		p256Select(t0.xyz[0:], precomp[0:], sel)
		p256NegCond(t0.xyz[4:8], sign)
		p256PointAddAsm(t1.xyz[:], p.xyz[:], t0.xyz[:])
		p256MovCond(t1.xyz[0:12], t1.xyz[0:12], p.xyz[0:12], sel)
		p256MovCond(p.xyz[0:12], t1.xyz[0:12], t0.xyz[0:12], zero)
		zero |= sel
	}

	p256PointDoubleAsm(p.xyz[:], p.xyz[:])
	p256PointDoubleAsm(p.xyz[:], p.xyz[:])
	p256PointDoubleAsm(p.xyz[:], p.xyz[:])
	p256PointDoubleAsm(p.xyz[:], p.xyz[:])
	p256PointDoubleAsm(p.xyz[:], p.xyz[:])

	wvalue = (scalar[0] << 1) & 0x3f
	sel, sign = boothW5(uint(wvalue))

	p256Select(t0.xyz[0:], precomp[0:], sel)
	p256NegCond(t0.xyz[4:8], sign)
	p256PointAddAsm(t1.xyz[:], p.xyz[:], t0.xyz[:])
	p256MovCond(t1.xyz[0:12], t1.xyz[0:12], p.xyz[0:12], sel)
	p256MovCond(p.xyz[0:12], t1.xyz[0:12], t0.xyz[0:12], zero)
}

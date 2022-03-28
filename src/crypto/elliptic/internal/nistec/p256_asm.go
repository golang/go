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
	"unsafe"
)

// p256Element is a P-256 base field element in [0, P-1] in the Montgomery
// domain (with R 2²⁵⁶) as four limbs in little-endian order value.
type p256Element [4]uint64

// p256One is one in the Montgomery domain.
var p256One = p256Element{0x0000000000000001, 0xffffffff00000000,
	0xffffffffffffffff, 0x00000000fffffffe}

var p256Zero = p256Element{}

// p256P is 2²⁵⁶ - 2²²⁴ + 2¹⁹² + 2⁹⁶ - 1 in the Montgomery domain.
var p256P = p256Element{0xffffffffffffffff, 0x00000000ffffffff,
	0x0000000000000000, 0xffffffff00000001}

// P256Point is a P-256 point. The zero value should not be assumed to be valid
// (although it is in this implementation).
type P256Point struct {
	// (X:Y:Z) are Jacobian coordinates where x = X/Z² and y = Y/Z³. The point
	// at infinity can be represented by any set of coordinates with Z = 0.
	x, y, z p256Element
}

// NewP256Point returns a new P256Point representing the point at infinity.
func NewP256Point() *P256Point {
	return &P256Point{
		x: p256One, y: p256One, z: p256Zero,
	}
}

// NewP256Generator returns a new P256Point set to the canonical generator.
func NewP256Generator() *P256Point {
	return &P256Point{
		x: p256Element{0x79e730d418a9143c, 0x75ba95fc5fedb601,
			0x79fb732b77622510, 0x18905f76a53755c6},
		y: p256Element{0xddf25357ce95560a, 0x8b4ab8e4ba19e45c,
			0xd2e88688dd21f325, 0x8571ff1825885d85},
		z: p256One,
	}
}

// Set sets p = q and returns p.
func (p *P256Point) Set(q *P256Point) *P256Point {
	p.x, p.y, p.z = q.x, q.y, q.z
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
		p256BigToLittle(&r.x, (*[32]byte)(b[1:33]))
		p256BigToLittle(&r.y, (*[32]byte)(b[33:65]))
		if p256LessThanP(&r.x) == 0 || p256LessThanP(&r.y) == 0 {
			return nil, errors.New("invalid P256 element encoding")
		}
		// p256Mul operates in the Montgomery domain with R = 2²⁵⁶ mod p. Thus rr
		// here is R in the Montgomery domain, or R×R mod p. See comment in
		// P256OrdInverse about how this is used.
		rr := p256Element{0x0000000000000003, 0xfffffffbffffffff,
			0xfffffffffffffffe, 0x00000004fffffffd}
		p256Mul(&r.x, &r.x, &rr)
		p256Mul(&r.y, &r.y, &rr)
		if err := p256CheckOnCurve(&r.x, &r.y); err != nil {
			return nil, err
		}
		r.z = p256One
		return p.Set(&r), nil

	// Compressed form.
	case len(b) == p256CompressedLength && (b[0] == 2 || b[0] == 3):
		return nil, errors.New("unimplemented") // TODO(filippo)

	default:
		return nil, errors.New("invalid P256 point encoding")
	}
}

func p256CheckOnCurve(x, y *p256Element) error {
	// x³ - 3x + b
	x3 := new(p256Element)
	p256Sqr(x3, x, 1)
	p256Mul(x3, x3, x)

	threeX := new(p256Element)
	p256Add(threeX, x, x)
	p256Add(threeX, threeX, x)
	p256NegCond(threeX, 1)

	p256B := &p256Element{0xd89cdf6229c4bddf, 0xacf005cd78843090,
		0xe5a220abf7212ed6, 0xdc30061d04874834}

	p256Add(x3, x3, threeX)
	p256Add(x3, x3, p256B)

	// y² = x³ - 3x + b
	y2 := new(p256Element)
	p256Sqr(y2, y, 1)

	if p256Equal(y2, x3) != 1 {
		return errors.New("P256 point not on curve")
	}
	return nil
}

// p256LessThanP returns 1 if x < p, and 0 otherwise. Note that a p256Element is
// not allowed to be equal to or greater than p, so if this function returns 0
// then x is invalid.
func p256LessThanP(x *p256Element) int {
	var b uint64
	_, b = bits.Sub64(x[0], p256P[0], b)
	_, b = bits.Sub64(x[1], p256P[1], b)
	_, b = bits.Sub64(x[2], p256P[2], b)
	_, b = bits.Sub64(x[3], p256P[3], b)
	return int(b)
}

// p256Add sets res = x + y.
func p256Add(res, x, y *p256Element) {
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

// The following assembly functions are implemented in p256_asm_*.s

// Montgomery multiplication. Sets res = in1 * in2 * R⁻¹ mod p.
//
//go:noescape
func p256Mul(res, in1, in2 *p256Element)

// Montgomery square, repeated n times (n >= 1).
//
//go:noescape
func p256Sqr(res, in *p256Element, n int)

// Montgomery multiplication by R⁻¹, or 1 outside the domain.
// Sets res = in * R⁻¹, bringing res out of the Montgomery domain.
//
//go:noescape
func p256FromMont(res, in *p256Element)

// If cond is not 0, sets val = -val mod p.
//
//go:noescape
func p256NegCond(val *p256Element, cond int)

// If cond is 0, sets res = b, otherwise sets res = a.
//
//go:noescape
func p256MovCond(res, a, b *P256Point, cond int)

//go:noescape
func p256BigToLittle(res *p256Element, in *[32]byte)

//go:noescape
func p256LittleToBig(res *[32]byte, in *p256Element)

//go:noescape
func p256OrdBigToLittle(res *p256OrdElement, in *[32]byte)

//go:noescape
func p256OrdLittleToBig(res *[32]byte, in *p256OrdElement)

// p256Table is a table of the first 16 multiples of a point. Points are stored
// at an index offset of -1 so [8]P is at index 7, P is at 0, and [16]P is at 15.
// [0]P is the point at infinity and it's not stored.
type p256Table [16]P256Point

// p256Select sets res to the point at index idx in the table.
// idx must be in [0, 15]. It executes in constant time.
//
//go:noescape
func p256Select(res *P256Point, table *p256Table, idx int)

// p256AffinePoint is a point in affine coordinates (x, y). x and y are still
// Montgomery domain elements. The point can't be the point at infinity.
type p256AffinePoint struct {
	x, y p256Element
}

// p256AffineTable is a table of the first 32 multiples of a point. Points are
// stored at an index offset of -1 like in p256Table, and [0]P is not stored.
type p256AffineTable [32]p256AffinePoint

// p256Precomputed is a series of precomputed multiples of G, the canonical
// generator. The first p256AffineTable contains multiples of G. The second one
// multiples of [2⁶]G, the third one of [2¹²]G, and so on, where each successive
// table is the previous table doubled six times. Six is the width of the
// sliding window used in p256ScalarMult, and having each table already
// pre-doubled lets us avoid the doublings between windows entirely. This table
// MUST NOT be modified, as it aliases into p256PrecomputedEmbed below.
var p256Precomputed *[43]p256AffineTable

//go:embed p256_asm_table.bin
var p256PrecomputedEmbed string

func init() {
	p256PrecomputedPtr := (*unsafe.Pointer)(unsafe.Pointer(&p256PrecomputedEmbed))
	p256Precomputed = (*[43]p256AffineTable)(*p256PrecomputedPtr)
}

// p256SelectAffine sets res to the point at index idx in the table.
// idx must be in [0, 31]. It executes in constant time.
//
//go:noescape
func p256SelectAffine(res *p256AffinePoint, table *p256AffineTable, idx int)

// Point addition with an affine point and constant time conditions.
// If zero is 0, sets res = in2. If sel is 0, sets res = in1.
// If sign is not 0, sets res = in1 + -in2. Otherwise, sets res = in1 + in2
//
//go:noescape
func p256PointAddAffineAsm(res, in1 *P256Point, in2 *p256AffinePoint, sign, sel, zero int)

// Point addition. Sets res = in1 + in2. Returns one if the two input points
// were equal and zero otherwise. If in1 or in2 are the point at infinity, res
// and the return value are undefined.
//
//go:noescape
func p256PointAddAsm(res, in1, in2 *P256Point) int

// Point doubling. Sets res = in + in. in can be the point at infinity.
//
//go:noescape
func p256PointDoubleAsm(res, in *P256Point)

// p256OrdElement is a P-256 scalar field element in [0, ord(G)-1] in the
// Montgomery domain (with R 2²⁵⁶) as four uint64 limbs in little-endian order.
type p256OrdElement [4]uint64

// Montgomery multiplication modulo org(G). Sets res = in1 * in2 * R⁻¹.
//
//go:noescape
func p256OrdMul(res, in1, in2 *p256OrdElement)

// Montgomery square modulo org(G), repeated n times (n >= 1).
//
//go:noescape
func p256OrdSqr(res, in *p256OrdElement, n int)

func P256OrdInverse(k []byte) ([]byte, error) {
	if len(k) != 32 {
		return nil, errors.New("invalid scalar length")
	}

	x := new(p256OrdElement)
	p256OrdBigToLittle(x, (*[32]byte)(k))

	// Inversion is implemented as exponentiation by n - 2, per Fermat's little theorem.
	//
	// The sequence of 38 multiplications and 254 squarings is derived from
	// https://briansmith.org/ecc-inversion-addition-chains-01#p256_scalar_inversion
	_1 := new(p256OrdElement)
	_11 := new(p256OrdElement)
	_101 := new(p256OrdElement)
	_111 := new(p256OrdElement)
	_1111 := new(p256OrdElement)
	_10101 := new(p256OrdElement)
	_101111 := new(p256OrdElement)
	t := new(p256OrdElement)

	// This code operates in the Montgomery domain where R = 2²⁵⁶ mod n and n is
	// the order of the scalar field. Elements in the Montgomery domain take the
	// form a×R and p256OrdMul calculates (a × b × R⁻¹) mod n. RR is R in the
	// domain, or R×R mod n, thus p256OrdMul(x, RR) gives x×R, i.e. converts x
	// into the Montgomery domain.
	RR := &p256OrdElement{0x83244c95be79eea2, 0x4699799c49bd6fa6,
		0x2845b2392b6bec59, 0x66e12d94f3d95620}

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

	sqrs := []int{
		6, 5, 4, 5, 5,
		4, 3, 3, 5, 9,
		6, 2, 5, 6, 5,
		4, 5, 5, 3, 10,
		2, 5, 5, 3, 7, 6}
	muls := []*p256OrdElement{
		_101111, _111, _11, _1111, _10101,
		_101, _101, _101, _111, _101111,
		_1111, _1, _1, _1111, _111,
		_111, _111, _101, _11, _101111,
		_11, _11, _11, _1, _10101, _1111}

	for i, s := range sqrs {
		p256OrdSqr(x, x, s)
		p256OrdMul(x, x, muls[i])
	}

	// Montgomery multiplication by R⁻¹, or 1 outside the domain as R⁻¹×R = 1,
	// converts a Montgomery value out of the domain.
	one := &p256OrdElement{1}
	p256OrdMul(x, x, one)

	var xOut [32]byte
	p256OrdLittleToBig(&xOut, x)
	return xOut[:], nil
}

// Add sets q = p1 + p2, and returns q. The points may overlap.
func (q *P256Point) Add(r1, r2 *P256Point) *P256Point {
	var sum, double P256Point
	r1IsInfinity := r1.isInfinity()
	r2IsInfinity := r2.isInfinity()
	pointsEqual := p256PointAddAsm(&sum, r1, r2)
	p256PointDoubleAsm(&double, r1)
	p256MovCond(&sum, &double, &sum, pointsEqual)
	p256MovCond(&sum, r1, &sum, r2IsInfinity)
	p256MovCond(&sum, r2, &sum, r1IsInfinity)
	return q.Set(&sum)
}

// Double sets q = p + p, and returns q. The points may overlap.
func (q *P256Point) Double(p *P256Point) *P256Point {
	var double P256Point
	p256PointDoubleAsm(&double, p)
	return q.Set(&double)
}

// ScalarBaseMult sets r = scalar * generator, where scalar is a 32-byte big
// endian value, and returns r. If scalar is not 32 bytes long, ScalarBaseMult
// returns an error and the receiver is unchanged.
func (r *P256Point) ScalarBaseMult(scalar []byte) (*P256Point, error) {
	if len(scalar) != 32 {
		return nil, errors.New("invalid scalar length")
	}
	scalarReversed := new(p256OrdElement)
	p256OrdBigToLittle(scalarReversed, (*[32]byte)(scalar))

	r.p256BaseMult(scalarReversed)
	return r, nil
}

// ScalarMult sets r = scalar * q, where scalar is a 32-byte big endian value,
// and returns r. If scalar is not 32 bytes long, ScalarBaseMult returns an
// error and the receiver is unchanged.
func (r *P256Point) ScalarMult(q *P256Point, scalar []byte) (*P256Point, error) {
	if len(scalar) != 32 {
		return nil, errors.New("invalid scalar length")
	}
	scalarReversed := new(p256OrdElement)
	p256OrdBigToLittle(scalarReversed, (*[32]byte)(scalar))

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

// p256Equal returns 1 if a and b are equal and 0 otherwise.
func p256Equal(a, b *p256Element) int {
	var acc uint64
	for i := range a {
		acc |= a[i] ^ b[i]
	}
	return uint64IsZero(acc)
}

// isInfinity returns 1 if p is the point at infinity and 0 otherwise.
func (p *P256Point) isInfinity() int {
	return p256Equal(&p.z, &p256Zero)
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

	zInv := new(p256Element)
	zInvSq := new(p256Element)
	p256Inverse(zInv, &p.z)
	p256Sqr(zInvSq, zInv, 1)
	p256Mul(zInv, zInv, zInvSq)

	p256Mul(zInvSq, &p.x, zInvSq)
	p256Mul(zInv, &p.y, zInv)

	p256FromMont(zInvSq, zInvSq)
	p256FromMont(zInv, zInv)

	out[0] = 4 // Uncompressed form.
	p256LittleToBig((*[32]byte)(out[1:33]), zInvSq)
	p256LittleToBig((*[32]byte)(out[33:65]), zInv)

	return out[:]
}

// Select sets q to p1 if cond == 1, and to p2 if cond == 0.
func (q *P256Point) Select(p1, p2 *P256Point, cond int) *P256Point {
	p256MovCond(q, p1, p2, cond)
	return q
}

// p256Inverse sets out to in⁻¹ mod p. If in is zero, out will be zero.
func p256Inverse(out, in *p256Element) {
	// Inversion is calculated through exponentiation by p - 2, per Fermat's
	// little theorem.
	//
	// The sequence of 12 multiplications and 255 squarings is derived from the
	// following addition chain generated with github.com/mmcloughlin/addchain
	// v0.4.0.
	//
	//  _10     = 2*1
	//  _11     = 1 + _10
	//  _110    = 2*_11
	//  _111    = 1 + _110
	//  _111000 = _111 << 3
	//  _111111 = _111 + _111000
	//  x12     = _111111 << 6 + _111111
	//  x15     = x12 << 3 + _111
	//  x16     = 2*x15 + 1
	//  x32     = x16 << 16 + x16
	//  i53     = x32 << 15
	//  x47     = x15 + i53
	//  i263    = ((i53 << 17 + 1) << 143 + x47) << 47
	//  return    (x47 + i263) << 2 + 1
	//
	var z = new(p256Element)
	var t0 = new(p256Element)
	var t1 = new(p256Element)

	p256Sqr(z, in, 1)
	p256Mul(z, in, z)
	p256Sqr(z, z, 1)
	p256Mul(z, in, z)
	p256Sqr(t0, z, 3)
	p256Mul(t0, z, t0)
	p256Sqr(t1, t0, 6)
	p256Mul(t0, t0, t1)
	p256Sqr(t0, t0, 3)
	p256Mul(z, z, t0)
	p256Sqr(t0, z, 1)
	p256Mul(t0, in, t0)
	p256Sqr(t1, t0, 16)
	p256Mul(t0, t0, t1)
	p256Sqr(t0, t0, 15)
	p256Mul(z, z, t0)
	p256Sqr(t0, t0, 17)
	p256Mul(t0, in, t0)
	p256Sqr(t0, t0, 143)
	p256Mul(t0, z, t0)
	p256Sqr(t0, t0, 47)
	p256Mul(z, z, t0)
	p256Sqr(z, z, 2)
	p256Mul(out, in, z)
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

func (p *P256Point) p256BaseMult(scalar *p256OrdElement) {
	var t0 p256AffinePoint

	wvalue := (scalar[0] << 1) & 0x7f
	sel, sign := boothW6(uint(wvalue))
	p256SelectAffine(&t0, &p256Precomputed[0], sel)
	p.x, p.y, p.z = t0.x, t0.y, p256One
	p256NegCond(&p.y, sign)

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
		p256SelectAffine(&t0, &p256Precomputed[i], sel)
		p256PointAddAffineAsm(p, p, &t0, sign, sel, zero)
		zero |= sel
	}

	// If the whole scalar was zero, set to the point at infinity.
	p256MovCond(p, p, NewP256Point(), zero)
}

func (p *P256Point) p256ScalarMult(scalar *p256OrdElement) {
	// precomp is a table of precomputed points that stores powers of p
	// from p^1 to p^16.
	var precomp p256Table
	var t0, t1, t2, t3 P256Point

	// Prepare the table
	precomp[0] = *p // 1

	p256PointDoubleAsm(&t0, p)
	p256PointDoubleAsm(&t1, &t0)
	p256PointDoubleAsm(&t2, &t1)
	p256PointDoubleAsm(&t3, &t2)
	precomp[1] = t0  // 2
	precomp[3] = t1  // 4
	precomp[7] = t2  // 8
	precomp[15] = t3 // 16

	p256PointAddAsm(&t0, &t0, p)
	p256PointAddAsm(&t1, &t1, p)
	p256PointAddAsm(&t2, &t2, p)
	precomp[2] = t0 // 3
	precomp[4] = t1 // 5
	precomp[8] = t2 // 9

	p256PointDoubleAsm(&t0, &t0)
	p256PointDoubleAsm(&t1, &t1)
	precomp[5] = t0 // 6
	precomp[9] = t1 // 10

	p256PointAddAsm(&t2, &t0, p)
	p256PointAddAsm(&t1, &t1, p)
	precomp[6] = t2  // 7
	precomp[10] = t1 // 11

	p256PointDoubleAsm(&t0, &t0)
	p256PointDoubleAsm(&t2, &t2)
	precomp[11] = t0 // 12
	precomp[13] = t2 // 14

	p256PointAddAsm(&t0, &t0, p)
	p256PointAddAsm(&t2, &t2, p)
	precomp[12] = t0 // 13
	precomp[14] = t2 // 15

	// Start scanning the window from top bit
	index := uint(254)
	var sel, sign int

	wvalue := (scalar[index/64] >> (index % 64)) & 0x3f
	sel, _ = boothW5(uint(wvalue))

	p256Select(p, &precomp, sel)
	zero := sel

	for index > 4 {
		index -= 5
		p256PointDoubleAsm(p, p)
		p256PointDoubleAsm(p, p)
		p256PointDoubleAsm(p, p)
		p256PointDoubleAsm(p, p)
		p256PointDoubleAsm(p, p)

		if index < 192 {
			wvalue = ((scalar[index/64] >> (index % 64)) + (scalar[index/64+1] << (64 - (index % 64)))) & 0x3f
		} else {
			wvalue = (scalar[index/64] >> (index % 64)) & 0x3f
		}

		sel, sign = boothW5(uint(wvalue))

		p256Select(&t0, &precomp, sel)
		p256NegCond(&t0.y, sign)
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

	wvalue = (scalar[0] << 1) & 0x3f
	sel, sign = boothW5(uint(wvalue))

	p256Select(&t0, &precomp, sel)
	p256NegCond(&t0.y, sign)
	p256PointAddAsm(&t1, p, &t0)
	p256MovCond(&t1, &t1, p, sel)
	p256MovCond(p, &t1, &t0, zero)
}

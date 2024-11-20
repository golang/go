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

//go:build (amd64 || arm64 || ppc64le || s390x) && !purego

package nistec

import (
	"crypto/internal/fipsdeps/byteorder"
	"errors"
	"math/bits"
	"runtime"
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

// SetGenerator sets p to the canonical generator and returns p.
func (p *P256Point) SetGenerator() *P256Point {
	p.x = p256Element{0x79e730d418a9143c, 0x75ba95fc5fedb601,
		0x79fb732b77622510, 0x18905f76a53755c6}
	p.y = p256Element{0xddf25357ce95560a, 0x8b4ab8e4ba19e45c,
		0xd2e88688dd21f325, 0x8571ff1825885d85}
	p.z = p256One
	return p
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
	// p256Mul operates in the Montgomery domain with R = 2²⁵⁶ mod p. Thus rr
	// here is R in the Montgomery domain, or R×R mod p. See comment in
	// P256OrdInverse about how this is used.
	rr := p256Element{0x0000000000000003, 0xfffffffbffffffff,
		0xfffffffffffffffe, 0x00000004fffffffd}

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
		p256Mul(&r.x, &r.x, &rr)
		p256Mul(&r.y, &r.y, &rr)
		if err := p256CheckOnCurve(&r.x, &r.y); err != nil {
			return nil, err
		}
		r.z = p256One
		return p.Set(&r), nil

	// Compressed form.
	case len(b) == p256CompressedLength && (b[0] == 2 || b[0] == 3):
		var r P256Point
		p256BigToLittle(&r.x, (*[32]byte)(b[1:33]))
		if p256LessThanP(&r.x) == 0 {
			return nil, errors.New("invalid P256 element encoding")
		}
		p256Mul(&r.x, &r.x, &rr)

		// y² = x³ - 3x + b
		p256Polynomial(&r.y, &r.x)
		if !p256Sqrt(&r.y, &r.y) {
			return nil, errors.New("invalid P256 compressed point encoding")
		}

		// Select the positive or negative root, as indicated by the least
		// significant bit, based on the encoding type byte.
		yy := new(p256Element)
		p256FromMont(yy, &r.y)
		cond := int(yy[0]&1) ^ int(b[0]&1)
		p256NegCond(&r.y, cond)

		r.z = p256One
		return p.Set(&r), nil

	default:
		return nil, errors.New("invalid P256 point encoding")
	}
}

// p256Polynomial sets y2 to x³ - 3x + b, and returns y2.
func p256Polynomial(y2, x *p256Element) *p256Element {
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

	*y2 = *x3
	return y2
}

func p256CheckOnCurve(x, y *p256Element) error {
	// y² = x³ - 3x + b
	rhs := p256Polynomial(new(p256Element), x)
	lhs := new(p256Element)
	p256Sqr(lhs, y, 1)
	if p256Equal(lhs, rhs) != 1 {
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

func p256BigToLittle(l *p256Element, b *[32]byte) {
	bytesToLimbs((*[4]uint64)(l), b)
}

func bytesToLimbs(l *[4]uint64, b *[32]byte) {
	l[0] = byteorder.BEUint64(b[24:])
	l[1] = byteorder.BEUint64(b[16:])
	l[2] = byteorder.BEUint64(b[8:])
	l[3] = byteorder.BEUint64(b[:])
}

func p256LittleToBig(b *[32]byte, l *p256Element) {
	limbsToBytes(b, (*[4]uint64)(l))
}

func limbsToBytes(b *[32]byte, l *[4]uint64) {
	byteorder.BEPutUint64(b[24:], l[0])
	byteorder.BEPutUint64(b[16:], l[1])
	byteorder.BEPutUint64(b[8:], l[2])
	byteorder.BEPutUint64(b[:], l[3])
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

// p256Sqrt sets e to a square root of x. If x is not a square, p256Sqrt returns
// false and e is unchanged. e and x can overlap.
func p256Sqrt(e, x *p256Element) (isSquare bool) {
	t0, t1 := new(p256Element), new(p256Element)

	// Since p = 3 mod 4, exponentiation by (p + 1) / 4 yields a square root candidate.
	//
	// The sequence of 7 multiplications and 253 squarings is derived from the
	// following addition chain generated with github.com/mmcloughlin/addchain v0.4.0.
	//
	//	_10       = 2*1
	//	_11       = 1 + _10
	//	_1100     = _11 << 2
	//	_1111     = _11 + _1100
	//	_11110000 = _1111 << 4
	//	_11111111 = _1111 + _11110000
	//	x16       = _11111111 << 8 + _11111111
	//	x32       = x16 << 16 + x16
	//	return      ((x32 << 32 + 1) << 96 + 1) << 94
	//
	p256Sqr(t0, x, 1)
	p256Mul(t0, x, t0)
	p256Sqr(t1, t0, 2)
	p256Mul(t0, t0, t1)
	p256Sqr(t1, t0, 4)
	p256Mul(t0, t0, t1)
	p256Sqr(t1, t0, 8)
	p256Mul(t0, t0, t1)
	p256Sqr(t1, t0, 16)
	p256Mul(t0, t0, t1)
	p256Sqr(t0, t0, 32)
	p256Mul(t0, x, t0)
	p256Sqr(t0, t0, 96)
	p256Mul(t0, x, t0)
	p256Sqr(t0, t0, 94)

	p256Sqr(t1, t0, 1)
	if p256Equal(t1, x) != 1 {
		return false
	}
	*e = *t0
	return true
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
// sliding window used in p256ScalarBaseMult, and having each table already
// pre-doubled lets us avoid the doublings between windows entirely. This table
// aliases into p256PrecomputedEmbed.
var p256Precomputed *[43]p256AffineTable

func init() {
	p256PrecomputedPtr := unsafe.Pointer(&p256PrecomputedEmbed)
	if runtime.GOARCH == "s390x" {
		var newTable [43 * 32 * 2 * 4]uint64
		for i, x := range (*[43 * 32 * 2 * 4][8]byte)(p256PrecomputedPtr) {
			newTable[i] = byteorder.LEUint64(x[:])
		}
		p256PrecomputedPtr = unsafe.Pointer(&newTable)
	}
	p256Precomputed = (*[43]p256AffineTable)(p256PrecomputedPtr)
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

// p256OrdReduce ensures s is in the range [0, ord(G)-1].
func p256OrdReduce(s *p256OrdElement) {
	// Since 2 * ord(G) > 2²⁵⁶, we can just conditionally subtract ord(G),
	// keeping the result if it doesn't underflow.
	t0, b := bits.Sub64(s[0], 0xf3b9cac2fc632551, 0)
	t1, b := bits.Sub64(s[1], 0xbce6faada7179e84, b)
	t2, b := bits.Sub64(s[2], 0xffffffffffffffff, b)
	t3, b := bits.Sub64(s[3], 0xffffffff00000000, b)
	tMask := b - 1 // zero if subtraction underflowed
	s[0] ^= (t0 ^ s[0]) & tMask
	s[1] ^= (t1 ^ s[1]) & tMask
	s[2] ^= (t2 ^ s[2]) & tMask
	s[3] ^= (t3 ^ s[3]) & tMask
}

func p256OrdLittleToBig(b *[32]byte, l *p256OrdElement) {
	limbsToBytes(b, (*[4]uint64)(l))
}

func p256OrdBigToLittle(l *p256OrdElement, b *[32]byte) {
	bytesToLimbs((*[4]uint64)(l), b)
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
	p256OrdReduce(scalarReversed)

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
	p256OrdReduce(scalarReversed)

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
		return append(out[:0], 0)
	}

	x, y := new(p256Element), new(p256Element)
	p.affineFromMont(x, y)

	out[0] = 4 // Uncompressed form.
	p256LittleToBig((*[32]byte)(out[1:33]), x)
	p256LittleToBig((*[32]byte)(out[33:65]), y)

	return out[:]
}

// affineFromMont sets (x, y) to the affine coordinates of p, converted out of the
// Montgomery domain.
func (p *P256Point) affineFromMont(x, y *p256Element) {
	p256Inverse(y, &p.z)
	p256Sqr(x, y, 1)
	p256Mul(y, y, x)

	p256Mul(x, &p.x, x)
	p256Mul(y, &p.y, y)

	p256FromMont(x, x)
	p256FromMont(y, y)
}

// BytesX returns the encoding of the x-coordinate of p, as specified in SEC 1,
// Version 2.0, Section 2.3.5, or an error if p is the point at infinity.
func (p *P256Point) BytesX() ([]byte, error) {
	// This function is outlined to make the allocations inline in the caller
	// rather than happen on the heap.
	var out [p256ElementLength]byte
	return p.bytesX(&out)
}

func (p *P256Point) bytesX(out *[p256ElementLength]byte) ([]byte, error) {
	if p.isInfinity() == 1 {
		return nil, errors.New("P256 point is the point at infinity")
	}

	x := new(p256Element)
	p256Inverse(x, &p.z)
	p256Sqr(x, x, 1)
	p256Mul(x, &p.x, x)
	p256FromMont(x, x)
	p256LittleToBig((*[32]byte)(out[:]), x)

	return out[:], nil
}

// BytesCompressed returns the compressed or infinity encoding of p, as
// specified in SEC 1, Version 2.0, Section 2.3.3. Note that the encoding of the
// point at infinity is shorter than all other encodings.
func (p *P256Point) BytesCompressed() []byte {
	// This function is outlined to make the allocations inline in the caller
	// rather than happen on the heap.
	var out [p256CompressedLength]byte
	return p.bytesCompressed(&out)
}

func (p *P256Point) bytesCompressed(out *[p256CompressedLength]byte) []byte {
	if p.isInfinity() == 1 {
		return append(out[:0], 0)
	}

	x, y := new(p256Element), new(p256Element)
	p.affineFromMont(x, y)

	out[0] = 2 | byte(y[0]&1)
	p256LittleToBig((*[32]byte)(out[1:33]), x)

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

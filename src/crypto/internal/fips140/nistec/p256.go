// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && !arm64 && !ppc64le && !s390x) || purego

package nistec

import (
	"crypto/internal/fips140/nistec/fiat"
	"crypto/internal/fips140/subtle"
	"crypto/internal/fips140deps/byteorder"
	"crypto/internal/fips140deps/cpu"
	"errors"
	"math/bits"
	"sync"
	"unsafe"
)

// P256Point is a P-256 point. The zero value is NOT valid.
type P256Point struct {
	// The point is represented in projective coordinates (X:Y:Z), where x = X/Z
	// and y = Y/Z. Infinity is (0:1:0).
	//
	// fiat.P256Element is a base field element in [0, P-1] in the Montgomery
	// domain (with R 2²⁵⁶ and P 2²⁵⁶ - 2²²⁴ + 2¹⁹² + 2⁹⁶ - 1) as four limbs in
	// little-endian order value.
	x, y, z fiat.P256Element
}

// NewP256Point returns a new P256Point representing the point at infinity point.
func NewP256Point() *P256Point {
	p := &P256Point{}
	p.y.One()
	return p
}

// SetGenerator sets p to the canonical generator and returns p.
func (p *P256Point) SetGenerator() *P256Point {
	p.x.SetBytes([]byte{0x6b, 0x17, 0xd1, 0xf2, 0xe1, 0x2c, 0x42, 0x47, 0xf8, 0xbc, 0xe6, 0xe5, 0x63, 0xa4, 0x40, 0xf2, 0x77, 0x3, 0x7d, 0x81, 0x2d, 0xeb, 0x33, 0xa0, 0xf4, 0xa1, 0x39, 0x45, 0xd8, 0x98, 0xc2, 0x96})
	p.y.SetBytes([]byte{0x4f, 0xe3, 0x42, 0xe2, 0xfe, 0x1a, 0x7f, 0x9b, 0x8e, 0xe7, 0xeb, 0x4a, 0x7c, 0xf, 0x9e, 0x16, 0x2b, 0xce, 0x33, 0x57, 0x6b, 0x31, 0x5e, 0xce, 0xcb, 0xb6, 0x40, 0x68, 0x37, 0xbf, 0x51, 0xf5})
	p.z.One()
	return p
}

// Set sets p = q and returns p.
func (p *P256Point) Set(q *P256Point) *P256Point {
	p.x.Set(&q.x)
	p.y.Set(&q.y)
	p.z.Set(&q.z)
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
		x, err := new(fiat.P256Element).SetBytes(b[1 : 1+p256ElementLength])
		if err != nil {
			return nil, err
		}
		y, err := new(fiat.P256Element).SetBytes(b[1+p256ElementLength:])
		if err != nil {
			return nil, err
		}
		if err := p256CheckOnCurve(x, y); err != nil {
			return nil, err
		}
		p.x.Set(x)
		p.y.Set(y)
		p.z.One()
		return p, nil

	// Compressed form.
	case len(b) == p256CompressedLength && (b[0] == 2 || b[0] == 3):
		x, err := new(fiat.P256Element).SetBytes(b[1:])
		if err != nil {
			return nil, err
		}

		// y² = x³ - 3x + b
		y := p256Polynomial(new(fiat.P256Element), x)
		if !p256Sqrt(y, y) {
			return nil, errors.New("invalid P256 compressed point encoding")
		}

		// Select the positive or negative root, as indicated by the least
		// significant bit, based on the encoding type byte.
		otherRoot := new(fiat.P256Element)
		otherRoot.Sub(otherRoot, y)
		cond := y.Bytes()[p256ElementLength-1]&1 ^ b[0]&1
		y.Select(otherRoot, y, int(cond))

		p.x.Set(x)
		p.y.Set(y)
		p.z.One()
		return p, nil

	default:
		return nil, errors.New("invalid P256 point encoding")
	}
}

var _p256B *fiat.P256Element
var _p256BOnce sync.Once

func p256B() *fiat.P256Element {
	_p256BOnce.Do(func() {
		_p256B, _ = new(fiat.P256Element).SetBytes([]byte{0x5a, 0xc6, 0x35, 0xd8, 0xaa, 0x3a, 0x93, 0xe7, 0xb3, 0xeb, 0xbd, 0x55, 0x76, 0x98, 0x86, 0xbc, 0x65, 0x1d, 0x6, 0xb0, 0xcc, 0x53, 0xb0, 0xf6, 0x3b, 0xce, 0x3c, 0x3e, 0x27, 0xd2, 0x60, 0x4b})
	})
	return _p256B
}

// p256Polynomial sets y2 to x³ - 3x + b, and returns y2.
func p256Polynomial(y2, x *fiat.P256Element) *fiat.P256Element {
	y2.Square(x)
	y2.Mul(y2, x)

	threeX := new(fiat.P256Element).Add(x, x)
	threeX.Add(threeX, x)
	y2.Sub(y2, threeX)

	return y2.Add(y2, p256B())
}

func p256CheckOnCurve(x, y *fiat.P256Element) error {
	// y² = x³ - 3x + b
	rhs := p256Polynomial(new(fiat.P256Element), x)
	lhs := new(fiat.P256Element).Square(y)
	if rhs.Equal(lhs) != 1 {
		return errors.New("P256 point not on curve")
	}
	return nil
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
	// The SEC 1 representation of the point at infinity is a single zero byte,
	// and only infinity has z = 0.
	if p.z.IsZero() == 1 {
		return append(out[:0], 0)
	}

	zinv := new(fiat.P256Element).Invert(&p.z)
	x := new(fiat.P256Element).Mul(&p.x, zinv)
	y := new(fiat.P256Element).Mul(&p.y, zinv)

	buf := append(out[:0], 4)
	buf = append(buf, x.Bytes()...)
	buf = append(buf, y.Bytes()...)
	return buf
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
	if p.z.IsZero() == 1 {
		return nil, errors.New("P256 point is the point at infinity")
	}

	zinv := new(fiat.P256Element).Invert(&p.z)
	x := new(fiat.P256Element).Mul(&p.x, zinv)

	return append(out[:0], x.Bytes()...), nil
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
	if p.z.IsZero() == 1 {
		return append(out[:0], 0)
	}

	zinv := new(fiat.P256Element).Invert(&p.z)
	x := new(fiat.P256Element).Mul(&p.x, zinv)
	y := new(fiat.P256Element).Mul(&p.y, zinv)

	// Encode the sign of the y coordinate (indicated by the least significant
	// bit) as the encoding type (2 or 3).
	buf := append(out[:0], 2)
	buf[0] |= y.Bytes()[p256ElementLength-1] & 1
	buf = append(buf, x.Bytes()...)
	return buf
}

// Add sets q = p1 + p2, and returns q. The points may overlap.
func (q *P256Point) Add(p1, p2 *P256Point) *P256Point {
	// Complete addition formula for a = -3 from "Complete addition formulas for
	// prime order elliptic curves" (https://eprint.iacr.org/2015/1060), §A.2.

	t0 := new(fiat.P256Element).Mul(&p1.x, &p2.x) // t0 := X1 * X2
	t1 := new(fiat.P256Element).Mul(&p1.y, &p2.y) // t1 := Y1 * Y2
	t2 := new(fiat.P256Element).Mul(&p1.z, &p2.z) // t2 := Z1 * Z2
	t3 := new(fiat.P256Element).Add(&p1.x, &p1.y) // t3 := X1 + Y1
	t4 := new(fiat.P256Element).Add(&p2.x, &p2.y) // t4 := X2 + Y2
	t3.Mul(t3, t4)                                // t3 := t3 * t4
	t4.Add(t0, t1)                                // t4 := t0 + t1
	t3.Sub(t3, t4)                                // t3 := t3 - t4
	t4.Add(&p1.y, &p1.z)                          // t4 := Y1 + Z1
	x3 := new(fiat.P256Element).Add(&p2.y, &p2.z) // X3 := Y2 + Z2
	t4.Mul(t4, x3)                                // t4 := t4 * X3
	x3.Add(t1, t2)                                // X3 := t1 + t2
	t4.Sub(t4, x3)                                // t4 := t4 - X3
	x3.Add(&p1.x, &p1.z)                          // X3 := X1 + Z1
	y3 := new(fiat.P256Element).Add(&p2.x, &p2.z) // Y3 := X2 + Z2
	x3.Mul(x3, y3)                                // X3 := X3 * Y3
	y3.Add(t0, t2)                                // Y3 := t0 + t2
	y3.Sub(x3, y3)                                // Y3 := X3 - Y3
	z3 := new(fiat.P256Element).Mul(p256B(), t2)  // Z3 := b * t2
	x3.Sub(y3, z3)                                // X3 := Y3 - Z3
	z3.Add(x3, x3)                                // Z3 := X3 + X3
	x3.Add(x3, z3)                                // X3 := X3 + Z3
	z3.Sub(t1, x3)                                // Z3 := t1 - X3
	x3.Add(t1, x3)                                // X3 := t1 + X3
	y3.Mul(p256B(), y3)                           // Y3 := b * Y3
	t1.Add(t2, t2)                                // t1 := t2 + t2
	t2.Add(t1, t2)                                // t2 := t1 + t2
	y3.Sub(y3, t2)                                // Y3 := Y3 - t2
	y3.Sub(y3, t0)                                // Y3 := Y3 - t0
	t1.Add(y3, y3)                                // t1 := Y3 + Y3
	y3.Add(t1, y3)                                // Y3 := t1 + Y3
	t1.Add(t0, t0)                                // t1 := t0 + t0
	t0.Add(t1, t0)                                // t0 := t1 + t0
	t0.Sub(t0, t2)                                // t0 := t0 - t2
	t1.Mul(t4, y3)                                // t1 := t4 * Y3
	t2.Mul(t0, y3)                                // t2 := t0 * Y3
	y3.Mul(x3, z3)                                // Y3 := X3 * Z3
	y3.Add(y3, t2)                                // Y3 := Y3 + t2
	x3.Mul(t3, x3)                                // X3 := t3 * X3
	x3.Sub(x3, t1)                                // X3 := X3 - t1
	z3.Mul(t4, z3)                                // Z3 := t4 * Z3
	t1.Mul(t3, t0)                                // t1 := t3 * t0
	z3.Add(z3, t1)                                // Z3 := Z3 + t1

	q.x.Set(x3)
	q.y.Set(y3)
	q.z.Set(z3)
	return q
}

// Double sets q = p + p, and returns q. The points may overlap.
func (q *P256Point) Double(p *P256Point) *P256Point {
	// Complete addition formula for a = -3 from "Complete addition formulas for
	// prime order elliptic curves" (https://eprint.iacr.org/2015/1060), §A.2.

	t0 := new(fiat.P256Element).Square(&p.x)     // t0 := X ^ 2
	t1 := new(fiat.P256Element).Square(&p.y)     // t1 := Y ^ 2
	t2 := new(fiat.P256Element).Square(&p.z)     // t2 := Z ^ 2
	t3 := new(fiat.P256Element).Mul(&p.x, &p.y)  // t3 := X * Y
	t3.Add(t3, t3)                               // t3 := t3 + t3
	z3 := new(fiat.P256Element).Mul(&p.x, &p.z)  // Z3 := X * Z
	z3.Add(z3, z3)                               // Z3 := Z3 + Z3
	y3 := new(fiat.P256Element).Mul(p256B(), t2) // Y3 := b * t2
	y3.Sub(y3, z3)                               // Y3 := Y3 - Z3
	x3 := new(fiat.P256Element).Add(y3, y3)      // X3 := Y3 + Y3
	y3.Add(x3, y3)                               // Y3 := X3 + Y3
	x3.Sub(t1, y3)                               // X3 := t1 - Y3
	y3.Add(t1, y3)                               // Y3 := t1 + Y3
	y3.Mul(x3, y3)                               // Y3 := X3 * Y3
	x3.Mul(x3, t3)                               // X3 := X3 * t3
	t3.Add(t2, t2)                               // t3 := t2 + t2
	t2.Add(t2, t3)                               // t2 := t2 + t3
	z3.Mul(p256B(), z3)                          // Z3 := b * Z3
	z3.Sub(z3, t2)                               // Z3 := Z3 - t2
	z3.Sub(z3, t0)                               // Z3 := Z3 - t0
	t3.Add(z3, z3)                               // t3 := Z3 + Z3
	z3.Add(z3, t3)                               // Z3 := Z3 + t3
	t3.Add(t0, t0)                               // t3 := t0 + t0
	t0.Add(t3, t0)                               // t0 := t3 + t0
	t0.Sub(t0, t2)                               // t0 := t0 - t2
	t0.Mul(t0, z3)                               // t0 := t0 * Z3
	y3.Add(y3, t0)                               // Y3 := Y3 + t0
	t0.Mul(&p.y, &p.z)                           // t0 := Y * Z
	t0.Add(t0, t0)                               // t0 := t0 + t0
	z3.Mul(t0, z3)                               // Z3 := t0 * Z3
	x3.Sub(x3, z3)                               // X3 := X3 - Z3
	z3.Mul(t0, t1)                               // Z3 := t0 * t1
	z3.Add(z3, z3)                               // Z3 := Z3 + Z3
	z3.Add(z3, z3)                               // Z3 := Z3 + Z3

	q.x.Set(x3)
	q.y.Set(y3)
	q.z.Set(z3)
	return q
}

// p256AffinePoint is a point in affine coordinates (x, y). x and y are still
// Montgomery domain elements. The point can't be the point at infinity.
type p256AffinePoint struct {
	x, y fiat.P256Element
}

func (p *p256AffinePoint) Projective() *P256Point {
	pp := &P256Point{x: p.x, y: p.y}
	pp.z.One()
	return pp
}

// AddAffine sets q = p1 + p2, if infinity == 0, and to p1 if infinity == 1.
// p2 can't be the point at infinity as it can't be represented in affine
// coordinates, instead callers can set p2 to an arbitrary point and set
// infinity to 1.
func (q *P256Point) AddAffine(p1 *P256Point, p2 *p256AffinePoint, infinity int) *P256Point {
	// Complete mixed addition formula for a = -3 from "Complete addition
	// formulas for prime order elliptic curves"
	// (https://eprint.iacr.org/2015/1060), Algorithm 5.

	t0 := new(fiat.P256Element).Mul(&p1.x, &p2.x)   // t0 ← X1 · X2
	t1 := new(fiat.P256Element).Mul(&p1.y, &p2.y)   // t1 ← Y1 · Y2
	t3 := new(fiat.P256Element).Add(&p2.x, &p2.y)   // t3 ← X2 + Y2
	t4 := new(fiat.P256Element).Add(&p1.x, &p1.y)   // t4 ← X1 + Y1
	t3.Mul(t3, t4)                                  // t3 ← t3 · t4
	t4.Add(t0, t1)                                  // t4 ← t0 + t1
	t3.Sub(t3, t4)                                  // t3 ← t3 − t4
	t4.Mul(&p2.y, &p1.z)                            // t4 ← Y2 · Z1
	t4.Add(t4, &p1.y)                               // t4 ← t4 + Y1
	y3 := new(fiat.P256Element).Mul(&p2.x, &p1.z)   // Y3 ← X2 · Z1
	y3.Add(y3, &p1.x)                               // Y3 ← Y3 + X1
	z3 := new(fiat.P256Element).Mul(p256B(), &p1.z) // Z3 ← b  · Z1
	x3 := new(fiat.P256Element).Sub(y3, z3)         // X3 ← Y3 − Z3
	z3.Add(x3, x3)                                  // Z3 ← X3 + X3
	x3.Add(x3, z3)                                  // X3 ← X3 + Z3
	z3.Sub(t1, x3)                                  // Z3 ← t1 − X3
	x3.Add(t1, x3)                                  // X3 ← t1 + X3
	y3.Mul(p256B(), y3)                             // Y3 ← b  · Y3
	t1.Add(&p1.z, &p1.z)                            // t1 ← Z1 + Z1
	t2 := new(fiat.P256Element).Add(t1, &p1.z)      // t2 ← t1 + Z1
	y3.Sub(y3, t2)                                  // Y3 ← Y3 − t2
	y3.Sub(y3, t0)                                  // Y3 ← Y3 − t0
	t1.Add(y3, y3)                                  // t1 ← Y3 + Y3
	y3.Add(t1, y3)                                  // Y3 ← t1 + Y3
	t1.Add(t0, t0)                                  // t1 ← t0 + t0
	t0.Add(t1, t0)                                  // t0 ← t1 + t0
	t0.Sub(t0, t2)                                  // t0 ← t0 − t2
	t1.Mul(t4, y3)                                  // t1 ← t4 · Y3
	t2.Mul(t0, y3)                                  // t2 ← t0 · Y3
	y3.Mul(x3, z3)                                  // Y3 ← X3 · Z3
	y3.Add(y3, t2)                                  // Y3 ← Y3 + t2
	x3.Mul(t3, x3)                                  // X3 ← t3 · X3
	x3.Sub(x3, t1)                                  // X3 ← X3 − t1
	z3.Mul(t4, z3)                                  // Z3 ← t4 · Z3
	t1.Mul(t3, t0)                                  // t1 ← t3 · t0
	z3.Add(z3, t1)                                  // Z3 ← Z3 + t1

	q.x.Select(&p1.x, x3, infinity)
	q.y.Select(&p1.y, y3, infinity)
	q.z.Select(&p1.z, z3, infinity)
	return q
}

// Select sets q to p1 if cond == 1, and to p2 if cond == 0.
func (q *P256Point) Select(p1, p2 *P256Point, cond int) *P256Point {
	q.x.Select(&p1.x, &p2.x, cond)
	q.y.Select(&p1.y, &p2.y, cond)
	q.z.Select(&p1.z, &p2.z, cond)
	return q
}

// p256OrdElement is a P-256 scalar field element in [0, ord(G)-1] in the
// Montgomery domain (with R 2²⁵⁶) as four uint64 limbs in little-endian order.
type p256OrdElement [4]uint64

// SetBytes sets s to the big-endian value of x, reducing it as necessary.
func (s *p256OrdElement) SetBytes(x []byte) (*p256OrdElement, error) {
	if len(x) != 32 {
		return nil, errors.New("invalid scalar length")
	}

	s[0] = byteorder.BEUint64(x[24:])
	s[1] = byteorder.BEUint64(x[16:])
	s[2] = byteorder.BEUint64(x[8:])
	s[3] = byteorder.BEUint64(x[:])

	// Ensure s is in the range [0, ord(G)-1]. Since 2 * ord(G) > 2²⁵⁶, we can
	// just conditionally subtract ord(G), keeping the result if it doesn't
	// underflow.
	t0, b := bits.Sub64(s[0], 0xf3b9cac2fc632551, 0)
	t1, b := bits.Sub64(s[1], 0xbce6faada7179e84, b)
	t2, b := bits.Sub64(s[2], 0xffffffffffffffff, b)
	t3, b := bits.Sub64(s[3], 0xffffffff00000000, b)
	tMask := b - 1 // zero if subtraction underflowed
	s[0] ^= (t0 ^ s[0]) & tMask
	s[1] ^= (t1 ^ s[1]) & tMask
	s[2] ^= (t2 ^ s[2]) & tMask
	s[3] ^= (t3 ^ s[3]) & tMask

	return s, nil
}

func (s *p256OrdElement) Bytes() []byte {
	var out [32]byte
	byteorder.BEPutUint64(out[24:], s[0])
	byteorder.BEPutUint64(out[16:], s[1])
	byteorder.BEPutUint64(out[8:], s[2])
	byteorder.BEPutUint64(out[:], s[3])
	return out[:]
}

// Rsh returns the 64 least significant bits of x >> n. n must be lower
// than 256. The value of n leaks through timing side-channels.
func (s *p256OrdElement) Rsh(n int) uint64 {
	i := n / 64
	n = n % 64
	res := s[i] >> n
	// Shift in the more significant limb, if present.
	if i := i + 1; i < len(s) {
		res |= s[i] << (64 - n)
	}
	return res
}

// p256Table is a table of the first 16 multiples of a point. Points are stored
// at an index offset of -1 so [8]P is at index 7, P is at 0, and [16]P is at 15.
// [0]P is the point at infinity and it's not stored.
type p256Table [16]P256Point

// Select selects the n-th multiple of the table base point into p. It works in
// constant time. n must be in [0, 16]. If n is 0, p is set to the identity point.
func (table *p256Table) Select(p *P256Point, n uint8) {
	if n > 16 {
		panic("nistec: internal error: p256Table called with out-of-bounds value")
	}
	p.Set(NewP256Point())
	for i := uint8(1); i <= 16; i++ {
		cond := subtle.ConstantTimeByteEq(i, n)
		p.Select(&table[i-1], p, cond)
	}
}

// Compute populates the table to the first 16 multiples of q.
func (table *p256Table) Compute(q *P256Point) *p256Table {
	table[0].Set(q)
	for i := 1; i < 16; i += 2 {
		table[i].Double(&table[i/2])
		if i+1 < 16 {
			table[i+1].Add(&table[i], q)
		}
	}
	return table
}

func boothW5(in uint64) (uint8, int) {
	s := ^((in >> 5) - 1)
	d := (1 << 6) - in - 1
	d = (d & s) | (in & (^s))
	d = (d >> 1) + (d & 1)
	return uint8(d), int(s & 1)
}

// ScalarMult sets r = scalar * q, where scalar is a 32-byte big endian value,
// and returns r. If scalar is not 32 bytes long, ScalarMult returns an error
// and the receiver is unchanged.
func (p *P256Point) ScalarMult(q *P256Point, scalar []byte) (*P256Point, error) {
	s, err := new(p256OrdElement).SetBytes(scalar)
	if err != nil {
		return nil, err
	}

	// Start scanning the window from the most significant bits. We move by
	// 5 bits at a time and need to finish at -1, so -1 + 5 * 51 = 254.
	index := 254

	sel, sign := boothW5(s.Rsh(index))
	// sign is always zero because the boothW5 input here is at
	// most two bits long, so the top bit is never set.
	_ = sign

	// Neither Select nor Add have exceptions for the point at infinity /
	// selector zero, so we don't need to check for it here or in the loop.
	table := new(p256Table).Compute(q)
	table.Select(p, sel)

	t := NewP256Point()
	for index >= 4 {
		index -= 5

		p.Double(p)
		p.Double(p)
		p.Double(p)
		p.Double(p)
		p.Double(p)

		if index >= 0 {
			sel, sign = boothW5(s.Rsh(index) & 0b111111)
		} else {
			// Booth encoding considers a virtual zero bit at index -1,
			// so we shift left the least significant limb.
			wvalue := (s[0] << 1) & 0b111111
			sel, sign = boothW5(wvalue)
		}

		table.Select(t, sel)
		t.Negate(sign)
		p.Add(p, t)
	}

	return p, nil
}

// Negate sets p to -p, if cond == 1, and to p if cond == 0.
func (p *P256Point) Negate(cond int) *P256Point {
	negY := new(fiat.P256Element)
	negY.Sub(negY, &p.y)
	p.y.Select(negY, &p.y, cond)
	return p
}

// p256AffineTable is a table of the first 32 multiples of a point. Points are
// stored at an index offset of -1 like in p256Table, and [0]P is not stored.
type p256AffineTable [32]p256AffinePoint

// Select selects the n-th multiple of the table base point into p. It works in
// constant time. n can be in [0, 32], but (unlike p256Table.Select) if n is 0,
// p is set to an undefined value.
func (table *p256AffineTable) Select(p *p256AffinePoint, n uint8) {
	if n > 32 {
		panic("nistec: internal error: p256AffineTable.Select called with out-of-bounds value")
	}
	for i := uint8(1); i <= 32; i++ {
		cond := subtle.ConstantTimeByteEq(i, n)
		p.x.Select(&table[i-1].x, &p.x, cond)
		p.y.Select(&table[i-1].y, &p.y, cond)
	}
}

// p256GeneratorTables is a series of precomputed multiples of G, the canonical
// generator. The first p256AffineTable contains multiples of G. The second one
// multiples of [2⁶]G, the third one of [2¹²]G, and so on, where each successive
// table is the previous table doubled six times. Six is the width of the
// sliding window used in ScalarBaseMult, and having each table already
// pre-doubled lets us avoid the doublings between windows entirely. This table
// aliases into p256PrecomputedEmbed.
var p256GeneratorTables *[43]p256AffineTable

func init() {
	p256GeneratorTablesPtr := unsafe.Pointer(&p256PrecomputedEmbed)
	if cpu.BigEndian {
		var newTable [43 * 32 * 2 * 4]uint64
		for i, x := range (*[43 * 32 * 2 * 4][8]byte)(p256GeneratorTablesPtr) {
			newTable[i] = byteorder.LEUint64(x[:])
		}
		p256GeneratorTablesPtr = unsafe.Pointer(&newTable)
	}
	p256GeneratorTables = (*[43]p256AffineTable)(p256GeneratorTablesPtr)
}

func boothW6(in uint64) (uint8, int) {
	s := ^((in >> 6) - 1)
	d := (1 << 7) - in - 1
	d = (d & s) | (in & (^s))
	d = (d >> 1) + (d & 1)
	return uint8(d), int(s & 1)
}

// ScalarBaseMult sets p = scalar * generator, where scalar is a 32-byte big
// endian value, and returns r. If scalar is not 32 bytes long, ScalarBaseMult
// returns an error and the receiver is unchanged.
func (p *P256Point) ScalarBaseMult(scalar []byte) (*P256Point, error) {
	// This function works like ScalarMult above, but the table is fixed and
	// "pre-doubled" for each iteration, so instead of doubling we move to the
	// next table at each iteration.

	s, err := new(p256OrdElement).SetBytes(scalar)
	if err != nil {
		return nil, err
	}

	// Start scanning the window from the most significant bits. We move by
	// 6 bits at a time and need to finish at -1, so -1 + 6 * 42 = 251.
	index := 251

	sel, sign := boothW6(s.Rsh(index))
	// sign is always zero because the boothW6 input here is at
	// most five bits long, so the top bit is never set.
	_ = sign

	t := &p256AffinePoint{}
	table := &p256GeneratorTables[(index+1)/6]
	table.Select(t, sel)

	// Select's output is undefined if the selector is zero, when it should be
	// the point at infinity (because infinity can't be represented in affine
	// coordinates). Here we conditionally set p to the infinity if sel is zero.
	// In the loop, that's handled by AddAffine.
	selIsZero := subtle.ConstantTimeByteEq(sel, 0)
	p.Select(NewP256Point(), t.Projective(), selIsZero)

	for index >= 5 {
		index -= 6

		if index >= 0 {
			sel, sign = boothW6(s.Rsh(index) & 0b1111111)
		} else {
			// Booth encoding considers a virtual zero bit at index -1,
			// so we shift left the least significant limb.
			wvalue := (s[0] << 1) & 0b1111111
			sel, sign = boothW6(wvalue)
		}

		table := &p256GeneratorTables[(index+1)/6]
		table.Select(t, sel)
		t.Negate(sign)
		selIsZero := subtle.ConstantTimeByteEq(sel, 0)
		p.AddAffine(p, t, selIsZero)
	}

	return p, nil
}

// Negate sets p to -p, if cond == 1, and to p if cond == 0.
func (p *p256AffinePoint) Negate(cond int) *p256AffinePoint {
	negY := new(fiat.P256Element)
	negY.Sub(negY, &p.y)
	p.y.Select(negY, &p.y, cond)
	return p
}

// p256Sqrt sets e to a square root of x. If x is not a square, p256Sqrt returns
// false and e is unchanged. e and x can overlap.
func p256Sqrt(e, x *fiat.P256Element) (isSquare bool) {
	t0, t1 := new(fiat.P256Element), new(fiat.P256Element)

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
	p256Square(t0, x, 1)
	t0.Mul(x, t0)
	p256Square(t1, t0, 2)
	t0.Mul(t0, t1)
	p256Square(t1, t0, 4)
	t0.Mul(t0, t1)
	p256Square(t1, t0, 8)
	t0.Mul(t0, t1)
	p256Square(t1, t0, 16)
	t0.Mul(t0, t1)
	p256Square(t0, t0, 32)
	t0.Mul(x, t0)
	p256Square(t0, t0, 96)
	t0.Mul(x, t0)
	p256Square(t0, t0, 94)

	// Check if the candidate t0 is indeed a square root of x.
	t1.Square(t0)
	if t1.Equal(x) != 1 {
		return false
	}
	e.Set(t0)
	return true
}

// p256Square sets e to the square of x, repeated n times > 1.
func p256Square(e, x *fiat.P256Element, n int) {
	e.Square(x)
	for i := 1; i < n; i++ {
		e.Square(e)
	}
}

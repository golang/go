// Copyright (c) 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package edwards25519

import (
	"crypto/internal/edwards25519/field"
	"errors"
)

// Point types.

type projP1xP1 struct {
	X, Y, Z, T field.Element
}

type projP2 struct {
	X, Y, Z field.Element
}

// Point represents a point on the edwards25519 curve.
//
// This type works similarly to math/big.Int, and all arguments and receivers
// are allowed to alias.
//
// The zero value is NOT valid, and it may be used only as a receiver.
type Point struct {
	// Make the type not comparable (i.e. used with == or as a map key), as
	// equivalent points can be represented by different Go values.
	_ incomparable

	// The point is internally represented in extended coordinates (X, Y, Z, T)
	// where x = X/Z, y = Y/Z, and xy = T/Z per https://eprint.iacr.org/2008/522.
	x, y, z, t field.Element
}

type incomparable [0]func()

func checkInitialized(points ...*Point) {
	for _, p := range points {
		if p.x == (field.Element{}) && p.y == (field.Element{}) {
			panic("edwards25519: use of uninitialized Point")
		}
	}
}

type projCached struct {
	YplusX, YminusX, Z, T2d field.Element
}

type affineCached struct {
	YplusX, YminusX, T2d field.Element
}

// Constructors.

func (v *projP2) Zero() *projP2 {
	v.X.Zero()
	v.Y.One()
	v.Z.One()
	return v
}

// identity is the point at infinity.
var identity, _ = new(Point).SetBytes([]byte{
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})

// NewIdentityPoint returns a new Point set to the identity.
func NewIdentityPoint() *Point {
	return new(Point).Set(identity)
}

// generator is the canonical curve basepoint. See TestGenerator for the
// correspondence of this encoding with the values in RFC 8032.
var generator, _ = new(Point).SetBytes([]byte{
	0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
	0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
	0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
	0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66})

// NewGeneratorPoint returns a new Point set to the canonical generator.
func NewGeneratorPoint() *Point {
	return new(Point).Set(generator)
}

func (v *projCached) Zero() *projCached {
	v.YplusX.One()
	v.YminusX.One()
	v.Z.One()
	v.T2d.Zero()
	return v
}

func (v *affineCached) Zero() *affineCached {
	v.YplusX.One()
	v.YminusX.One()
	v.T2d.Zero()
	return v
}

// Assignments.

// Set sets v = u, and returns v.
func (v *Point) Set(u *Point) *Point {
	*v = *u
	return v
}

// Encoding.

// Bytes returns the canonical 32-byte encoding of v, according to RFC 8032,
// Section 5.1.2.
func (v *Point) Bytes() []byte {
	// This function is outlined to make the allocations inline in the caller
	// rather than happen on the heap.
	var buf [32]byte
	return v.bytes(&buf)
}

func (v *Point) bytes(buf *[32]byte) []byte {
	checkInitialized(v)

	var zInv, x, y field.Element
	zInv.Invert(&v.z)       // zInv = 1 / Z
	x.Multiply(&v.x, &zInv) // x = X / Z
	y.Multiply(&v.y, &zInv) // y = Y / Z

	out := copyFieldElement(buf, &y)
	out[31] |= byte(x.IsNegative() << 7)
	return out
}

var feOne = new(field.Element).One()

// SetBytes sets v = x, where x is a 32-byte encoding of v. If x does not
// represent a valid point on the curve, SetBytes returns nil and an error and
// the receiver is unchanged. Otherwise, SetBytes returns v.
//
// Note that SetBytes accepts all non-canonical encodings of valid points.
// That is, it follows decoding rules that match most implementations in
// the ecosystem rather than RFC 8032.
func (v *Point) SetBytes(x []byte) (*Point, error) {
	// Specifically, the non-canonical encodings that are accepted are
	//   1) the ones where the field element is not reduced (see the
	//      (*field.Element).SetBytes docs) and
	//   2) the ones where the x-coordinate is zero and the sign bit is set.
	//
	// Read more at https://hdevalence.ca/blog/2020-10-04-its-25519am,
	// specifically the "Canonical A, R" section.

	y, err := new(field.Element).SetBytes(x)
	if err != nil {
		return nil, errors.New("edwards25519: invalid point encoding length")
	}

	// -x² + y² = 1 + dx²y²
	// x² + dx²y² = x²(dy² + 1) = y² - 1
	// x² = (y² - 1) / (dy² + 1)

	// u = y² - 1
	y2 := new(field.Element).Square(y)
	u := new(field.Element).Subtract(y2, feOne)

	// v = dy² + 1
	vv := new(field.Element).Multiply(y2, d)
	vv = vv.Add(vv, feOne)

	// x = +√(u/v)
	xx, wasSquare := new(field.Element).SqrtRatio(u, vv)
	if wasSquare == 0 {
		return nil, errors.New("edwards25519: invalid point encoding")
	}

	// Select the negative square root if the sign bit is set.
	xxNeg := new(field.Element).Negate(xx)
	xx = xx.Select(xxNeg, xx, int(x[31]>>7))

	v.x.Set(xx)
	v.y.Set(y)
	v.z.One()
	v.t.Multiply(xx, y) // xy = T / Z

	return v, nil
}

func copyFieldElement(buf *[32]byte, v *field.Element) []byte {
	copy(buf[:], v.Bytes())
	return buf[:]
}

// Conversions.

func (v *projP2) FromP1xP1(p *projP1xP1) *projP2 {
	v.X.Multiply(&p.X, &p.T)
	v.Y.Multiply(&p.Y, &p.Z)
	v.Z.Multiply(&p.Z, &p.T)
	return v
}

func (v *projP2) FromP3(p *Point) *projP2 {
	v.X.Set(&p.x)
	v.Y.Set(&p.y)
	v.Z.Set(&p.z)
	return v
}

func (v *Point) fromP1xP1(p *projP1xP1) *Point {
	v.x.Multiply(&p.X, &p.T)
	v.y.Multiply(&p.Y, &p.Z)
	v.z.Multiply(&p.Z, &p.T)
	v.t.Multiply(&p.X, &p.Y)
	return v
}

func (v *Point) fromP2(p *projP2) *Point {
	v.x.Multiply(&p.X, &p.Z)
	v.y.Multiply(&p.Y, &p.Z)
	v.z.Square(&p.Z)
	v.t.Multiply(&p.X, &p.Y)
	return v
}

// d is a constant in the curve equation.
var d, _ = new(field.Element).SetBytes([]byte{
	0xa3, 0x78, 0x59, 0x13, 0xca, 0x4d, 0xeb, 0x75,
	0xab, 0xd8, 0x41, 0x41, 0x4d, 0x0a, 0x70, 0x00,
	0x98, 0xe8, 0x79, 0x77, 0x79, 0x40, 0xc7, 0x8c,
	0x73, 0xfe, 0x6f, 0x2b, 0xee, 0x6c, 0x03, 0x52})
var d2 = new(field.Element).Add(d, d)

func (v *projCached) FromP3(p *Point) *projCached {
	v.YplusX.Add(&p.y, &p.x)
	v.YminusX.Subtract(&p.y, &p.x)
	v.Z.Set(&p.z)
	v.T2d.Multiply(&p.t, d2)
	return v
}

func (v *affineCached) FromP3(p *Point) *affineCached {
	v.YplusX.Add(&p.y, &p.x)
	v.YminusX.Subtract(&p.y, &p.x)
	v.T2d.Multiply(&p.t, d2)

	var invZ field.Element
	invZ.Invert(&p.z)
	v.YplusX.Multiply(&v.YplusX, &invZ)
	v.YminusX.Multiply(&v.YminusX, &invZ)
	v.T2d.Multiply(&v.T2d, &invZ)
	return v
}

// (Re)addition and subtraction.

// Add sets v = p + q, and returns v.
func (v *Point) Add(p, q *Point) *Point {
	checkInitialized(p, q)
	qCached := new(projCached).FromP3(q)
	result := new(projP1xP1).Add(p, qCached)
	return v.fromP1xP1(result)
}

// Subtract sets v = p - q, and returns v.
func (v *Point) Subtract(p, q *Point) *Point {
	checkInitialized(p, q)
	qCached := new(projCached).FromP3(q)
	result := new(projP1xP1).Sub(p, qCached)
	return v.fromP1xP1(result)
}

func (v *projP1xP1) Add(p *Point, q *projCached) *projP1xP1 {
	var YplusX, YminusX, PP, MM, TT2d, ZZ2 field.Element

	YplusX.Add(&p.y, &p.x)
	YminusX.Subtract(&p.y, &p.x)

	PP.Multiply(&YplusX, &q.YplusX)
	MM.Multiply(&YminusX, &q.YminusX)
	TT2d.Multiply(&p.t, &q.T2d)
	ZZ2.Multiply(&p.z, &q.Z)

	ZZ2.Add(&ZZ2, &ZZ2)

	v.X.Subtract(&PP, &MM)
	v.Y.Add(&PP, &MM)
	v.Z.Add(&ZZ2, &TT2d)
	v.T.Subtract(&ZZ2, &TT2d)
	return v
}

func (v *projP1xP1) Sub(p *Point, q *projCached) *projP1xP1 {
	var YplusX, YminusX, PP, MM, TT2d, ZZ2 field.Element

	YplusX.Add(&p.y, &p.x)
	YminusX.Subtract(&p.y, &p.x)

	PP.Multiply(&YplusX, &q.YminusX) // flipped sign
	MM.Multiply(&YminusX, &q.YplusX) // flipped sign
	TT2d.Multiply(&p.t, &q.T2d)
	ZZ2.Multiply(&p.z, &q.Z)

	ZZ2.Add(&ZZ2, &ZZ2)

	v.X.Subtract(&PP, &MM)
	v.Y.Add(&PP, &MM)
	v.Z.Subtract(&ZZ2, &TT2d) // flipped sign
	v.T.Add(&ZZ2, &TT2d)      // flipped sign
	return v
}

func (v *projP1xP1) AddAffine(p *Point, q *affineCached) *projP1xP1 {
	var YplusX, YminusX, PP, MM, TT2d, Z2 field.Element

	YplusX.Add(&p.y, &p.x)
	YminusX.Subtract(&p.y, &p.x)

	PP.Multiply(&YplusX, &q.YplusX)
	MM.Multiply(&YminusX, &q.YminusX)
	TT2d.Multiply(&p.t, &q.T2d)

	Z2.Add(&p.z, &p.z)

	v.X.Subtract(&PP, &MM)
	v.Y.Add(&PP, &MM)
	v.Z.Add(&Z2, &TT2d)
	v.T.Subtract(&Z2, &TT2d)
	return v
}

func (v *projP1xP1) SubAffine(p *Point, q *affineCached) *projP1xP1 {
	var YplusX, YminusX, PP, MM, TT2d, Z2 field.Element

	YplusX.Add(&p.y, &p.x)
	YminusX.Subtract(&p.y, &p.x)

	PP.Multiply(&YplusX, &q.YminusX) // flipped sign
	MM.Multiply(&YminusX, &q.YplusX) // flipped sign
	TT2d.Multiply(&p.t, &q.T2d)

	Z2.Add(&p.z, &p.z)

	v.X.Subtract(&PP, &MM)
	v.Y.Add(&PP, &MM)
	v.Z.Subtract(&Z2, &TT2d) // flipped sign
	v.T.Add(&Z2, &TT2d)      // flipped sign
	return v
}

// Doubling.

func (v *projP1xP1) Double(p *projP2) *projP1xP1 {
	var XX, YY, ZZ2, XplusYsq field.Element

	XX.Square(&p.X)
	YY.Square(&p.Y)
	ZZ2.Square(&p.Z)
	ZZ2.Add(&ZZ2, &ZZ2)
	XplusYsq.Add(&p.X, &p.Y)
	XplusYsq.Square(&XplusYsq)

	v.Y.Add(&YY, &XX)
	v.Z.Subtract(&YY, &XX)

	v.X.Subtract(&XplusYsq, &v.Y)
	v.T.Subtract(&ZZ2, &v.Z)
	return v
}

// Negation.

// Negate sets v = -p, and returns v.
func (v *Point) Negate(p *Point) *Point {
	checkInitialized(p)
	v.x.Negate(&p.x)
	v.y.Set(&p.y)
	v.z.Set(&p.z)
	v.t.Negate(&p.t)
	return v
}

// Equal returns 1 if v is equivalent to u, and 0 otherwise.
func (v *Point) Equal(u *Point) int {
	checkInitialized(v, u)

	var t1, t2, t3, t4 field.Element
	t1.Multiply(&v.x, &u.z)
	t2.Multiply(&u.x, &v.z)
	t3.Multiply(&v.y, &u.z)
	t4.Multiply(&u.y, &v.z)

	return t1.Equal(&t2) & t3.Equal(&t4)
}

// Constant-time operations

// Select sets v to a if cond == 1 and to b if cond == 0.
func (v *projCached) Select(a, b *projCached, cond int) *projCached {
	v.YplusX.Select(&a.YplusX, &b.YplusX, cond)
	v.YminusX.Select(&a.YminusX, &b.YminusX, cond)
	v.Z.Select(&a.Z, &b.Z, cond)
	v.T2d.Select(&a.T2d, &b.T2d, cond)
	return v
}

// Select sets v to a if cond == 1 and to b if cond == 0.
func (v *affineCached) Select(a, b *affineCached, cond int) *affineCached {
	v.YplusX.Select(&a.YplusX, &b.YplusX, cond)
	v.YminusX.Select(&a.YminusX, &b.YminusX, cond)
	v.T2d.Select(&a.T2d, &b.T2d, cond)
	return v
}

// CondNeg negates v if cond == 1 and leaves it unchanged if cond == 0.
func (v *projCached) CondNeg(cond int) *projCached {
	v.YplusX.Swap(&v.YminusX, cond)
	v.T2d.Select(new(field.Element).Negate(&v.T2d), &v.T2d, cond)
	return v
}

// CondNeg negates v if cond == 1 and leaves it unchanged if cond == 0.
func (v *affineCached) CondNeg(cond int) *affineCached {
	v.YplusX.Swap(&v.YminusX, cond)
	v.T2d.Select(new(field.Element).Negate(&v.T2d), &v.T2d, cond)
	return v
}

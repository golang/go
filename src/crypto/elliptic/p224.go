// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"crypto/elliptic/internal/nistec"
	"crypto/rand"
	"math/big"
)

// p224Curve is a Curve implementation based on nistec.P224Point.
//
// It's a wrapper that exposes the big.Int-based Curve interface and encodes the
// legacy idiosyncrasies it requires, such as invalid and infinity point
// handling.
//
// To interact with the nistec package, points are encoded into and decoded from
// properly formatted byte slices. All big.Int use is limited to this package.
// Encoding and decoding is 1/1000th of the runtime of a scalar multiplication,
// so the overhead is acceptable.
type p224Curve struct {
	params *CurveParams
}

var p224 p224Curve
var _ Curve = p224

func initP224() {
	p224.params = &CurveParams{
		Name:    "P-224",
		BitSize: 224,
		// FIPS 186-4, section D.1.2.2
		P:  bigFromDecimal("26959946667150639794667015087019630673557916260026308143510066298881"),
		N:  bigFromDecimal("26959946667150639794667015087019625940457807714424391721682722368061"),
		B:  bigFromHex("b4050a850c04b3abf54132565044b0b7d7bfd8ba270b39432355ffb4"),
		Gx: bigFromHex("b70e0cbd6bb4bf7f321390b94a03c1d356c21122343280d6115c1d21"),
		Gy: bigFromHex("bd376388b5f723fb4c22dfe6cd4375a05a07476444d5819985007e34"),
	}
}

func (curve p224Curve) Params() *CurveParams {
	return curve.params
}

func (curve p224Curve) IsOnCurve(x, y *big.Int) bool {
	// IsOnCurve is documented to reject (0, 0), the conventional point at
	// infinity, which however is accepted by p224PointFromAffine.
	if x.Sign() == 0 && y.Sign() == 0 {
		return false
	}
	_, ok := p224PointFromAffine(x, y)
	return ok
}

func p224PointFromAffine(x, y *big.Int) (p *nistec.P224Point, ok bool) {
	// (0, 0) is by convention the point at infinity, which can't be represented
	// in affine coordinates. Marshal incorrectly encodes it as an uncompressed
	// point, which SetBytes would correctly reject. See Issue 37294.
	if x.Sign() == 0 && y.Sign() == 0 {
		return nistec.NewP224Point(), true
	}
	if x.Sign() < 0 || y.Sign() < 0 {
		return nil, false
	}
	if x.BitLen() > 224 || y.BitLen() > 224 {
		return nil, false
	}
	p, err := nistec.NewP224Point().SetBytes(Marshal(P224(), x, y))
	if err != nil {
		return nil, false
	}
	return p, true
}

func p224PointToAffine(p *nistec.P224Point) (x, y *big.Int) {
	out := p.Bytes()
	if len(out) == 1 && out[0] == 0 {
		// This is the correct encoding of the point at infinity, which
		// Unmarshal does not support. See Issue 37294.
		return new(big.Int), new(big.Int)
	}
	x, y = Unmarshal(P224(), out)
	if x == nil {
		panic("crypto/elliptic: internal error: Unmarshal rejected a valid point encoding")
	}
	return x, y
}

// p224RandomPoint returns a random point on the curve. It's used when Add,
// Double, or ScalarMult are fed a point not on the curve, which is undefined
// behavior. Originally, we used to do the math on it anyway (which allows
// invalid curve attacks) and relied on the caller and Unmarshal to avoid this
// happening in the first place. Now, we just can't construct a nistec.P224Point
// for an invalid pair of coordinates, because that API is safer. If we panic,
// we risk introducing a DoS. If we return nil, we risk a panic. If we return
// the input, ecdsa.Verify might fail open. The safest course seems to be to
// return a valid, random point, which hopefully won't help the attacker.
func p224RandomPoint() (x, y *big.Int) {
	_, x, y, err := GenerateKey(P224(), rand.Reader)
	if err != nil {
		panic("crypto/elliptic: failed to generate random point")
	}
	return x, y
}

func (p224Curve) Add(x1, y1, x2, y2 *big.Int) (*big.Int, *big.Int) {
	p1, ok := p224PointFromAffine(x1, y1)
	if !ok {
		return p224RandomPoint()
	}
	p2, ok := p224PointFromAffine(x2, y2)
	if !ok {
		return p224RandomPoint()
	}
	return p224PointToAffine(p1.Add(p1, p2))
}

func (p224Curve) Double(x1, y1 *big.Int) (*big.Int, *big.Int) {
	p, ok := p224PointFromAffine(x1, y1)
	if !ok {
		return p224RandomPoint()
	}
	return p224PointToAffine(p.Double(p))
}

func (p224Curve) ScalarMult(Bx, By *big.Int, scalar []byte) (*big.Int, *big.Int) {
	p, ok := p224PointFromAffine(Bx, By)
	if !ok {
		return p224RandomPoint()
	}
	return p224PointToAffine(p.ScalarMult(p, scalar))
}

func (p224Curve) ScalarBaseMult(scalar []byte) (*big.Int, *big.Int) {
	p := nistec.NewP224Generator()
	return p224PointToAffine(p.ScalarMult(p, scalar))
}

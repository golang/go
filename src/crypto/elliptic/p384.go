// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"crypto/elliptic/internal/nistec"
	"crypto/rand"
	"math/big"
)

// p384Curve is a Curve implementation based on nistec.P384Point.
//
// It's a wrapper that exposes the big.Int-based Curve interface and encodes the
// legacy idiosyncrasies it requires, such as invalid and infinity point
// handling.
//
// To interact with the nistec package, points are encoded into and decoded from
// properly formatted byte slices. All big.Int use is limited to this package.
// Encoding and decoding is 1/1000th of the runtime of a scalar multiplication,
// so the overhead is acceptable.
type p384Curve struct {
	params *CurveParams
}

var p384 p384Curve
var _ Curve = p384

func initP384() {
	p384.params = &CurveParams{
		Name:    "P-384",
		BitSize: 384,
		// FIPS 186-4, section D.1.2.4
		P: bigFromDecimal("394020061963944792122790401001436138050797392704654" +
			"46667948293404245721771496870329047266088258938001861606973112319"),
		N: bigFromDecimal("394020061963944792122790401001436138050797392704654" +
			"46667946905279627659399113263569398956308152294913554433653942643"),
		B: bigFromHex("b3312fa7e23ee7e4988e056be3f82d19181d9c6efe8141120314088" +
			"f5013875ac656398d8a2ed19d2a85c8edd3ec2aef"),
		Gx: bigFromHex("aa87ca22be8b05378eb1c71ef320ad746e1d3b628ba79b9859f741" +
			"e082542a385502f25dbf55296c3a545e3872760ab7"),
		Gy: bigFromHex("3617de4a96262c6f5d9e98bf9292dc29f8f41dbd289a147ce9da31" +
			"13b5f0b8c00a60b1ce1d7e819d7a431d7c90ea0e5f"),
	}
}

func (curve p384Curve) Params() *CurveParams {
	return curve.params
}

func (curve p384Curve) IsOnCurve(x, y *big.Int) bool {
	// IsOnCurve is documented to reject (0, 0), the conventional point at
	// infinity, which however is accepted by p384PointFromAffine.
	if x.Sign() == 0 && y.Sign() == 0 {
		return false
	}
	_, ok := p384PointFromAffine(x, y)
	return ok
}

func p384PointFromAffine(x, y *big.Int) (p *nistec.P384Point, ok bool) {
	// (0, 0) is by convention the point at infinity, which can't be represented
	// in affine coordinates. Marshal incorrectly encodes it as an uncompressed
	// point, which SetBytes would correctly reject. See Issue 37294.
	if x.Sign() == 0 && y.Sign() == 0 {
		return nistec.NewP384Point(), true
	}
	if x.Sign() < 0 || y.Sign() < 0 {
		return nil, false
	}
	if x.BitLen() > 384 || y.BitLen() > 384 {
		return nil, false
	}
	p, err := nistec.NewP384Point().SetBytes(Marshal(P384(), x, y))
	if err != nil {
		return nil, false
	}
	return p, true
}

func p384PointToAffine(p *nistec.P384Point) (x, y *big.Int) {
	out := p.Bytes()
	if len(out) == 1 && out[0] == 0 {
		// This is the correct encoding of the point at infinity, which
		// Unmarshal does not support. See Issue 37294.
		return new(big.Int), new(big.Int)
	}
	x, y = Unmarshal(P384(), out)
	if x == nil {
		panic("crypto/elliptic: internal error: Unmarshal rejected a valid point encoding")
	}
	return x, y
}

// p384RandomPoint returns a random point on the curve. It's used when Add,
// Double, or ScalarMult are fed a point not on the curve, which is undefined
// behavior. Originally, we used to do the math on it anyway (which allows
// invalid curve attacks) and relied on the caller and Unmarshal to avoid this
// happening in the first place. Now, we just can't construct a nistec.P384Point
// for an invalid pair of coordinates, because that API is safer. If we panic,
// we risk introducing a DoS. If we return nil, we risk a panic. If we return
// the input, ecdsa.Verify might fail open. The safest course seems to be to
// return a valid, random point, which hopefully won't help the attacker.
func p384RandomPoint() (x, y *big.Int) {
	_, x, y, err := GenerateKey(P384(), rand.Reader)
	if err != nil {
		panic("crypto/elliptic: failed to generate random point")
	}
	return x, y
}

func (p384Curve) Add(x1, y1, x2, y2 *big.Int) (*big.Int, *big.Int) {
	p1, ok := p384PointFromAffine(x1, y1)
	if !ok {
		return p384RandomPoint()
	}
	p2, ok := p384PointFromAffine(x2, y2)
	if !ok {
		return p384RandomPoint()
	}
	return p384PointToAffine(p1.Add(p1, p2))
}

func (p384Curve) Double(x1, y1 *big.Int) (*big.Int, *big.Int) {
	p, ok := p384PointFromAffine(x1, y1)
	if !ok {
		return p384RandomPoint()
	}
	return p384PointToAffine(p.Double(p))
}

func (p384Curve) ScalarMult(Bx, By *big.Int, scalar []byte) (*big.Int, *big.Int) {
	p, ok := p384PointFromAffine(Bx, By)
	if !ok {
		return p384RandomPoint()
	}
	return p384PointToAffine(p.ScalarMult(p, scalar))
}

func (p384Curve) ScalarBaseMult(scalar []byte) (*big.Int, *big.Int) {
	p := nistec.NewP384Generator()
	return p384PointToAffine(p.ScalarMult(p, scalar))
}

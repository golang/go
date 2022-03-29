// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"crypto/elliptic/internal/nistec"
	"crypto/rand"
	"math/big"
)

// p521Curve is a Curve implementation based on nistec.P521Point.
//
// It's a wrapper that exposes the big.Int-based Curve interface and encodes the
// legacy idiosyncrasies it requires, such as invalid and infinity point
// handling.
//
// To interact with the nistec package, points are encoded into and decoded from
// properly formatted byte slices. All big.Int use is limited to this package.
// Encoding and decoding is 1/1000th of the runtime of a scalar multiplication,
// so the overhead is acceptable.
type p521Curve struct {
	params *CurveParams
}

var p521 p521Curve
var _ Curve = p521

func initP521() {
	p521.params = &CurveParams{
		Name:    "P-521",
		BitSize: 521,
		// FIPS 186-4, section D.1.2.5
		P: bigFromDecimal("68647976601306097149819007990813932172694353001433" +
			"0540939446345918554318339765605212255964066145455497729631139148" +
			"0858037121987999716643812574028291115057151"),
		N: bigFromDecimal("68647976601306097149819007990813932172694353001433" +
			"0540939446345918554318339765539424505774633321719753296399637136" +
			"3321113864768612440380340372808892707005449"),
		B: bigFromHex("0051953eb9618e1c9a1f929a21a0b68540eea2da725b99b315f3b8" +
			"b489918ef109e156193951ec7e937b1652c0bd3bb1bf073573df883d2c34f1ef" +
			"451fd46b503f00"),
		Gx: bigFromHex("00c6858e06b70404e9cd9e3ecb662395b4429c648139053fb521f8" +
			"28af606b4d3dbaa14b5e77efe75928fe1dc127a2ffa8de3348b3c1856a429bf9" +
			"7e7e31c2e5bd66"),
		Gy: bigFromHex("011839296a789a3bc0045c8a5fb42c7d1bd998f54449579b446817" +
			"afbd17273e662c97ee72995ef42640c550b9013fad0761353c7086a272c24088" +
			"be94769fd16650"),
	}
}

func (curve p521Curve) Params() *CurveParams {
	return curve.params
}

func (curve p521Curve) IsOnCurve(x, y *big.Int) bool {
	// IsOnCurve is documented to reject (0, 0), the conventional point at
	// infinity, which however is accepted by p521PointFromAffine.
	if x.Sign() == 0 && y.Sign() == 0 {
		return false
	}
	_, ok := p521PointFromAffine(x, y)
	return ok
}

func p521PointFromAffine(x, y *big.Int) (p *nistec.P521Point, ok bool) {
	// (0, 0) is by convention the point at infinity, which can't be represented
	// in affine coordinates. Marshal incorrectly encodes it as an uncompressed
	// point, which SetBytes would correctly reject. See Issue 37294.
	if x.Sign() == 0 && y.Sign() == 0 {
		return nistec.NewP521Point(), true
	}
	if x.Sign() < 0 || y.Sign() < 0 {
		return nil, false
	}
	if x.BitLen() > 521 || y.BitLen() > 521 {
		return nil, false
	}
	p, err := nistec.NewP521Point().SetBytes(Marshal(P521(), x, y))
	if err != nil {
		return nil, false
	}
	return p, true
}

func p521PointToAffine(p *nistec.P521Point) (x, y *big.Int) {
	out := p.Bytes()
	if len(out) == 1 && out[0] == 0 {
		// This is the correct encoding of the point at infinity, which
		// Unmarshal does not support. See Issue 37294.
		return new(big.Int), new(big.Int)
	}
	x, y = Unmarshal(P521(), out)
	if x == nil {
		panic("crypto/elliptic: internal error: Unmarshal rejected a valid point encoding")
	}
	return x, y
}

// p521RandomPoint returns a random point on the curve. It's used when Add,
// Double, or ScalarMult are fed a point not on the curve, which is undefined
// behavior. Originally, we used to do the math on it anyway (which allows
// invalid curve attacks) and relied on the caller and Unmarshal to avoid this
// happening in the first place. Now, we just can't construct a nistec.P521Point
// for an invalid pair of coordinates, because that API is safer. If we panic,
// we risk introducing a DoS. If we return nil, we risk a panic. If we return
// the input, ecdsa.Verify might fail open. The safest course seems to be to
// return a valid, random point, which hopefully won't help the attacker.
func p521RandomPoint() (x, y *big.Int) {
	_, x, y, err := GenerateKey(P521(), rand.Reader)
	if err != nil {
		panic("crypto/elliptic: failed to generate random point")
	}
	return x, y
}

func (p521Curve) Add(x1, y1, x2, y2 *big.Int) (*big.Int, *big.Int) {
	p1, ok := p521PointFromAffine(x1, y1)
	if !ok {
		return p521RandomPoint()
	}
	p2, ok := p521PointFromAffine(x2, y2)
	if !ok {
		return p521RandomPoint()
	}
	return p521PointToAffine(p1.Add(p1, p2))
}

func (p521Curve) Double(x1, y1 *big.Int) (*big.Int, *big.Int) {
	p, ok := p521PointFromAffine(x1, y1)
	if !ok {
		return p521RandomPoint()
	}
	return p521PointToAffine(p.Double(p))
}

func (p521Curve) ScalarMult(Bx, By *big.Int, scalar []byte) (*big.Int, *big.Int) {
	p, ok := p521PointFromAffine(Bx, By)
	if !ok {
		return p521RandomPoint()
	}
	return p521PointToAffine(p.ScalarMult(p, scalar))
}

func (p521Curve) ScalarBaseMult(scalar []byte) (*big.Int, *big.Int) {
	p := nistec.NewP521Generator()
	return p521PointToAffine(p.ScalarMult(p, scalar))
}

func bigFromDecimal(s string) *big.Int {
	b, ok := new(big.Int).SetString(s, 10)
	if !ok {
		panic("invalid encoding")
	}
	return b
}

func bigFromHex(s string) *big.Int {
	b, ok := new(big.Int).SetString(s, 16)
	if !ok {
		panic("invalid encoding")
	}
	return b
}

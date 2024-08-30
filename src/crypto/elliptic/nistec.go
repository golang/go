// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"crypto/internal/nistec"
	"errors"
	"math/big"
)

var p224 = &nistCurve[*nistec.P224Point]{
	newPoint: nistec.NewP224Point,
}

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

type p256Curve struct {
	nistCurve[*nistec.P256Point]
}

var p256 = &p256Curve{nistCurve[*nistec.P256Point]{
	newPoint: nistec.NewP256Point,
}}

func initP256() {
	p256.params = &CurveParams{
		Name:    "P-256",
		BitSize: 256,
		// FIPS 186-4, section D.1.2.3
		P:  bigFromDecimal("115792089210356248762697446949407573530086143415290314195533631308867097853951"),
		N:  bigFromDecimal("115792089210356248762697446949407573529996955224135760342422259061068512044369"),
		B:  bigFromHex("5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b"),
		Gx: bigFromHex("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"),
		Gy: bigFromHex("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5"),
	}
}

var p384 = &nistCurve[*nistec.P384Point]{
	newPoint: nistec.NewP384Point,
}

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

var p521 = &nistCurve[*nistec.P521Point]{
	newPoint: nistec.NewP521Point,
}

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

// nistCurve is a Curve implementation based on a nistec Point.
//
// It's a wrapper that exposes the big.Int-based Curve interface and encodes the
// legacy idiosyncrasies it requires, such as invalid and infinity point
// handling.
//
// To interact with the nistec package, points are encoded into and decoded from
// properly formatted byte slices. All big.Int use is limited to this package.
// Encoding and decoding is 1/1000th of the runtime of a scalar multiplication,
// so the overhead is acceptable.
type nistCurve[Point nistPoint[Point]] struct {
	newPoint func() Point
	params   *CurveParams
}

// nistPoint is a generic constraint for the nistec Point types.
type nistPoint[T any] interface {
	Bytes() []byte
	SetBytes([]byte) (T, error)
	Add(T, T) T
	Double(T) T
	ScalarMult(T, []byte) (T, error)
	ScalarBaseMult([]byte) (T, error)
}

func (curve *nistCurve[Point]) Params() *CurveParams {
	return curve.params
}

func (curve *nistCurve[Point]) IsOnCurve(x, y *big.Int) bool {
	// IsOnCurve is documented to reject (0, 0), the conventional point at
	// infinity, which however is accepted by pointFromAffine.
	if x.Sign() == 0 && y.Sign() == 0 {
		return false
	}
	_, err := curve.pointFromAffine(x, y)
	return err == nil
}

func (curve *nistCurve[Point]) pointFromAffine(x, y *big.Int) (p Point, err error) {
	// (0, 0) is by convention the point at infinity, which can't be represented
	// in affine coordinates. See Issue 37294.
	if x.Sign() == 0 && y.Sign() == 0 {
		return curve.newPoint(), nil
	}
	// Reject values that would not get correctly encoded.
	if x.Sign() < 0 || y.Sign() < 0 {
		return p, errors.New("negative coordinate")
	}
	if x.BitLen() > curve.params.BitSize || y.BitLen() > curve.params.BitSize {
		return p, errors.New("overflowing coordinate")
	}
	// Encode the coordinates and let SetBytes reject invalid points.
	byteLen := (curve.params.BitSize + 7) / 8
	buf := make([]byte, 1+2*byteLen)
	buf[0] = 4 // uncompressed point
	x.FillBytes(buf[1 : 1+byteLen])
	y.FillBytes(buf[1+byteLen : 1+2*byteLen])
	return curve.newPoint().SetBytes(buf)
}

func (curve *nistCurve[Point]) pointToAffine(p Point) (x, y *big.Int) {
	out := p.Bytes()
	if len(out) == 1 && out[0] == 0 {
		// This is the encoding of the point at infinity, which the affine
		// coordinates API represents as (0, 0) by convention.
		return new(big.Int), new(big.Int)
	}
	byteLen := (curve.params.BitSize + 7) / 8
	x = new(big.Int).SetBytes(out[1 : 1+byteLen])
	y = new(big.Int).SetBytes(out[1+byteLen:])
	return x, y
}

func (curve *nistCurve[Point]) Add(x1, y1, x2, y2 *big.Int) (*big.Int, *big.Int) {
	p1, err := curve.pointFromAffine(x1, y1)
	if err != nil {
		panic("crypto/elliptic: Add was called on an invalid point")
	}
	p2, err := curve.pointFromAffine(x2, y2)
	if err != nil {
		panic("crypto/elliptic: Add was called on an invalid point")
	}
	return curve.pointToAffine(p1.Add(p1, p2))
}

func (curve *nistCurve[Point]) Double(x1, y1 *big.Int) (*big.Int, *big.Int) {
	p, err := curve.pointFromAffine(x1, y1)
	if err != nil {
		panic("crypto/elliptic: Double was called on an invalid point")
	}
	return curve.pointToAffine(p.Double(p))
}

// normalizeScalar brings the scalar within the byte size of the order of the
// curve, as expected by the nistec scalar multiplication functions.
func (curve *nistCurve[Point]) normalizeScalar(scalar []byte) []byte {
	byteSize := (curve.params.N.BitLen() + 7) / 8
	if len(scalar) == byteSize {
		return scalar
	}
	s := new(big.Int).SetBytes(scalar)
	if len(scalar) > byteSize {
		s.Mod(s, curve.params.N)
	}
	out := make([]byte, byteSize)
	return s.FillBytes(out)
}

func (curve *nistCurve[Point]) ScalarMult(Bx, By *big.Int, scalar []byte) (*big.Int, *big.Int) {
	p, err := curve.pointFromAffine(Bx, By)
	if err != nil {
		panic("crypto/elliptic: ScalarMult was called on an invalid point")
	}
	scalar = curve.normalizeScalar(scalar)
	p, err = p.ScalarMult(p, scalar)
	if err != nil {
		panic("crypto/elliptic: nistec rejected normalized scalar")
	}
	return curve.pointToAffine(p)
}

func (curve *nistCurve[Point]) ScalarBaseMult(scalar []byte) (*big.Int, *big.Int) {
	scalar = curve.normalizeScalar(scalar)
	p, err := curve.newPoint().ScalarBaseMult(scalar)
	if err != nil {
		panic("crypto/elliptic: nistec rejected normalized scalar")
	}
	return curve.pointToAffine(p)
}

// CombinedMult returns [s1]G + [s2]P where G is the generator. It's used
// through an interface upgrade in crypto/ecdsa.
func (curve *nistCurve[Point]) CombinedMult(Px, Py *big.Int, s1, s2 []byte) (x, y *big.Int) {
	s1 = curve.normalizeScalar(s1)
	q, err := curve.newPoint().ScalarBaseMult(s1)
	if err != nil {
		panic("crypto/elliptic: nistec rejected normalized scalar")
	}
	p, err := curve.pointFromAffine(Px, Py)
	if err != nil {
		panic("crypto/elliptic: CombinedMult was called on an invalid point")
	}
	s2 = curve.normalizeScalar(s2)
	p, err = p.ScalarMult(p, s2)
	if err != nil {
		panic("crypto/elliptic: nistec rejected normalized scalar")
	}
	return curve.pointToAffine(p.Add(p, q))
}

func (curve *nistCurve[Point]) Unmarshal(data []byte) (x, y *big.Int) {
	if len(data) == 0 || data[0] != 4 {
		return nil, nil
	}
	// Use SetBytes to check that data encodes a valid point.
	_, err := curve.newPoint().SetBytes(data)
	if err != nil {
		return nil, nil
	}
	// We don't use pointToAffine because it involves an expensive field
	// inversion to convert from Jacobian to affine coordinates, which we
	// already have.
	byteLen := (curve.params.BitSize + 7) / 8
	x = new(big.Int).SetBytes(data[1 : 1+byteLen])
	y = new(big.Int).SetBytes(data[1+byteLen:])
	return x, y
}

func (curve *nistCurve[Point]) UnmarshalCompressed(data []byte) (x, y *big.Int) {
	if len(data) == 0 || (data[0] != 2 && data[0] != 3) {
		return nil, nil
	}
	p, err := curve.newPoint().SetBytes(data)
	if err != nil {
		return nil, nil
	}
	return curve.pointToAffine(p)
}

func bigFromDecimal(s string) *big.Int {
	b, ok := new(big.Int).SetString(s, 10)
	if !ok {
		panic("crypto/elliptic: internal error: invalid encoding")
	}
	return b
}

func bigFromHex(s string) *big.Int {
	b, ok := new(big.Int).SetString(s, 16)
	if !ok {
		panic("crypto/elliptic: internal error: invalid encoding")
	}
	return b
}

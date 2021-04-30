// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"crypto/elliptic/internal/fiat"
	"math/big"
)

type p521Curve struct {
	*CurveParams
}

var p521 p521Curve
var p521Params *CurveParams

func initP521() {
	// See FIPS 186-3, section D.2.5
	p521.CurveParams = &CurveParams{Name: "P-521"}
	p521.P, _ = new(big.Int).SetString("6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057151", 10)
	p521.N, _ = new(big.Int).SetString("6864797660130609714981900799081393217269435300143305409394463459185543183397655394245057746333217197532963996371363321113864768612440380340372808892707005449", 10)
	p521.B, _ = new(big.Int).SetString("051953eb9618e1c9a1f929a21a0b68540eea2da725b99b315f3b8b489918ef109e156193951ec7e937b1652c0bd3bb1bf073573df883d2c34f1ef451fd46b503f00", 16)
	p521.Gx, _ = new(big.Int).SetString("c6858e06b70404e9cd9e3ecb662395b4429c648139053fb521f828af606b4d3dbaa14b5e77efe75928fe1dc127a2ffa8de3348b3c1856a429bf97e7e31c2e5bd66", 16)
	p521.Gy, _ = new(big.Int).SetString("11839296a789a3bc0045c8a5fb42c7d1bd998f54449579b446817afbd17273e662c97ee72995ef42640c550b9013fad0761353c7086a272c24088be94769fd16650", 16)
	p521.BitSize = 521
}

func (curve p521Curve) Params() *CurveParams {
	return curve.CurveParams
}

func (curve p521Curve) IsOnCurve(x, y *big.Int) bool {
	x1 := bigIntToFiatP521(x)
	y1 := bigIntToFiatP521(y)
	b := bigIntToFiatP521(curve.B) // TODO: precompute this value.

	// x³ - 3x + b.
	x3 := new(fiat.P521Element).Square(x1)
	x3.Mul(x3, x1)

	threeX := new(fiat.P521Element).Add(x1, x1)
	threeX.Add(threeX, x1)

	x3.Sub(x3, threeX)
	x3.Add(x3, b)

	// y² = x³ - 3x + b
	y2 := new(fiat.P521Element).Square(y1)

	return x3.Equal(y2) == 1
}

type p512Point struct {
	x, y, z *fiat.P521Element
}

func fiatP521ToBigInt(x *fiat.P521Element) *big.Int {
	xBytes := x.Bytes()
	for i := range xBytes[:len(xBytes)/2] {
		xBytes[i], xBytes[len(xBytes)-i-1] = xBytes[len(xBytes)-i-1], xBytes[i]
	}
	return new(big.Int).SetBytes(xBytes)
}

// affineFromJacobian brings a point in Jacobian coordinates back to affine
// coordinates, with (0, 0) representing infinity by convention. It also goes
// back to big.Int values to match the exposed API.
func (curve p521Curve) affineFromJacobian(p *p512Point) (x, y *big.Int) {
	if p.z.IsZero() == 1 {
		return new(big.Int), new(big.Int)
	}

	zinv := new(fiat.P521Element).Invert(p.z)
	zinvsq := new(fiat.P521Element).Mul(zinv, zinv)

	xx := new(fiat.P521Element).Mul(p.x, zinvsq)
	zinvsq.Mul(zinvsq, zinv)
	yy := new(fiat.P521Element).Mul(p.y, zinvsq)

	return fiatP521ToBigInt(xx), fiatP521ToBigInt(yy)
}

func bigIntToFiatP521(x *big.Int) *fiat.P521Element {
	xBytes := new(big.Int).Mod(x, p521.P).FillBytes(make([]byte, 66))
	for i := range xBytes[:len(xBytes)/2] {
		xBytes[i], xBytes[len(xBytes)-i-1] = xBytes[len(xBytes)-i-1], xBytes[i]
	}
	x1, err := new(fiat.P521Element).SetBytes(xBytes)
	if err != nil {
		// The input is reduced modulo P and encoded in a fixed size bytes
		// slice, this should be impossible.
		panic("internal error: bigIntToFiatP521")
	}
	return x1
}

// jacobianFromAffine converts (x, y) affine coordinates into (x, y, z) Jacobian
// coordinates. It also converts from big.Int to fiat, which is necessarily a
// messy and variable-time operation, which we can't avoid due to the exposed API.
func (curve p521Curve) jacobianFromAffine(x, y *big.Int) *p512Point {
	// (0, 0) is by convention the point at infinity, which can't be represented
	// in affine coordinates, but is (0, 0, 0) in Jacobian.
	if x.Sign() == 0 && y.Sign() == 0 {
		return &p512Point{
			x: new(fiat.P521Element),
			y: new(fiat.P521Element),
			z: new(fiat.P521Element),
		}
	}
	return &p512Point{
		x: bigIntToFiatP521(x),
		y: bigIntToFiatP521(y),
		z: new(fiat.P521Element).One(),
	}
}

func (curve p521Curve) Add(x1, y1, x2, y2 *big.Int) (*big.Int, *big.Int) {
	p1 := curve.jacobianFromAffine(x1, y1)
	p2 := curve.jacobianFromAffine(x2, y2)
	return curve.affineFromJacobian(p1.addJacobian(p1, p2))
}

// addJacobian sets q = p1 + p2, and returns q. The points may overlap.
func (q *p512Point) addJacobian(p1, p2 *p512Point) *p512Point {
	// https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#addition-add-2007-bl
	if p1.z.IsZero() == 1 {
		q.x.Set(p2.x)
		q.y.Set(p2.y)
		q.z.Set(p2.z)
		return q
	}
	if p2.z.IsZero() == 1 {
		q.x.Set(p1.x)
		q.y.Set(p1.y)
		q.z.Set(p1.z)
		return q
	}

	z1z1 := new(fiat.P521Element).Square(p1.z)
	z2z2 := new(fiat.P521Element).Square(p2.z)

	u1 := new(fiat.P521Element).Mul(p1.x, z2z2)
	u2 := new(fiat.P521Element).Mul(p2.x, z1z1)
	h := new(fiat.P521Element).Sub(u2, u1)
	xEqual := h.IsZero() == 1
	i := new(fiat.P521Element).Add(h, h)
	i.Square(i)
	j := new(fiat.P521Element).Mul(h, i)

	s1 := new(fiat.P521Element).Mul(p1.y, p2.z)
	s1.Mul(s1, z2z2)
	s2 := new(fiat.P521Element).Mul(p2.y, p1.z)
	s2.Mul(s2, z1z1)
	r := new(fiat.P521Element).Sub(s2, s1)
	yEqual := r.IsZero() == 1
	if xEqual && yEqual {
		return q.doubleJacobian(p1)
	}
	r.Add(r, r)
	v := new(fiat.P521Element).Mul(u1, i)

	q.x.Set(r)
	q.x.Square(q.x)
	q.x.Sub(q.x, j)
	q.x.Sub(q.x, v)
	q.x.Sub(q.x, v)

	q.y.Set(r)
	v.Sub(v, q.x)
	q.y.Mul(q.y, v)
	s1.Mul(s1, j)
	s1.Add(s1, s1)
	q.y.Sub(q.y, s1)

	q.z.Add(p1.z, p2.z)
	q.z.Square(q.z)
	q.z.Sub(q.z, z1z1)
	q.z.Sub(q.z, z2z2)
	q.z.Mul(q.z, h)

	return q
}

func (curve p521Curve) Double(x1, y1 *big.Int) (*big.Int, *big.Int) {
	p := curve.jacobianFromAffine(x1, y1)
	return curve.affineFromJacobian(p.doubleJacobian(p))
}

// doubleJacobian sets q = p + p, and returns q. The points may overlap.
func (q *p512Point) doubleJacobian(p *p512Point) *p512Point {
	// https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#doubling-dbl-2001-b
	delta := new(fiat.P521Element).Square(p.z)
	gamma := new(fiat.P521Element).Square(p.y)
	alpha := new(fiat.P521Element).Sub(p.x, delta)
	alpha2 := new(fiat.P521Element).Add(p.x, delta)
	alpha.Mul(alpha, alpha2)
	alpha2.Set(alpha)
	alpha.Add(alpha, alpha)
	alpha.Add(alpha, alpha2)

	beta := alpha2.Mul(p.x, gamma)

	q.x.Square(alpha)
	beta8 := new(fiat.P521Element).Add(beta, beta)
	beta8.Add(beta8, beta8)
	beta8.Add(beta8, beta8)
	q.x.Sub(q.x, beta8)

	q.z.Add(p.y, p.z)
	q.z.Square(q.z)
	q.z.Sub(q.z, gamma)
	q.z.Sub(q.z, delta)

	beta.Add(beta, beta)
	beta.Add(beta, beta)
	beta.Sub(beta, q.x)
	q.y.Mul(alpha, beta)

	gamma.Square(gamma)
	gamma.Add(gamma, gamma)
	gamma.Add(gamma, gamma)
	gamma.Add(gamma, gamma)

	q.y.Sub(q.y, gamma)

	return q
}

func (curve p521Curve) ScalarMult(Bx, By *big.Int, k []byte) (*big.Int, *big.Int) {
	B := curve.jacobianFromAffine(Bx, By)
	p := &p512Point{
		x: new(fiat.P521Element),
		y: new(fiat.P521Element),
		z: new(fiat.P521Element),
	}

	for _, byte := range k {
		for bitNum := 0; bitNum < 8; bitNum++ {
			p.doubleJacobian(p)
			if byte&0x80 == 0x80 {
				p.addJacobian(B, p)
			}
			byte <<= 1
		}
	}

	return curve.affineFromJacobian(p)
}

func (curve p521Curve) ScalarBaseMult(k []byte) (*big.Int, *big.Int) {
	return curve.ScalarMult(curve.Gx, curve.Gy, k)
}

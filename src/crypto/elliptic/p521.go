// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"crypto/elliptic/internal/fiat"
	"crypto/subtle"
	"math/big"
)

type p521Curve struct {
	*CurveParams
	b *fiat.P521Element
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
	p521.b = bigIntToFiatP521(p521.B)
}

func (curve p521Curve) Params() *CurveParams {
	return curve.CurveParams
}

func (curve p521Curve) IsOnCurve(x, y *big.Int) bool {
	x1 := bigIntToFiatP521(x)
	y1 := bigIntToFiatP521(y)

	// x³ - 3x + b.
	x3 := new(fiat.P521Element).Square(x1)
	x3.Mul(x3, x1)

	threeX := new(fiat.P521Element).Add(x1, x1)
	threeX.Add(threeX, x1)

	x3.Sub(x3, threeX)
	x3.Add(x3, curve.b)

	// y² = x³ - 3x + b
	y2 := new(fiat.P521Element).Square(y1)

	return x3.Equal(y2) == 1
}

// p521Point is a P-521 point in projective coordinates, where x = X/Z, y = Y/Z.
type p521Point struct {
	x, y, z *fiat.P521Element
}

// newP521Point returns a new p521Point representing the identity point.
func newP521Point() *p521Point {
	return &p521Point{
		x: new(fiat.P521Element),
		y: new(fiat.P521Element).One(),
		z: new(fiat.P521Element),
	}
}

func fiatP521ToBigInt(x *fiat.P521Element) *big.Int {
	xBytes := x.Bytes()
	for i := range xBytes[:len(xBytes)/2] {
		xBytes[i], xBytes[len(xBytes)-i-1] = xBytes[len(xBytes)-i-1], xBytes[i]
	}
	return new(big.Int).SetBytes(xBytes)
}

// Affine returns p in affine coordinates, with (0, 0) representing infinity by
// convention. It also goes back to big.Int values to match the exposed API.
func (p *p521Point) Affine() (x, y *big.Int) {
	if p.z.IsZero() == 1 {
		return new(big.Int), new(big.Int)
	}

	zinv := new(fiat.P521Element).Invert(p.z)
	xx := new(fiat.P521Element).Mul(p.x, zinv)
	yy := new(fiat.P521Element).Mul(p.y, zinv)

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

// newP521PointFromAffine converts (x, y) affine coordinates into (X, Y, Z) projective
// coordinates. It also converts from big.Int to fiat, which is necessarily a
// messy and variable-time operation, which we can't avoid due to the exposed API.
func newP521PointFromAffine(x, y *big.Int) *p521Point {
	// (0, 0) is by convention the point at infinity, which can't be represented
	// in affine coordinates, but is (0, 0, 0) in projective coordinates.
	if x.Sign() == 0 && y.Sign() == 0 {
		return newP521Point()
	}
	return &p521Point{
		x: bigIntToFiatP521(x),
		y: bigIntToFiatP521(y),
		z: new(fiat.P521Element).One(),
	}
}

func (curve p521Curve) Add(x1, y1, x2, y2 *big.Int) (*big.Int, *big.Int) {
	p1 := newP521PointFromAffine(x1, y1)
	p2 := newP521PointFromAffine(x2, y2)
	return p1.Add(p1, p2).Affine()
}

// Add sets q = p1 + p2, and returns q. The points may overlap.
func (q *p521Point) Add(p1, p2 *p521Point) *p521Point {
	// Complete addition formula for a = -3 from "Complete addition formulas for
	// prime order elliptic curves" (https://eprint.iacr.org/2015/1060), §A.2.

	t0 := new(fiat.P521Element).Mul(p1.x, p2.x) // t0 := X1 * X2
	t1 := new(fiat.P521Element).Mul(p1.y, p2.y) // t1 := Y1 * Y2
	t2 := new(fiat.P521Element).Mul(p1.z, p2.z) // t2 := Z1 * Z2
	t3 := new(fiat.P521Element).Add(p1.x, p1.y) // t3 := X1 + Y1
	t4 := new(fiat.P521Element).Add(p2.x, p2.y) // t4 := X2 + Y2
	t3.Mul(t3, t4)                              // t3 := t3 * t4
	t4.Add(t0, t1)                              // t4 := t0 + t1
	t3.Sub(t3, t4)                              // t3 := t3 - t4
	t4.Add(p1.y, p1.z)                          // t4 := Y1 + Z1
	x := new(fiat.P521Element).Add(p2.y, p2.z)  // X3 := Y2 + Z2
	t4.Mul(t4, x)                               // t4 := t4 * X3
	x.Add(t1, t2)                               // X3 := t1 + t2
	t4.Sub(t4, x)                               // t4 := t4 - X3
	x.Add(p1.x, p1.z)                           // X3 := X1 + Z1
	y := new(fiat.P521Element).Add(p2.x, p2.z)  // Y3 := X2 + Z2
	x.Mul(x, y)                                 // X3 := X3 * Y3
	y.Add(t0, t2)                               // Y3 := t0 + t2
	y.Sub(x, y)                                 // Y3 := X3 - Y3
	z := new(fiat.P521Element).Mul(p521.b, t2)  // Z3 := b * t2
	x.Sub(y, z)                                 // X3 := Y3 - Z3
	z.Add(x, x)                                 // Z3 := X3 + X3
	x.Add(x, z)                                 // X3 := X3 + Z3
	z.Sub(t1, x)                                // Z3 := t1 - X3
	x.Add(t1, x)                                // X3 := t1 + X3
	y.Mul(p521.b, y)                            // Y3 := b * Y3
	t1.Add(t2, t2)                              // t1 := t2 + t2
	t2.Add(t1, t2)                              // t2 := t1 + t2
	y.Sub(y, t2)                                // Y3 := Y3 - t2
	y.Sub(y, t0)                                // Y3 := Y3 - t0
	t1.Add(y, y)                                // t1 := Y3 + Y3
	y.Add(t1, y)                                // Y3 := t1 + Y3
	t1.Add(t0, t0)                              // t1 := t0 + t0
	t0.Add(t1, t0)                              // t0 := t1 + t0
	t0.Sub(t0, t2)                              // t0 := t0 - t2
	t1.Mul(t4, y)                               // t1 := t4 * Y3
	t2.Mul(t0, y)                               // t2 := t0 * Y3
	y.Mul(x, z)                                 // Y3 := X3 * Z3
	y.Add(y, t2)                                // Y3 := Y3 + t2
	x.Mul(t3, x)                                // X3 := t3 * X3
	x.Sub(x, t1)                                // X3 := X3 - t1
	z.Mul(t4, z)                                // Z3 := t4 * Z3
	t1.Mul(t3, t0)                              // t1 := t3 * t0
	z.Add(z, t1)                                // Z3 := Z3 + t1

	q.x.Set(x)
	q.y.Set(y)
	q.z.Set(z)
	return q
}

func (curve p521Curve) Double(x1, y1 *big.Int) (*big.Int, *big.Int) {
	p := newP521PointFromAffine(x1, y1)
	return p.Double(p).Affine()
}

// Double sets q = p + p, and returns q. The points may overlap.
func (q *p521Point) Double(p *p521Point) *p521Point {
	// Complete addition formula for a = -3 from "Complete addition formulas for
	// prime order elliptic curves" (https://eprint.iacr.org/2015/1060), §A.2.

	t0 := new(fiat.P521Element).Square(p.x)    // t0 := X ^ 2
	t1 := new(fiat.P521Element).Square(p.y)    // t1 := Y ^ 2
	t2 := new(fiat.P521Element).Square(p.z)    // t2 := Z ^ 2
	t3 := new(fiat.P521Element).Mul(p.x, p.y)  // t3 := X * Y
	t3.Add(t3, t3)                             // t3 := t3 + t3
	z := new(fiat.P521Element).Mul(p.x, p.z)   // Z3 := X * Z
	z.Add(z, z)                                // Z3 := Z3 + Z3
	y := new(fiat.P521Element).Mul(p521.b, t2) // Y3 := b * t2
	y.Sub(y, z)                                // Y3 := Y3 - Z3
	x := new(fiat.P521Element).Add(y, y)       // X3 := Y3 + Y3
	y.Add(x, y)                                // Y3 := X3 + Y3
	x.Sub(t1, y)                               // X3 := t1 - Y3
	y.Add(t1, y)                               // Y3 := t1 + Y3
	y.Mul(x, y)                                // Y3 := X3 * Y3
	x.Mul(x, t3)                               // X3 := X3 * t3
	t3.Add(t2, t2)                             // t3 := t2 + t2
	t2.Add(t2, t3)                             // t2 := t2 + t3
	z.Mul(p521.b, z)                           // Z3 := b * Z3
	z.Sub(z, t2)                               // Z3 := Z3 - t2
	z.Sub(z, t0)                               // Z3 := Z3 - t0
	t3.Add(z, z)                               // t3 := Z3 + Z3
	z.Add(z, t3)                               // Z3 := Z3 + t3
	t3.Add(t0, t0)                             // t3 := t0 + t0
	t0.Add(t3, t0)                             // t0 := t3 + t0
	t0.Sub(t0, t2)                             // t0 := t0 - t2
	t0.Mul(t0, z)                              // t0 := t0 * Z3
	y.Add(y, t0)                               // Y3 := Y3 + t0
	t0.Mul(p.y, p.z)                           // t0 := Y * Z
	t0.Add(t0, t0)                             // t0 := t0 + t0
	z.Mul(t0, z)                               // Z3 := t0 * Z3
	x.Sub(x, z)                                // X3 := X3 - Z3
	z.Mul(t0, t1)                              // Z3 := t0 * t1
	z.Add(z, z)                                // Z3 := Z3 + Z3
	z.Add(z, z)                                // Z3 := Z3 + Z3

	q.x.Set(x)
	q.y.Set(y)
	q.z.Set(z)
	return q
}

// Select sets q to p1 if cond == 1, and to p2 if cond == 0.
func (q *p521Point) Select(p1, p2 *p521Point, cond int) *p521Point {
	q.x.Select(p1.x, p2.x, cond)
	q.y.Select(p1.y, p2.y, cond)
	q.z.Select(p1.z, p2.z, cond)
	return q
}

func (curve p521Curve) ScalarMult(Bx, By *big.Int, scalar []byte) (*big.Int, *big.Int) {
	B := newP521PointFromAffine(Bx, By)
	p, t := newP521Point(), newP521Point()

	// table holds the first 16 multiples of q. The explicit newP521Point calls
	// get inlined, letting the allocations live on the stack.
	var table = [16]*p521Point{
		newP521Point(), newP521Point(), newP521Point(), newP521Point(),
		newP521Point(), newP521Point(), newP521Point(), newP521Point(),
		newP521Point(), newP521Point(), newP521Point(), newP521Point(),
		newP521Point(), newP521Point(), newP521Point(), newP521Point(),
	}
	for i := 1; i < 16; i++ {
		table[i].Add(table[i-1], B)
	}

	// Instead of doing the classic double-and-add chain, we do it with a
	// four-bit window: we double four times, and then add [0-15]P.
	for _, byte := range scalar {
		p.Double(p)
		p.Double(p)
		p.Double(p)
		p.Double(p)

		for i := uint8(0); i < 16; i++ {
			cond := subtle.ConstantTimeByteEq(byte>>4, i)
			t.Select(table[i], t, cond)
		}
		p.Add(p, t)

		p.Double(p)
		p.Double(p)
		p.Double(p)
		p.Double(p)

		for i := uint8(0); i < 16; i++ {
			cond := subtle.ConstantTimeByteEq(byte&0b1111, i)
			t.Select(table[i], t, cond)
		}
		p.Add(p, t)
	}

	return p.Affine()
}

func (curve p521Curve) ScalarBaseMult(k []byte) (*big.Int, *big.Int) {
	return curve.ScalarMult(curve.Gx, curve.Gy, k)
}

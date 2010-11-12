// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The elliptic package implements several standard elliptic curves over prime
// fields
package elliptic

// WARNING: this implementation is simple but slow and not constant time.
// A significant speedup could be obtained by using either a projective or
// Jacobian transform.

import (
	"big"
	"sync"
)

// A Curve represents a short-form Weierstrass curve with a=-3.
// See http://www.hyperelliptic.org/EFD/g1p/auto-shortw.html
type Curve struct {
	P      *big.Int // the order of the underlying field
	B      *big.Int // the constant of the curve equation
	Gx, Gy *big.Int // (x,y) of the base point
}

// IsOnCurve returns true if the given (x,y) lies on the curve.
func (curve *Curve) IsOnCurve(x, y *big.Int) bool {
	// y² = x³ - 3x + b
	y2 := new(big.Int).Mul(y, y)
	y2.Mod(y2, curve.P)

	x3 := new(big.Int).Mul(x, x)
	x3.Mul(x3, x)

	threeX := new(big.Int).Lsh(x, 1)
	threeX.Add(threeX, x)

	x3.Sub(x3, threeX)
	x3.Add(x3, curve.B)
	x3.Mod(x3, curve.P)

	return x3.Cmp(y2) == 0
}

// Add returns the sum of (x1,y1) and (x2,y2)
func (curve *Curve) Add(x1, y1, x2, y2 *big.Int) (*big.Int, *big.Int) {
	// x = (y2-y1)²/(x2-x1)²-x1-x2
	y2my1 := new(big.Int).Sub(y2, y1)
	if y2my1.Sign() < 0 {
		y2my1.Add(y2my1, curve.P)
	}
	y2my1sq := new(big.Int).Mul(y2my1, y2my1)
	x2mx1 := new(big.Int).Sub(x2, x1)
	if x2mx1.Sign() < 0 {
		x2mx1.Add(x2mx1, curve.P)
	}
	x2mx1sq := new(big.Int).Mul(x2mx1, x2mx1)
	x2mx1sqinv := new(big.Int).ModInverse(x2mx1sq, curve.P)

	x := new(big.Int).Mul(y2my1sq, x2mx1sqinv)
	x.Sub(x, x1)
	x.Sub(x, x2)
	x.Mod(x, curve.P)

	// y = (2x1+x2)*(y2-y1)/(x2-x1)-(y2-y1)³/(x2-x1)³-y1
	y := new(big.Int).Lsh(x1, 1)
	y.Add(y, x2)
	x2mx1inv := new(big.Int).ModInverse(x2mx1, curve.P)
	x2mx1inv.Mul(y2my1, x2mx1inv)
	y.Mul(y, x2mx1inv)

	y2my1sq.Mul(y2my1sq, y2my1)
	x2mx1sq.Mul(x2mx1sq, x2mx1)
	x2mx1sqinv.ModInverse(x2mx1sq, curve.P)
	y2my1sq.Mul(y2my1sq, x2mx1sqinv)
	y.Sub(y, y2my1sq)
	y.Sub(y, y1)
	y.Mod(y, curve.P)

	return x, y
}

// Double returns 2*(x,y)
func (curve *Curve) Double(x, y *big.Int) (*big.Int, *big.Int) {
	// x = (3x²-3)²/(2y)²-x-x
	threexsqm3 := new(big.Int).Mul(x, x)
	three := new(big.Int).SetInt64(3)
	threexsqm3.Mul(threexsqm3, three)
	threexsqm3.Sub(threexsqm3, three)
	threexsqm3sq := new(big.Int).Mul(threexsqm3, threexsqm3)

	twoy := new(big.Int).Lsh(y, 1)
	twoysq := new(big.Int).Mul(twoy, twoy)
	twoysqinv := new(big.Int).ModInverse(twoysq, curve.P)

	outx := new(big.Int).Mul(threexsqm3sq, twoysqinv)
	outx.Sub(outx, x)
	outx.Sub(outx, x)
	outx.Mod(outx, curve.P)

	// y = 3x*(3x²-3)/(2y)-(3x²-3)³/(2y)³-y
	outy := new(big.Int).Mul(x, three)
	outy.Mul(outy, threexsqm3)
	twoyinv := new(big.Int).ModInverse(twoy, curve.P)
	outy.Mul(outy, twoyinv)

	threexsqm3sq.Mul(threexsqm3sq, threexsqm3)
	twoysq.Mul(twoysq, twoy)
	twoysqinv.ModInverse(twoysq, curve.P)
	threexsqm3sq.Mul(threexsqm3sq, twoysqinv)
	outy.Sub(outy, threexsqm3sq)
	outy.Sub(outy, y)
	outy.Mod(outy, curve.P)

	return outx, outy
}

// ScalarMult returns k*(Bx,By) where k is a number in big-endian form.
func (curve *Curve) ScalarMult(Bx, By *big.Int, k []byte) (*big.Int, *big.Int) {
	// We have a slight problem in that the identity of the group (the
	// point at infinity) cannot be represented in (x, y) form on a finite
	// machine. Thus the standard add/double algorithm has to be tweaked
	// slightly: our initial state is not the identity, but x, and we
	// ignore the first true bit in |k|.  If we don't find any true bits in
	// |k|, then we return nil, nil, because we cannot return the identity
	// element.

	x := Bx
	y := By

	seenFirstTrue := false
	for _, byte := range k {
		for bitNum := 0; bitNum < 8; bitNum++ {
			if seenFirstTrue {
				x, y = curve.Double(x, y)
			}
			if byte&0x80 == 0x80 {
				if !seenFirstTrue {
					seenFirstTrue = true
				} else {
					x, y = curve.Add(Bx, By, x, y)
				}
			}
			byte <<= 1
		}
	}

	if !seenFirstTrue {
		return nil, nil
	}

	return x, y
}

// ScalarBaseMult returns k*G, where G is the base point of the group and k is
// an integer in big-endian form.
func (curve *Curve) ScalarBaseMult(k []byte) (*big.Int, *big.Int) {
	return curve.ScalarMult(curve.Gx, curve.Gy, k)
}

var initonce sync.Once
var p224 *Curve
var p256 *Curve
var p384 *Curve
var p521 *Curve

func initAll() {
	initP224()
	initP256()
	initP384()
	initP521()
}

func initP224() {
	// See FIPS 186-3, section D.2.2
	p224 = new(Curve)
	p224.P, _ = new(big.Int).SetString("26959946667150639794667015087019630673557916260026308143510066298881", 10)
	p224.B, _ = new(big.Int).SetString("b4050a850c04b3abf54132565044b0b7d7bfd8ba270b39432355ffb4", 16)
	p224.Gx, _ = new(big.Int).SetString("b70e0cbd6bb4bf7f321390b94a03c1d356c21122343280d6115c1d21", 16)
	p224.Gy, _ = new(big.Int).SetString("bd376388b5f723fb4c22dfe6cd4375a05a07476444d5819985007e34", 16)
}

func initP256() {
	// See FIPS 186-3, section D.2.3
	p256 = new(Curve)
	p256.P, _ = new(big.Int).SetString("115792089210356248762697446949407573530086143415290314195533631308867097853951", 10)
	p256.B, _ = new(big.Int).SetString("5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b", 16)
	p256.Gx, _ = new(big.Int).SetString("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296", 16)
	p256.Gy, _ = new(big.Int).SetString("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5", 16)
}

func initP384() {
	// See FIPS 186-3, section D.2.4
	p384 = new(Curve)
	p384.P, _ = new(big.Int).SetString("39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319", 10)
	p384.B, _ = new(big.Int).SetString("b3312fa7e23ee7e4988e056be3f82d19181d9c6efe8141120314088f5013875ac656398d8a2ed19d2a85c8edd3ec2aef", 16)
	p384.Gx, _ = new(big.Int).SetString("aa87ca22be8b05378eb1c71ef320ad746e1d3b628ba79b9859f741e082542a385502f25dbf55296c3a545e3872760ab7", 16)
	p384.Gy, _ = new(big.Int).SetString("3617de4a96262c6f5d9e98bf9292dc29f8f41dbd289a147ce9da3113b5f0b8c00a60b1ce1d7e819d7a431d7c90ea0e5f", 16)
}

func initP521() {
	// See FIPS 186-3, section D.2.5
	p521 = new(Curve)
	p521.P, _ = new(big.Int).SetString("6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057151", 10)
	p521.B, _ = new(big.Int).SetString("051953eb9618e1c9a1f929a21a0b68540eea2da725b99b315f3b8b489918ef109e156193951ec7e937b1652c0bd3bb1bf073573df883d2c34f1ef451fd46b503f00", 16)
	p521.Gx, _ = new(big.Int).SetString("c6858e06b70404e9cd9e3ecb662395b4429c648139053fb521f828af606b4d3dbaa14b5e77efe75928fe1dc127a2ffa8de3348b3c1856a429bf97e7e31c2e5bd66", 16)
	p521.Gy, _ = new(big.Int).SetString("11839296a789a3bc0045c8a5fb42c7d1bd998f54449579b446817afbd17273e662c97ee72995ef42640c550b9013fad0761353c7086a272c24088be94769fd16650", 16)
}

// P224 returns a Curve which implements P-224 (see FIPS 186-3, section D.2.2)
func P224() *Curve {
	initonce.Do(initAll)
	return p224
}

// P256 returns a Curve which implements P-256 (see FIPS 186-3, section D.2.3)
func P256() *Curve {
	initonce.Do(initAll)
	return p256
}

// P384 returns a Curve which implements P-384 (see FIPS 186-3, section D.2.4)
func P384() *Curve {
	initonce.Do(initAll)
	return p384
}

// P256 returns a Curve which implements P-521 (see FIPS 186-3, section D.2.5)
func P521() *Curve {
	initonce.Do(initAll)
	return p521
}

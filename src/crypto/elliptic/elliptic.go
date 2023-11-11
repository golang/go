// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package elliptic implements the standard NIST P-224, P-256, P-384, and P-521
// elliptic curves over prime fields.
//
// Direct use of this package is deprecated, beyond the [P224], [P256], [P384],
// and [P521] values necessary to use [crypto/ecdsa]. Most other uses
// should migrate to the more efficient and safer [crypto/ecdh], or to
// third-party modules for lower-level functionality.
package elliptic

import (
	"io"
	"math/big"
	"sync"
)

// A Curve represents a short-form Weierstrass curve with a=-3.
//
// The behavior of Add, Double, and ScalarMult when the input is not a point on
// the curve is undefined.
//
// Note that the conventional point at infinity (0, 0) is not considered on the
// curve, although it can be returned by Add, Double, ScalarMult, or
// ScalarBaseMult (but not the [Unmarshal] or [UnmarshalCompressed] functions).
//
// Using Curve implementations besides those returned by [P224], [P256], [P384],
// and [P521] is deprecated.
type Curve interface {
	// Params returns the parameters for the curve.
	Params() *CurveParams

	// IsOnCurve reports whether the given (x,y) lies on the curve.
	//
	// Deprecated: this is a low-level unsafe API. For ECDH, use the crypto/ecdh
	// package. The NewPublicKey methods of NIST curves in crypto/ecdh accept
	// the same encoding as the Unmarshal function, and perform on-curve checks.
	IsOnCurve(x, y *big.Int) bool

	// Add returns the sum of (x1,y1) and (x2,y2).
	//
	// Deprecated: this is a low-level unsafe API.
	Add(x1, y1, x2, y2 *big.Int) (x, y *big.Int)

	// Double returns 2*(x,y).
	//
	// Deprecated: this is a low-level unsafe API.
	Double(x1, y1 *big.Int) (x, y *big.Int)

	// ScalarMult returns k*(x,y) where k is an integer in big-endian form.
	//
	// Deprecated: this is a low-level unsafe API. For ECDH, use the crypto/ecdh
	// package. Most uses of ScalarMult can be replaced by a call to the ECDH
	// methods of NIST curves in crypto/ecdh.
	ScalarMult(x1, y1 *big.Int, k []byte) (x, y *big.Int)

	// ScalarBaseMult returns k*G, where G is the base point of the group
	// and k is an integer in big-endian form.
	//
	// Deprecated: this is a low-level unsafe API. For ECDH, use the crypto/ecdh
	// package. Most uses of ScalarBaseMult can be replaced by a call to the
	// PrivateKey.PublicKey method in crypto/ecdh.
	ScalarBaseMult(k []byte) (x, y *big.Int)
}

var mask = []byte{0xff, 0x1, 0x3, 0x7, 0xf, 0x1f, 0x3f, 0x7f}

// GenerateKey returns a public/private key pair. The private key is
// generated using the given reader, which must return random data.
//
// Deprecated: for ECDH, use the GenerateKey methods of the [crypto/ecdh] package;
// for ECDSA, use the GenerateKey function of the crypto/ecdsa package.
func GenerateKey(curve Curve, rand io.Reader) (priv []byte, x, y *big.Int, err error) {
	N := curve.Params().N
	bitSize := N.BitLen()
	byteLen := (bitSize + 7) / 8
	priv = make([]byte, byteLen)

	for x == nil {
		_, err = io.ReadFull(rand, priv)
		if err != nil {
			return
		}
		// We have to mask off any excess bits in the case that the size of the
		// underlying field is not a whole number of bytes.
		priv[0] &= mask[bitSize%8]
		// This is because, in tests, rand will return all zeros and we don't
		// want to get the point at infinity and loop forever.
		priv[1] ^= 0x42

		// If the scalar is out of range, sample another random number.
		if new(big.Int).SetBytes(priv).Cmp(N) >= 0 {
			continue
		}

		x, y = curve.ScalarBaseMult(priv)
	}
	return
}

// Marshal converts a point on the curve into the uncompressed form specified in
// SEC 1, Version 2.0, Section 2.3.3. If the point is not on the curve (or is
// the conventional point at infinity), the behavior is undefined.
//
// Deprecated: for ECDH, use the crypto/ecdh package. This function returns an
// encoding equivalent to that of PublicKey.Bytes in crypto/ecdh.
func Marshal(curve Curve, x, y *big.Int) []byte {
	panicIfNotOnCurve(curve, x, y)

	byteLen := (curve.Params().BitSize + 7) / 8

	ret := make([]byte, 1+2*byteLen)
	ret[0] = 4 // uncompressed point

	x.FillBytes(ret[1 : 1+byteLen])
	y.FillBytes(ret[1+byteLen : 1+2*byteLen])

	return ret
}

// MarshalCompressed converts a point on the curve into the compressed form
// specified in SEC 1, Version 2.0, Section 2.3.3. If the point is not on the
// curve (or is the conventional point at infinity), the behavior is undefined.
func MarshalCompressed(curve Curve, x, y *big.Int) []byte {
	panicIfNotOnCurve(curve, x, y)
	byteLen := (curve.Params().BitSize + 7) / 8
	compressed := make([]byte, 1+byteLen)
	compressed[0] = byte(y.Bit(0)) | 2
	x.FillBytes(compressed[1:])
	return compressed
}

// unmarshaler is implemented by curves with their own constant-time Unmarshal.
//
// There isn't an equivalent interface for Marshal/MarshalCompressed because
// that doesn't involve any mathematical operations, only FillBytes and Bit.
type unmarshaler interface {
	Unmarshal([]byte) (x, y *big.Int)
	UnmarshalCompressed([]byte) (x, y *big.Int)
}

// Assert that the known curves implement unmarshaler.
var _ = []unmarshaler{p224, p256, p384, p521}

// Unmarshal converts a point, serialized by [Marshal], into an x, y pair. It is
// an error if the point is not in uncompressed form, is not on the curve, or is
// the point at infinity. On error, x = nil.
//
// Deprecated: for ECDH, use the crypto/ecdh package. This function accepts an
// encoding equivalent to that of the NewPublicKey methods in crypto/ecdh.
func Unmarshal(curve Curve, data []byte) (x, y *big.Int) {
	if c, ok := curve.(unmarshaler); ok {
		return c.Unmarshal(data)
	}

	byteLen := (curve.Params().BitSize + 7) / 8
	if len(data) != 1+2*byteLen {
		return nil, nil
	}
	if data[0] != 4 { // uncompressed form
		return nil, nil
	}
	p := curve.Params().P
	x = new(big.Int).SetBytes(data[1 : 1+byteLen])
	y = new(big.Int).SetBytes(data[1+byteLen:])
	if x.Cmp(p) >= 0 || y.Cmp(p) >= 0 {
		return nil, nil
	}
	if !curve.IsOnCurve(x, y) {
		return nil, nil
	}
	return
}

// UnmarshalCompressed converts a point, serialized by [MarshalCompressed], into
// an x, y pair. It is an error if the point is not in compressed form, is not
// on the curve, or is the point at infinity. On error, x = nil.
func UnmarshalCompressed(curve Curve, data []byte) (x, y *big.Int) {
	if c, ok := curve.(unmarshaler); ok {
		return c.UnmarshalCompressed(data)
	}

	byteLen := (curve.Params().BitSize + 7) / 8
	if len(data) != 1+byteLen {
		return nil, nil
	}
	if data[0] != 2 && data[0] != 3 { // compressed form
		return nil, nil
	}
	p := curve.Params().P
	x = new(big.Int).SetBytes(data[1:])
	if x.Cmp(p) >= 0 {
		return nil, nil
	}
	// y² = x³ - 3x + b
	y = curve.Params().polynomial(x)
	y = y.ModSqrt(y, p)
	if y == nil {
		return nil, nil
	}
	if byte(y.Bit(0)) != data[0]&1 {
		y.Neg(y).Mod(y, p)
	}
	if !curve.IsOnCurve(x, y) {
		return nil, nil
	}
	return
}

func panicIfNotOnCurve(curve Curve, x, y *big.Int) {
	// (0, 0) is the point at infinity by convention. It's ok to operate on it,
	// although IsOnCurve is documented to return false for it. See Issue 37294.
	if x.Sign() == 0 && y.Sign() == 0 {
		return
	}

	if !curve.IsOnCurve(x, y) {
		panic("crypto/elliptic: attempted operation on invalid point")
	}
}

var initonce sync.Once

func initAll() {
	initP224()
	initP256()
	initP384()
	initP521()
}

// P224 returns a [Curve] which implements NIST P-224 (FIPS 186-3, section D.2.2),
// also known as secp224r1. The CurveParams.Name of this [Curve] is "P-224".
//
// Multiple invocations of this function will return the same value, so it can
// be used for equality checks and switch statements.
//
// The cryptographic operations are implemented using constant-time algorithms.
func P224() Curve {
	initonce.Do(initAll)
	return p224
}

// P256 returns a [Curve] which implements NIST P-256 (FIPS 186-3, section D.2.3),
// also known as secp256r1 or prime256v1. The CurveParams.Name of this [Curve] is
// "P-256".
//
// Multiple invocations of this function will return the same value, so it can
// be used for equality checks and switch statements.
//
// The cryptographic operations are implemented using constant-time algorithms.
func P256() Curve {
	initonce.Do(initAll)
	return p256
}

// P384 returns a [Curve] which implements NIST P-384 (FIPS 186-3, section D.2.4),
// also known as secp384r1. The CurveParams.Name of this [Curve] is "P-384".
//
// Multiple invocations of this function will return the same value, so it can
// be used for equality checks and switch statements.
//
// The cryptographic operations are implemented using constant-time algorithms.
func P384() Curve {
	initonce.Do(initAll)
	return p384
}

// P521 returns a [Curve] which implements NIST P-521 (FIPS 186-3, section D.2.5),
// also known as secp521r1. The CurveParams.Name of this [Curve] is "P-521".
//
// Multiple invocations of this function will return the same value, so it can
// be used for equality checks and switch statements.
//
// The cryptographic operations are implemented using constant-time algorithms.
func P521() Curve {
	initonce.Do(initAll)
	return p521
}

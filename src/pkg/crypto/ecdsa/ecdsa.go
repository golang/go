// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ecdsa implements the Elliptic Curve Digital Signature Algorithm, as
// defined in FIPS 186-3.
package ecdsa

// References:
//   [NSA]: Suite B implementor's guide to FIPS 186-3,
//     http://www.nsa.gov/ia/_files/ecdsa.pdf

import (
	"big"
	"crypto/elliptic"
	"io"
	"os"
)

// PublicKey represents an ECDSA public key.
type PublicKey struct {
	*elliptic.Curve
	X, Y *big.Int
}

// PrivateKey represents a ECDSA private key.
type PrivateKey struct {
	PublicKey
	D *big.Int
}

var one = new(big.Int).SetInt64(1)

// randFieldElement returns a random element of the field underlying the given
// curve using the procedure given in [NSA] A.2.1.
func randFieldElement(c *elliptic.Curve, rand io.Reader) (k *big.Int, err os.Error) {
	b := make([]byte, c.BitSize/8+8)
	_, err = rand.Read(b)
	if err != nil {
		return
	}

	k = new(big.Int).SetBytes(b)
	n := new(big.Int).Sub(c.N, one)
	k.Mod(k, n)
	k.Add(k, one)
	return
}

// GenerateKey generates a public&private key pair.
func GenerateKey(c *elliptic.Curve, rand io.Reader) (priv *PrivateKey, err os.Error) {
	k, err := randFieldElement(c, rand)
	if err != nil {
		return
	}

	priv = new(PrivateKey)
	priv.PublicKey.Curve = c
	priv.D = k
	priv.PublicKey.X, priv.PublicKey.Y = c.ScalarBaseMult(k.Bytes())
	return
}

// Sign signs an arbitrary length hash (which should be the result of hashing a
// larger message) using the private key, priv. It returns the signature as a
// pair of integers. The security of the private key depends on the entropy of
// rand.
func Sign(rand io.Reader, priv *PrivateKey, hash []byte) (r, s *big.Int, err os.Error) {
	// See [NSA] 3.4.1
	c := priv.PublicKey.Curve

	var k, kInv *big.Int
	for {
		for {
			k, err = randFieldElement(c, rand)
			if err != nil {
				r = nil
				return
			}

			kInv = new(big.Int).ModInverse(k, c.N)
			r, _ = priv.Curve.ScalarBaseMult(k.Bytes())
			r.Mod(r, priv.Curve.N)
			if r.Sign() != 0 {
				break
			}
		}

		e := new(big.Int).SetBytes(hash)
		s = new(big.Int).Mul(priv.D, r)
		s.Add(s, e)
		s.Mul(s, kInv)
		s.Mod(s, priv.PublicKey.Curve.N)
		if s.Sign() != 0 {
			break
		}
	}

	return
}

// Verify verifies the signature in r, s of hash using the public key, pub. It
// returns true iff the signature is valid.
func Verify(pub *PublicKey, hash []byte, r, s *big.Int) bool {
	// See [NSA] 3.4.2
	c := pub.Curve

	if r.Sign() == 0 || s.Sign() == 0 {
		return false
	}
	if r.Cmp(c.N) >= 0 || s.Cmp(c.N) >= 0 {
		return false
	}
	e := new(big.Int).SetBytes(hash)
	w := new(big.Int).ModInverse(s, c.N)

	u1 := e.Mul(e, w)
	u2 := w.Mul(r, w)

	x1, y1 := c.ScalarBaseMult(u1.Bytes())
	x2, y2 := c.ScalarMult(pub.X, pub.Y, u2.Bytes())
	if x1.Cmp(x2) == 0 {
		return false
	}
	x, _ := c.Add(x1, y1, x2, y2)
	x.Mod(x, c.N)
	return x.Cmp(r) == 0
}

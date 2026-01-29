// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdh

import (
	"bytes"
	"crypto/internal/fips140"
	"crypto/internal/fips140/drbg"
	"crypto/internal/fips140/nistec"
	"crypto/internal/fips140deps/byteorder"
	"errors"
	"io"
	"math/bits"
)

// PrivateKey and PublicKey are not generic to make it possible to use them
// in other types without instantiating them with a specific point type.
// They are tied to one of the Curve types below through the curveID field.

// All this is duplicated from crypto/internal/fips/ecdsa, but the standards are
// different and FIPS 140 does not allow reusing keys across them.

type PrivateKey struct {
	pub PublicKey
	d   []byte // bigmod.(*Nat).Bytes output (fixed length)
}

func (priv *PrivateKey) Bytes() []byte {
	return priv.d
}

func (priv *PrivateKey) PublicKey() *PublicKey {
	return &priv.pub
}

type PublicKey struct {
	curve curveID
	q     []byte // uncompressed nistec Point.Bytes output
}

func (pub *PublicKey) Bytes() []byte {
	return pub.q
}

type curveID string

const (
	p224 curveID = "P-224"
	p256 curveID = "P-256"
	p384 curveID = "P-384"
	p521 curveID = "P-521"
)

type Curve[P Point[P]] struct {
	curve    curveID
	newPoint func() P
	N        []byte
}

// Point is a generic constraint for the [nistec] Point types.
type Point[P any] interface {
	*nistec.P224Point | *nistec.P256Point | *nistec.P384Point | *nistec.P521Point
	Bytes() []byte
	BytesX() ([]byte, error)
	SetBytes([]byte) (P, error)
	ScalarMult(P, []byte) (P, error)
	ScalarBaseMult([]byte) (P, error)
}

func P224() *Curve[*nistec.P224Point] {
	return &Curve[*nistec.P224Point]{
		curve:    p224,
		newPoint: nistec.NewP224Point,
		N:        p224Order,
	}
}

var p224Order = []byte{
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x16, 0xa2,
	0xe0, 0xb8, 0xf0, 0x3e, 0x13, 0xdd, 0x29, 0x45,
	0x5c, 0x5c, 0x2a, 0x3d,
}

func P256() *Curve[*nistec.P256Point] {
	return &Curve[*nistec.P256Point]{
		curve:    p256,
		newPoint: nistec.NewP256Point,
		N:        p256Order,
	}
}

var p256Order = []byte{
	0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xbc, 0xe6, 0xfa, 0xad, 0xa7, 0x17, 0x9e, 0x84,
	0xf3, 0xb9, 0xca, 0xc2, 0xfc, 0x63, 0x25, 0x51,
}

func P384() *Curve[*nistec.P384Point] {
	return &Curve[*nistec.P384Point]{
		curve:    p384,
		newPoint: nistec.NewP384Point,
		N:        p384Order,
	}
}

var p384Order = []byte{
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xc7, 0x63, 0x4d, 0x81, 0xf4, 0x37, 0x2d, 0xdf,
	0x58, 0x1a, 0x0d, 0xb2, 0x48, 0xb0, 0xa7, 0x7a,
	0xec, 0xec, 0x19, 0x6a, 0xcc, 0xc5, 0x29, 0x73,
}

func P521() *Curve[*nistec.P521Point] {
	return &Curve[*nistec.P521Point]{
		curve:    p521,
		newPoint: nistec.NewP521Point,
		N:        p521Order,
	}
}

var p521Order = []byte{0x01, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfa,
	0x51, 0x86, 0x87, 0x83, 0xbf, 0x2f, 0x96, 0x6b,
	0x7f, 0xcc, 0x01, 0x48, 0xf7, 0x09, 0xa5, 0xd0,
	0x3b, 0xb5, 0xc9, 0xb8, 0x89, 0x9c, 0x47, 0xae,
	0xbb, 0x6f, 0xb7, 0x1e, 0x91, 0x38, 0x64, 0x09,
}

// GenerateKey generates a new ECDSA private key pair for the specified curve.
func GenerateKey[P Point[P]](c *Curve[P], rand io.Reader) (*PrivateKey, error) {
	fips140.RecordApproved()
	// This procedure is equivalent to Key Pair Generation by Testing
	// Candidates, specified in NIST SP 800-56A Rev. 3, Section 5.6.1.2.2.

	for {
		key := make([]byte, len(c.N))
		if err := drbg.ReadWithReader(rand, key); err != nil {
			return nil, err
		}
		// In tests, rand will return all zeros and NewPrivateKey will reject
		// the zero key as it generates the identity as a public key. This also
		// makes this function consistent with crypto/elliptic.GenerateKey.
		key[1] ^= 0x42

		// Mask off any excess bits if the size of the underlying field is not a
		// whole number of bytes, which is only the case for P-521.
		if c.curve == p521 && c.N[0]&0b1111_1110 == 0 {
			key[0] &= 0b0000_0001
		}

		privateKey, err := NewPrivateKey(c, key)
		if err != nil {
			continue
		}

		// A "Pairwise Consistency Test" makes no sense if we just generated the
		// public key from an ephemeral private key. Moreover, there is no way to
		// check it aside from redoing the exact same computation again. SP 800-56A
		// Rev. 3, Section 5.6.2.1.4 acknowledges that, and doesn't require it.
		// However, ISO 19790:2012, Section 7.10.3.3 has a blanket requirement for a
		// PCT for all generated keys (AS10.35) and FIPS 140-3 IG 10.3.A, Additional
		// Comment 1 goes out of its way to say that "the PCT shall be performed
		// consistent [...], even if the underlying standard does not require a
		// PCT". So we do it. And make ECDH nearly 50% slower (only) in FIPS mode.
		fips140.PCT("ECDH PCT", func() error {
			p1, err := c.newPoint().ScalarBaseMult(privateKey.d)
			if err != nil {
				return err
			}
			if !bytes.Equal(p1.Bytes(), privateKey.pub.q) {
				return errors.New("crypto/ecdh: public key does not match private key")
			}
			return nil
		})

		return privateKey, nil
	}
}

func NewPrivateKey[P Point[P]](c *Curve[P], key []byte) (*PrivateKey, error) {
	// SP 800-56A Rev. 3, Section 5.6.1.2.2 checks that c <= n – 2 and then
	// returns d = c + 1. Note that it follows that 0 < d < n. Equivalently,
	// we check that 0 < d < n, and return d.
	if len(key) != len(c.N) || isZero(key) || !isLess(key, c.N) {
		return nil, errors.New("crypto/ecdh: invalid private key")
	}

	p, err := c.newPoint().ScalarBaseMult(key)
	if err != nil {
		// This is unreachable because the only error condition of
		// ScalarBaseMult is if the input is not the right size.
		panic("crypto/ecdh: internal error: nistec ScalarBaseMult failed for a fixed-size input")
	}

	publicKey := p.Bytes()
	if len(publicKey) == 1 {
		// The encoding of the identity is a single 0x00 byte. This is
		// unreachable because the only scalar that generates the identity is
		// zero, which is rejected above.
		panic("crypto/ecdh: internal error: public key is the identity element")
	}

	k := &PrivateKey{d: bytes.Clone(key), pub: PublicKey{curve: c.curve, q: publicKey}}
	return k, nil
}

func NewPublicKey[P Point[P]](c *Curve[P], key []byte) (*PublicKey, error) {
	// Reject the point at infinity and compressed encodings.
	if len(key) == 0 || key[0] != 4 {
		return nil, errors.New("crypto/ecdh: invalid public key")
	}

	// SetBytes checks that x and y are in the interval [0, p - 1], and that
	// the point is on the curve. Along with the rejection of the point at
	// infinity (the identity element) above, this fulfills the requirements
	// of NIST SP 800-56A Rev. 3, Section 5.6.2.3.4.
	if _, err := c.newPoint().SetBytes(key); err != nil {
		return nil, err
	}

	return &PublicKey{curve: c.curve, q: bytes.Clone(key)}, nil
}

func ECDH[P Point[P]](c *Curve[P], k *PrivateKey, peer *PublicKey) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	return ecdh(c, k, peer)
}

func ecdh[P Point[P]](c *Curve[P], k *PrivateKey, peer *PublicKey) ([]byte, error) {
	if c.curve != k.pub.curve {
		return nil, errors.New("crypto/ecdh: mismatched curves")
	}
	if k.pub.curve != peer.curve {
		return nil, errors.New("crypto/ecdh: mismatched curves")
	}

	// This applies the Shared Secret Computation of the Ephemeral Unified Model
	// scheme specified in NIST SP 800-56A Rev. 3, Section 6.1.2.2.

	// Per Section 5.6.2.3.4, Step 1, reject the identity element (0x00).
	if len(k.pub.q) == 1 {
		return nil, errors.New("crypto/ecdh: public key is the identity element")
	}

	// SetBytes checks that (x, y) are reduced modulo p, and that they are on
	// the curve, performing Steps 2-3 of Section 5.6.2.3.4.
	p, err := c.newPoint().SetBytes(peer.q)
	if err != nil {
		return nil, err
	}

	// Compute P according to Section 5.7.1.2.
	if _, err := p.ScalarMult(p, k.d); err != nil {
		return nil, err
	}

	// BytesX checks that the result is not the identity element, and returns the
	// x-coordinate of the result, performing Steps 2-5 of Section 5.7.1.2.
	return p.BytesX()
}

// isZero reports whether x is all zeroes in constant time.
func isZero(x []byte) bool {
	var acc byte
	for _, b := range x {
		acc |= b
	}
	return acc == 0
}

// isLess reports whether a < b, where a and b are big-endian buffers of the
// same length and shorter than 72 bytes.
func isLess(a, b []byte) bool {
	if len(a) != len(b) {
		panic("crypto/ecdh: internal error: mismatched isLess inputs")
	}

	// Copy the values into a fixed-size preallocated little-endian buffer.
	// 72 bytes is enough for every scalar in this package, and having a fixed
	// size lets us avoid heap allocations.
	if len(a) > 72 {
		panic("crypto/ecdh: internal error: isLess input too large")
	}
	bufA, bufB := make([]byte, 72), make([]byte, 72)
	for i := range a {
		bufA[i], bufB[i] = a[len(a)-i-1], b[len(b)-i-1]
	}

	// Perform a subtraction with borrow.
	var borrow uint64
	for i := 0; i < len(bufA); i += 8 {
		limbA, limbB := byteorder.LEUint64(bufA[i:]), byteorder.LEUint64(bufB[i:])
		_, borrow = bits.Sub64(limbA, limbB, borrow)
	}

	// If there is a borrow at the end of the operation, then a < b.
	return borrow == 1
}

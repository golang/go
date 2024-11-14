// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdh

import (
	"bytes"
	"crypto/internal/edwards25519/field"
	"crypto/internal/randutil"
	"errors"
	"io"
)

var (
	x25519PublicKeySize    = 32
	x25519PrivateKeySize   = 32
	x25519SharedSecretSize = 32
)

// X25519 returns a [Curve] which implements the X25519 function over Curve25519
// (RFC 7748, Section 5).
//
// Multiple invocations of this function will return the same value, so it can
// be used for equality checks and switch statements.
func X25519() Curve { return x25519 }

var x25519 = &x25519Curve{}

type x25519Curve struct{}

func (c *x25519Curve) String() string {
	return "X25519"
}

func (c *x25519Curve) GenerateKey(rand io.Reader) (*PrivateKey, error) {
	key := make([]byte, x25519PrivateKeySize)
	randutil.MaybeReadByte(rand)
	if _, err := io.ReadFull(rand, key); err != nil {
		return nil, err
	}
	return c.NewPrivateKey(key)
}

func (c *x25519Curve) NewPrivateKey(key []byte) (*PrivateKey, error) {
	if len(key) != x25519PrivateKeySize {
		return nil, errors.New("crypto/ecdh: invalid private key size")
	}
	publicKey := make([]byte, x25519PublicKeySize)
	x25519Basepoint := [32]byte{9}
	x25519ScalarMult(publicKey, key, x25519Basepoint[:])
	// We don't check for the all-zero public key here because the scalar is
	// never zero because of clamping, and the basepoint is not the identity in
	// the prime-order subgroup(s).
	return &PrivateKey{
		curve:      c,
		privateKey: bytes.Clone(key),
		publicKey:  &PublicKey{curve: c, publicKey: publicKey},
	}, nil
}

func (c *x25519Curve) NewPublicKey(key []byte) (*PublicKey, error) {
	if len(key) != x25519PublicKeySize {
		return nil, errors.New("crypto/ecdh: invalid public key")
	}
	return &PublicKey{
		curve:     c,
		publicKey: bytes.Clone(key),
	}, nil
}

func (c *x25519Curve) ecdh(local *PrivateKey, remote *PublicKey) ([]byte, error) {
	out := make([]byte, x25519SharedSecretSize)
	x25519ScalarMult(out, local.privateKey, remote.publicKey)
	if isZero(out) {
		return nil, errors.New("crypto/ecdh: bad X25519 remote ECDH input: low order point")
	}
	return out, nil
}

func x25519ScalarMult(dst, scalar, point []byte) {
	var e [32]byte

	copy(e[:], scalar[:])
	e[0] &= 248
	e[31] &= 127
	e[31] |= 64

	var x1, x2, z2, x3, z3, tmp0, tmp1 field.Element
	x1.SetBytes(point[:])
	x2.One()
	x3.Set(&x1)
	z3.One()

	swap := 0
	for pos := 254; pos >= 0; pos-- {
		b := e[pos/8] >> uint(pos&7)
		b &= 1
		swap ^= int(b)
		x2.Swap(&x3, swap)
		z2.Swap(&z3, swap)
		swap = int(b)

		tmp0.Subtract(&x3, &z3)
		tmp1.Subtract(&x2, &z2)
		x2.Add(&x2, &z2)
		z2.Add(&x3, &z3)
		z3.Multiply(&tmp0, &x2)
		z2.Multiply(&z2, &tmp1)
		tmp0.Square(&tmp1)
		tmp1.Square(&x2)
		x3.Add(&z3, &z2)
		z2.Subtract(&z3, &z2)
		x2.Multiply(&tmp1, &tmp0)
		tmp1.Subtract(&tmp1, &tmp0)
		z2.Square(&z2)

		z3.Mult32(&tmp1, 121666)
		x3.Square(&x3)
		tmp0.Add(&tmp0, &z3)
		z3.Multiply(&x1, &z2)
		z2.Multiply(&tmp1, &tmp0)
	}

	x2.Swap(&x3, swap)
	z2.Swap(&z3, swap)

	z2.Invert(&z2)
	x2.Multiply(&x2, &z2)
	copy(dst[:], x2.Bytes())
}

// isZero reports whether x is all zeroes in constant time.
func isZero(x []byte) bool {
	var acc byte
	for _, b := range x {
		acc |= b
	}
	return acc == 0
}

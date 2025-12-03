// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdh

import (
	"bytes"
	"crypto/internal/boring"
	"crypto/internal/fips140/ecdh"
	"crypto/internal/fips140only"
	"crypto/internal/rand"
	"errors"
	"io"
)

type nistCurve struct {
	name          string
	generate      func(io.Reader) (*ecdh.PrivateKey, error)
	newPrivateKey func([]byte) (*ecdh.PrivateKey, error)
	newPublicKey  func(publicKey []byte) (*ecdh.PublicKey, error)
	sharedSecret  func(*ecdh.PrivateKey, *ecdh.PublicKey) (sharedSecret []byte, err error)
}

func (c *nistCurve) String() string {
	return c.name
}

func (c *nistCurve) GenerateKey(r io.Reader) (*PrivateKey, error) {
	if boring.Enabled && rand.IsDefaultReader(r) {
		key, bytes, err := boring.GenerateKeyECDH(c.name)
		if err != nil {
			return nil, err
		}
		pub, err := key.PublicKey()
		if err != nil {
			return nil, err
		}
		k := &PrivateKey{
			curve:      c,
			privateKey: bytes,
			publicKey:  &PublicKey{curve: c, publicKey: pub.Bytes(), boring: pub},
			boring:     key,
		}
		return k, nil
	}

	r = rand.CustomReader(r)

	if fips140only.Enforced() && !fips140only.ApprovedRandomReader(r) {
		return nil, errors.New("crypto/ecdh: only crypto/rand.Reader is allowed in FIPS 140-only mode")
	}

	privateKey, err := c.generate(r)
	if err != nil {
		return nil, err
	}

	k := &PrivateKey{
		curve:      c,
		privateKey: privateKey.Bytes(),
		fips:       privateKey,
		publicKey: &PublicKey{
			curve:     c,
			publicKey: privateKey.PublicKey().Bytes(),
			fips:      privateKey.PublicKey(),
		},
	}
	if boring.Enabled {
		bk, err := boring.NewPrivateKeyECDH(c.name, k.privateKey)
		if err != nil {
			return nil, err
		}
		pub, err := bk.PublicKey()
		if err != nil {
			return nil, err
		}
		k.boring = bk
		k.publicKey.boring = pub
	}
	return k, nil
}

func (c *nistCurve) NewPrivateKey(key []byte) (*PrivateKey, error) {
	if boring.Enabled {
		bk, err := boring.NewPrivateKeyECDH(c.name, key)
		if err != nil {
			return nil, errors.New("crypto/ecdh: invalid private key")
		}
		pub, err := bk.PublicKey()
		if err != nil {
			return nil, errors.New("crypto/ecdh: invalid private key")
		}
		k := &PrivateKey{
			curve:      c,
			privateKey: bytes.Clone(key),
			publicKey:  &PublicKey{curve: c, publicKey: pub.Bytes(), boring: pub},
			boring:     bk,
		}
		return k, nil
	}

	fk, err := c.newPrivateKey(key)
	if err != nil {
		return nil, err
	}
	k := &PrivateKey{
		curve:      c,
		privateKey: bytes.Clone(key),
		fips:       fk,
		publicKey: &PublicKey{
			curve:     c,
			publicKey: fk.PublicKey().Bytes(),
			fips:      fk.PublicKey(),
		},
	}
	return k, nil
}

func (c *nistCurve) NewPublicKey(key []byte) (*PublicKey, error) {
	// Reject the point at infinity and compressed encodings.
	// Note that boring.NewPublicKeyECDH would accept them.
	if len(key) == 0 || key[0] != 4 {
		return nil, errors.New("crypto/ecdh: invalid public key")
	}
	k := &PublicKey{
		curve:     c,
		publicKey: bytes.Clone(key),
	}
	if boring.Enabled {
		bk, err := boring.NewPublicKeyECDH(c.name, k.publicKey)
		if err != nil {
			return nil, errors.New("crypto/ecdh: invalid public key")
		}
		k.boring = bk
	} else {
		fk, err := c.newPublicKey(key)
		if err != nil {
			return nil, err
		}
		k.fips = fk
	}
	return k, nil
}

func (c *nistCurve) ecdh(local *PrivateKey, remote *PublicKey) ([]byte, error) {
	// Note that this function can't return an error, as NewPublicKey rejects
	// invalid points and the point at infinity, and NewPrivateKey rejects
	// invalid scalars and the zero value. BytesX returns an error for the point
	// at infinity, but in a prime order group such as the NIST curves that can
	// only be the result of a scalar multiplication if one of the inputs is the
	// zero scalar or the point at infinity.

	if boring.Enabled {
		return boring.ECDH(local.boring, remote.boring)
	}
	return c.sharedSecret(local.fips, remote.fips)
}

// P256 returns a [Curve] which implements NIST P-256 (FIPS 186-3, section D.2.3),
// also known as secp256r1 or prime256v1.
//
// Multiple invocations of this function will return the same value, which can
// be used for equality checks and switch statements.
func P256() Curve { return p256 }

var p256 = &nistCurve{
	name: "P-256",
	generate: func(r io.Reader) (*ecdh.PrivateKey, error) {
		return ecdh.GenerateKey(ecdh.P256(), r)
	},
	newPrivateKey: func(b []byte) (*ecdh.PrivateKey, error) {
		return ecdh.NewPrivateKey(ecdh.P256(), b)
	},
	newPublicKey: func(publicKey []byte) (*ecdh.PublicKey, error) {
		return ecdh.NewPublicKey(ecdh.P256(), publicKey)
	},
	sharedSecret: func(priv *ecdh.PrivateKey, pub *ecdh.PublicKey) (sharedSecret []byte, err error) {
		return ecdh.ECDH(ecdh.P256(), priv, pub)
	},
}

// P384 returns a [Curve] which implements NIST P-384 (FIPS 186-3, section D.2.4),
// also known as secp384r1.
//
// Multiple invocations of this function will return the same value, which can
// be used for equality checks and switch statements.
func P384() Curve { return p384 }

var p384 = &nistCurve{
	name: "P-384",
	generate: func(r io.Reader) (*ecdh.PrivateKey, error) {
		return ecdh.GenerateKey(ecdh.P384(), r)
	},
	newPrivateKey: func(b []byte) (*ecdh.PrivateKey, error) {
		return ecdh.NewPrivateKey(ecdh.P384(), b)
	},
	newPublicKey: func(publicKey []byte) (*ecdh.PublicKey, error) {
		return ecdh.NewPublicKey(ecdh.P384(), publicKey)
	},
	sharedSecret: func(priv *ecdh.PrivateKey, pub *ecdh.PublicKey) (sharedSecret []byte, err error) {
		return ecdh.ECDH(ecdh.P384(), priv, pub)
	},
}

// P521 returns a [Curve] which implements NIST P-521 (FIPS 186-3, section D.2.5),
// also known as secp521r1.
//
// Multiple invocations of this function will return the same value, which can
// be used for equality checks and switch statements.
func P521() Curve { return p521 }

var p521 = &nistCurve{
	name: "P-521",
	generate: func(r io.Reader) (*ecdh.PrivateKey, error) {
		return ecdh.GenerateKey(ecdh.P521(), r)
	},
	newPrivateKey: func(b []byte) (*ecdh.PrivateKey, error) {
		return ecdh.NewPrivateKey(ecdh.P521(), b)
	},
	newPublicKey: func(publicKey []byte) (*ecdh.PublicKey, error) {
		return ecdh.NewPublicKey(ecdh.P521(), publicKey)
	},
	sharedSecret: func(priv *ecdh.PrivateKey, pub *ecdh.PublicKey) (sharedSecret []byte, err error) {
		return ecdh.ECDH(ecdh.P521(), priv, pub)
	},
}

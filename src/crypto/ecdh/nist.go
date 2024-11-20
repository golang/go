// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdh

import (
	"bytes"
	"crypto/internal/boring"
	"crypto/internal/fips140/ecdh"
	"errors"
	"io"
)

type nistCurve struct {
	name         string
	generate     func(io.Reader) (privateKey, publicKey []byte, err error)
	importKey    func([]byte) (publicKey []byte, err error)
	checkPubkey  func(publicKey []byte) error
	sharedSecret func(privateKey, publicKey []byte) (sharedSecret []byte, err error)
}

func (c *nistCurve) String() string {
	return c.name
}

func (c *nistCurve) GenerateKey(rand io.Reader) (*PrivateKey, error) {
	if boring.Enabled && rand == boring.RandReader {
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

	privateKey, publicKey, err := c.generate(rand)
	if err != nil {
		return nil, err
	}

	k := &PrivateKey{
		curve:      c,
		privateKey: privateKey,
		publicKey:  &PublicKey{curve: c, publicKey: publicKey},
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

	publicKey, err := c.importKey(key)
	if err != nil {
		return nil, err
	}

	k := &PrivateKey{
		curve:      c,
		privateKey: bytes.Clone(key),
		publicKey:  &PublicKey{curve: c, publicKey: publicKey},
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
		if err := c.checkPubkey(k.publicKey); err != nil {
			return nil, err
		}
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
	return c.sharedSecret(local.privateKey, remote.publicKey)
}

// P256 returns a [Curve] which implements NIST P-256 (FIPS 186-3, section D.2.3),
// also known as secp256r1 or prime256v1.
//
// Multiple invocations of this function will return the same value, which can
// be used for equality checks and switch statements.
func P256() Curve { return p256 }

var p256 = &nistCurve{
	name:         "P-256",
	generate:     ecdh.GenerateKeyP256,
	importKey:    ecdh.ImportKeyP256,
	checkPubkey:  ecdh.CheckPublicKeyP256,
	sharedSecret: ecdh.ECDHP256,
}

// P384 returns a [Curve] which implements NIST P-384 (FIPS 186-3, section D.2.4),
// also known as secp384r1.
//
// Multiple invocations of this function will return the same value, which can
// be used for equality checks and switch statements.
func P384() Curve { return p384 }

var p384 = &nistCurve{
	name:         "P-384",
	generate:     ecdh.GenerateKeyP384,
	importKey:    ecdh.ImportKeyP384,
	checkPubkey:  ecdh.CheckPublicKeyP384,
	sharedSecret: ecdh.ECDHP384,
}

// P521 returns a [Curve] which implements NIST P-521 (FIPS 186-3, section D.2.5),
// also known as secp521r1.
//
// Multiple invocations of this function will return the same value, which can
// be used for equality checks and switch statements.
func P521() Curve { return p521 }

var p521 = &nistCurve{
	name:         "P-521",
	generate:     ecdh.GenerateKeyP521,
	importKey:    ecdh.ImportKeyP521,
	checkPubkey:  ecdh.CheckPublicKeyP521,
	sharedSecret: ecdh.ECDHP521,
}

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package ecdsa

import (
	"crypto/internal/boring"
	"crypto/internal/boring/bbig"
	"crypto/internal/boring/bcache"
	"math/big"
)

// Cached conversions from Go PublicKey/PrivateKey to BoringCrypto.
//
// The first operation on a PublicKey or PrivateKey makes a parallel
// BoringCrypto key and saves it in pubCache or privCache.
//
// We could just assume that once used in a Sign or Verify operation,
// a particular key is never again modified, but that has not been a
// stated assumption before. Just in case there is any existing code that
// does modify the key between operations, we save the original values
// alongside the cached BoringCrypto key and check that the real key
// still matches before using the cached key. The theory is that the real
// operations are significantly more expensive than the comparison.

var pubCache bcache.Cache[PublicKey, boringPub]
var privCache bcache.Cache[PrivateKey, boringPriv]

func init() {
	pubCache.Register()
	privCache.Register()
}

type boringPub struct {
	key  *boring.PublicKeyECDSA
	orig PublicKey
}

func boringPublicKey(pub *PublicKey) (*boring.PublicKeyECDSA, error) {
	b := pubCache.Get(pub)
	if b != nil && publicKeyEqual(&b.orig, pub) {
		return b.key, nil
	}

	b = new(boringPub)
	b.orig = copyPublicKey(pub)
	key, err := boring.NewPublicKeyECDSA(b.orig.Curve.Params().Name, bbig.Enc(b.orig.X), bbig.Enc(b.orig.Y))
	if err != nil {
		return nil, err
	}
	b.key = key
	pubCache.Put(pub, b)
	return key, nil
}

type boringPriv struct {
	key  *boring.PrivateKeyECDSA
	orig PrivateKey
}

func boringPrivateKey(priv *PrivateKey) (*boring.PrivateKeyECDSA, error) {
	b := privCache.Get(priv)
	if b != nil && privateKeyEqual(&b.orig, priv) {
		return b.key, nil
	}

	b = new(boringPriv)
	b.orig = copyPrivateKey(priv)
	key, err := boring.NewPrivateKeyECDSA(b.orig.Curve.Params().Name, bbig.Enc(b.orig.X), bbig.Enc(b.orig.Y), bbig.Enc(b.orig.D))
	if err != nil {
		return nil, err
	}
	b.key = key
	privCache.Put(priv, b)
	return key, nil
}

func publicKeyEqual(k1, k2 *PublicKey) bool {
	return k1.X != nil &&
		k1.Curve.Params() == k2.Curve.Params() &&
		k1.X.Cmp(k2.X) == 0 &&
		k1.Y.Cmp(k2.Y) == 0
}

func privateKeyEqual(k1, k2 *PrivateKey) bool {
	return publicKeyEqual(&k1.PublicKey, &k2.PublicKey) &&
		k1.D.Cmp(k2.D) == 0
}

func copyPublicKey(k *PublicKey) PublicKey {
	return PublicKey{
		Curve: k.Curve,
		X:     new(big.Int).Set(k.X),
		Y:     new(big.Int).Set(k.Y),
	}
}

func copyPrivateKey(k *PrivateKey) PrivateKey {
	return PrivateKey{
		PublicKey: copyPublicKey(&k.PublicKey),
		D:         new(big.Int).Set(k.D),
	}
}

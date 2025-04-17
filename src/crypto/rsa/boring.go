// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package rsa

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
// We could just assume that once used in a sign/verify/encrypt/decrypt operation,
// a particular key is never again modified, but that has not been a
// stated assumption before. Just in case there is any existing code that
// does modify the key between operations, we save the original values
// alongside the cached BoringCrypto key and check that the real key
// still matches before using the cached key. The theory is that the real
// operations are significantly more expensive than the comparison.

type boringPub struct {
	key  *boring.PublicKeyRSA
	orig PublicKey
}

var pubCache bcache.Cache[PublicKey, boringPub]
var privCache bcache.Cache[PrivateKey, boringPriv]

func init() {
	pubCache.Register()
	privCache.Register()
}

func boringPublicKey(pub *PublicKey) (*boring.PublicKeyRSA, error) {
	b := pubCache.Get(pub)
	if b != nil && publicKeyEqual(&b.orig, pub) {
		return b.key, nil
	}

	b = new(boringPub)
	b.orig = copyPublicKey(pub)
	key, err := boring.NewPublicKeyRSA(bbig.Enc(b.orig.N), bbig.Enc(big.NewInt(int64(b.orig.E))))
	if err != nil {
		return nil, err
	}
	b.key = key
	pubCache.Put(pub, b)
	return key, nil
}

type boringPriv struct {
	key  *boring.PrivateKeyRSA
	orig PrivateKey
}

func boringPrivateKey(priv *PrivateKey) (*boring.PrivateKeyRSA, error) {
	b := privCache.Get(priv)
	if b != nil && privateKeyEqual(&b.orig, priv) {
		return b.key, nil
	}

	b = new(boringPriv)
	b.orig = copyPrivateKey(priv)

	var N, E, D, P, Q, Dp, Dq, Qinv *big.Int
	N = b.orig.N
	E = big.NewInt(int64(b.orig.E))
	D = b.orig.D
	if len(b.orig.Primes) == 2 {
		P = b.orig.Primes[0]
		Q = b.orig.Primes[1]
		Dp = b.orig.Precomputed.Dp
		Dq = b.orig.Precomputed.Dq
		Qinv = b.orig.Precomputed.Qinv
	}
	key, err := boring.NewPrivateKeyRSA(bbig.Enc(N), bbig.Enc(E), bbig.Enc(D), bbig.Enc(P), bbig.Enc(Q), bbig.Enc(Dp), bbig.Enc(Dq), bbig.Enc(Qinv))
	if err != nil {
		return nil, err
	}
	b.key = key
	privCache.Put(priv, b)
	return key, nil
}

func publicKeyEqual(k1, k2 *PublicKey) bool {
	return k1.N != nil &&
		k1.N.Cmp(k2.N) == 0 &&
		k1.E == k2.E
}

func copyPublicKey(k *PublicKey) PublicKey {
	return PublicKey{
		N: new(big.Int).Set(k.N),
		E: k.E,
	}
}

func privateKeyEqual(k1, k2 *PrivateKey) bool {
	return publicKeyEqual(&k1.PublicKey, &k2.PublicKey) &&
		k1.D.Cmp(k2.D) == 0
}

func copyPrivateKey(k *PrivateKey) PrivateKey {
	dst := PrivateKey{
		PublicKey: copyPublicKey(&k.PublicKey),
		D:         new(big.Int).Set(k.D),
	}
	dst.Primes = make([]*big.Int, len(k.Primes))
	for i, p := range k.Primes {
		dst.Primes[i] = new(big.Int).Set(p)
	}
	if x := k.Precomputed.Dp; x != nil {
		dst.Precomputed.Dp = new(big.Int).Set(x)
	}
	if x := k.Precomputed.Dq; x != nil {
		dst.Precomputed.Dq = new(big.Int).Set(x)
	}
	if x := k.Precomputed.Qinv; x != nil {
		dst.Precomputed.Qinv = new(big.Int).Set(x)
	}
	return dst
}

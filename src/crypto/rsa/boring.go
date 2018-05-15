// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"crypto/internal/boring"
	"crypto/rand"
	"io"
	"math/big"
	"sync/atomic"
	"unsafe"
)

// Cached conversions from Go PublicKey/PrivateKey to BoringCrypto.
//
// A new 'boring atomic.Value' field in both PublicKey and PrivateKey
// serves as a cache for the most recent conversion. The cache is an
// atomic.Value because code might reasonably set up a key and then
// (thinking it immutable) use it from multiple goroutines simultaneously.
// The first operation initializes the cache; if there are multiple simultaneous
// first operations, they will do redundant work but not step on each other.
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

func boringPublicKey(pub *PublicKey) (*boring.PublicKeyRSA, error) {
	b := (*boringPub)(atomic.LoadPointer(&pub.boring))
	if b != nil && publicKeyEqual(&b.orig, pub) {
		return b.key, nil
	}

	b = new(boringPub)
	b.orig = copyPublicKey(pub)
	key, err := boring.NewPublicKeyRSA(b.orig.N, big.NewInt(int64(b.orig.E)))
	if err != nil {
		return nil, err
	}
	b.key = key
	atomic.StorePointer(&pub.boring, unsafe.Pointer(b))
	return key, nil
}

type boringPriv struct {
	key  *boring.PrivateKeyRSA
	orig PrivateKey
}

func boringPrivateKey(priv *PrivateKey) (*boring.PrivateKeyRSA, error) {
	b := (*boringPriv)(atomic.LoadPointer(&priv.boring))
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
	key, err := boring.NewPrivateKeyRSA(N, E, D, P, Q, Dp, Dq, Qinv)
	if err != nil {
		return nil, err
	}
	b.key = key
	atomic.StorePointer(&priv.boring, unsafe.Pointer(b))
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

// boringFakeRandomBlind consumes from random to mimic the
// blinding operation done in the standard Go func decrypt.
// When we are using BoringCrypto, we always let it handle decrypt
// regardless of random source, because the blind doesn't affect
// the visible output of decryption, but if the random source is not
// true randomness then the caller might still observe the side effect
// of consuming from the source. We consume from the source
// to give the same side effect. This should only happen during tests
// (verified by the UnreachableExceptTests call below).
//
// We go to the trouble of doing this so that we can verify that
// func decrypt (standard RSA decryption) is dropped from
// BoringCrypto-linked binaries entirely; otherwise we'd have to
// keep it in the binary just in case a call happened with a
// non-standard randomness source.
func boringFakeRandomBlind(random io.Reader, priv *PrivateKey) {
	if random == nil || random == boring.RandReader {
		return
	}
	boring.UnreachableExceptTests()

	// Copied from func decrypt.
	ir := new(big.Int)
	for {
		r, err := rand.Int(random, priv.N)
		if err != nil {
			return
		}
		if r.Cmp(bigZero) == 0 {
			r = bigOne
		}
		ok := ir.ModInverse(r, priv.N)
		if ok != nil {
			break
		}
	}
}

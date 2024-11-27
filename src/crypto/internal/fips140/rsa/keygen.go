// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"crypto/internal/fips140/bigmod"
	"crypto/internal/fips140/drbg"
	"errors"
)

// isPrime runs the Miller-Rabin Probabilistic Primality Test from
// FIPS 186-5, Appendix B.3.1.
//
// w must be a random odd integer greater than three in big-endian order.
// isPrime might return false positives for adversarially chosen values.
//
// isPrime is not constant-time.
func isPrime(w []byte) bool {
	mr, err := millerRabinSetup(w)
	if err != nil {
		// w is zero, one, or even.
		return false
	}

	// iterations is the number of Miller-Rabin rounds, each with a
	// randomly-selected base.
	//
	// The worst case false positive rate for a single iteration is 1/4 per
	// https://eprint.iacr.org/2018/749, so if w were selected adversarially, we
	// would need up to 64 iterations to get to a negligible (2⁻¹²⁸) chance of
	// false positive.
	//
	// However, since this function is only used for randomly-selected w in the
	// context of RSA key generation, we can use a smaller number of iterations.
	// The exact number depends on the size of the prime (and the implied
	// security level). See BoringSSL for the full formula.
	// https://cs.opensource.google/boringssl/boringssl/+/master:crypto/fipsmodule/bn/prime.c.inc;l=208-283;drc=3a138e43
	bits := mr.w.BitLen()
	var iterations int
	switch {
	case bits >= 3747:
		iterations = 3
	case bits >= 1345:
		iterations = 4
	case bits >= 476:
		iterations = 5
	case bits >= 400:
		iterations = 6
	case bits >= 347:
		iterations = 7
	case bits >= 308:
		iterations = 8
	case bits >= 55:
		iterations = 27
	default:
		iterations = 34
	}

	b := make([]byte, (bits+7)/8)
	for {
		drbg.Read(b)
		if excess := len(b)*8 - bits; excess != 0 {
			b[0] >>= excess
		}
		result, err := millerRabinIteration(mr, b)
		if err != nil {
			// b was rejected.
			continue
		}
		if result == millerRabinCOMPOSITE {
			return false
		}
		iterations--
		if iterations == 0 {
			return true
		}
	}
}

type millerRabin struct {
	w *bigmod.Modulus
	a uint
	m []byte
}

// millerRabinSetup prepares state that's reused across multiple iterations of
// the Miller-Rabin test.
func millerRabinSetup(w []byte) (*millerRabin, error) {
	mr := &millerRabin{}

	// Check that w is odd, and precompute Montgomery parameters.
	wm, err := bigmod.NewModulus(w)
	if err != nil {
		return nil, err
	}
	if wm.Nat().IsOdd() == 0 {
		return nil, errors.New("candidate is even")
	}
	mr.w = wm

	// Compute m = (w-1)/2^a, where m is odd.
	wMinus1 := mr.w.Nat().SubOne(mr.w)
	if wMinus1.IsZero() == 1 {
		return nil, errors.New("candidate is one")
	}
	mr.a = wMinus1.TrailingZeroBitsVarTime()

	// Store mr.m as a big-endian byte slice with leading zero bytes removed,
	// for use with [bigmod.Nat.Exp].
	m := wMinus1.ShiftRightVarTime(mr.a)
	mr.m = m.Bytes(mr.w)
	for mr.m[0] == 0 {
		mr.m = mr.m[1:]
	}

	return mr, nil
}

const millerRabinCOMPOSITE = false
const millerRabinPOSSIBLYPRIME = true

func millerRabinIteration(mr *millerRabin, bb []byte) (bool, error) {
	// Reject b ≤ 1 or b ≥ w − 1.
	if len(bb) != (mr.w.BitLen()+7)/8 {
		return false, errors.New("incorrect length")
	}
	b := bigmod.NewNat()
	if _, err := b.SetBytes(bb, mr.w); err != nil {
		return false, err
	}
	if b.IsZero() == 1 || b.IsOne() == 1 || b.IsMinusOne(mr.w) == 1 {
		return false, errors.New("out-of-range candidate")
	}

	// Compute b^(m*2^i) mod w for successive i.
	// If b^m mod w = 1, b is a possible prime.
	// If b^(m*2^i) mod w = -1 for some 0 <= i < a, b is a possible prime.
	// Otherwise b is composite.

	// Start by computing and checking b^m mod w (also the i = 0 case).
	z := bigmod.NewNat().Exp(b, mr.m, mr.w)
	if z.IsOne() == 1 || z.IsMinusOne(mr.w) == 1 {
		return millerRabinPOSSIBLYPRIME, nil
	}

	// Check b^(m*2^i) mod w = -1 for 0 < i < a.
	for range mr.a - 1 {
		z.Mul(z, mr.w)
		if z.IsMinusOne(mr.w) == 1 {
			return millerRabinPOSSIBLYPRIME, nil
		}
		if z.IsOne() == 1 {
			// Future squaring will not turn z == 1 into -1.
			break
		}
	}

	return millerRabinCOMPOSITE, nil
}

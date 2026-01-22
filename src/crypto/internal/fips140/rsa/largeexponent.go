// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"bytes"
	"crypto/internal/fips140"
	"crypto/internal/fips140/bigmod"
	"errors"
	"hash"
)

// TestingOnlyLargeExponentPublicKey is a variant of [PublicKey] that supports
// large public exponents. It is only meant for supporting the full ACVP test
// suite, which unfortunately forces us to choose between a fixed exponent and
// the full (2¹⁶, 2²⁵⁶) range. This type must not be used in production code,
// nor for e values < 2³¹, which instead must use [PublicKey].
type TestingOnlyLargeExponentPublicKey struct {
	N *bigmod.Modulus
	// E is the public exponent, represented as a big-endian byte slice.
	E []byte
}

func (pub *TestingOnlyLargeExponentPublicKey) Size() int {
	return (pub.N.BitLen() + 7) / 8
}

func checkLargeExponentPublicKey(pub *TestingOnlyLargeExponentPublicKey) error {
	if pub.N == nil {
		return errors.New("crypto/rsa: missing public modulus")
	}
	if pub.N.Nat().IsOdd() == 0 {
		return errors.New("crypto/rsa: public modulus is even")
	}
	if pub.N.BitLen() < 2048 {
		return errors.New("crypto/rsa: public modulus too small")
	}
	if pub.N.BitLen()%2 == 1 {
		return errors.New("crypto/rsa: public modulus bit length not even")
	}
	E := pub.E
	for len(E) > 0 && E[0] == 0 {
		E = E[1:]
	}
	if len(E) < 32/8 || (len(E) == 32/8 && E[0] < 0x80) {
		// Exponents less than 2^31 must use [PublicKey].
		return errors.New("crypto/rsa: public exponent too small")
	}
	if len(E) > 256/8 {
		return errors.New("crypto/rsa: public exponent too large")
	}
	if E[len(E)-1]&1 == 0 {
		return errors.New("crypto/rsa: public exponent is even")
	}
	return nil
}

func encryptLargeExponent(pub *TestingOnlyLargeExponentPublicKey, plaintext []byte) ([]byte, error) {
	m, err := bigmod.NewNat().SetBytes(plaintext, pub.N)
	if err != nil {
		return nil, err
	}
	return bigmod.NewNat().Exp(m, pub.E, pub.N).Bytes(pub.N), nil
}

func TestingOnlyLargeExponentVerifyPKCS1v15(pub *TestingOnlyLargeExponentPublicKey, hash string, hashed []byte, sig []byte) error {
	fipsSelfTest()
	fips140.RecordApproved()
	checkApprovedHashName(hash)
	if err := checkLargeExponentPublicKey(pub); err != nil {
		return err
	}
	if pub.Size() != len(sig) {
		return ErrVerification
	}
	em, err := encryptLargeExponent(pub, sig)
	if err != nil {
		return ErrVerification
	}
	expected, err := pkcs1v15ConstructEM(&PublicKey{N: pub.N}, hash, hashed)
	if err != nil {
		return ErrVerification
	}
	if !bytes.Equal(em, expected) {
		return ErrVerification
	}
	return nil
}

func TestingOnlyLargeExponentVerifyPSS(pub *TestingOnlyLargeExponentPublicKey, hash hash.Hash, digest []byte, sig []byte) error {
	fipsSelfTest()
	fips140.RecordApproved()
	checkApprovedHash(hash)
	if err := checkLargeExponentPublicKey(pub); err != nil {
		return err
	}
	if len(sig) != pub.Size() {
		return ErrVerification
	}
	emBits := pub.N.BitLen() - 1
	emLen := (emBits + 7) / 8
	em, err := encryptLargeExponent(pub, sig)
	if err != nil {
		return ErrVerification
	}
	for len(em) > emLen && len(em) > 0 {
		if em[0] != 0 {
			return ErrVerification
		}
		em = em[1:]
	}
	return emsaPSSVerify(digest, em, emBits, pssSaltLengthAutodetect, hash)
}

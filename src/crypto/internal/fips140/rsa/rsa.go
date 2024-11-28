// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/bigmod"
	"errors"
)

type PublicKey struct {
	N *bigmod.Modulus
	E int
}

// Size returns the modulus size in bytes. Raw signatures and ciphertexts
// for or by this public key will have the same size.
func (pub *PublicKey) Size() int {
	return (pub.N.BitLen() + 7) / 8
}

type PrivateKey struct {
	// pub has already been checked with checkPublicKey.
	pub PublicKey
	d   *bigmod.Nat
	// The following values are not set for deprecated multi-prime keys.
	//
	// Since they are always set for keys in FIPS mode, for SP 800-56B Rev. 2
	// purposes we always use the Chinese Remainder Theorem (CRT) format.
	p, q *bigmod.Modulus // p × q = n
	// dP and dQ are used as exponents, so we store them as big-endian byte
	// slices to be passed to [bigmod.Nat.Exp].
	dP   []byte      // d mod (p – 1)
	dQ   []byte      // d mod (q – 1)
	qInv *bigmod.Nat // qInv = q⁻¹ mod p
}

func (priv *PrivateKey) PublicKey() *PublicKey {
	return &priv.pub
}

// NewPrivateKey creates a new RSA private key from the given parameters.
//
// All values are in big-endian byte slice format, and may have leading zeros
// or be shorter if leading zeroes were trimmed.
//
// N, e, d, P, and Q are required. dP, dQ, and qInv can be nil and will be
// precomputed if missing.
func NewPrivateKey(N []byte, e int, d, P, Q, dP, dQ, qInv []byte) (*PrivateKey, error) {
	n, err := bigmod.NewModulus(N)
	if err != nil {
		return nil, err
	}
	p, err := bigmod.NewModulus(P)
	if err != nil {
		return nil, err
	}
	q, err := bigmod.NewModulus(Q)
	if err != nil {
		return nil, err
	}
	dN, err := bigmod.NewNat().SetBytes(d, n)
	if err != nil {
		return nil, err
	}
	// TODO(filippo): implement CRT computation. For now, NewPrivateKey is
	// always called with CRT values.
	if dP == nil || dQ == nil || qInv == nil {
		panic("crypto/internal/fips140/rsa: internal error: missing CRT parameters")
	}
	qInvN, err := bigmod.NewNat().SetBytes(qInv, p)
	if err != nil {
		return nil, err
	}
	pk := &PrivateKey{
		pub: PublicKey{
			N: n, E: e,
		},
		d: dN, p: p, q: q,
		dP: dP, dQ: dQ, qInv: qInvN,
	}
	if err := checkPublicKey(&pk.pub); err != nil {
		return nil, err
	}
	return pk, nil
}

// NewPrivateKeyWithoutCRT creates a new RSA private key from the given parameters.
//
// This is meant for deprecated multi-prime keys, and is not FIPS 140 compliant.
func NewPrivateKeyWithoutCRT(N []byte, e int, d []byte) (*PrivateKey, error) {
	n, err := bigmod.NewModulus(N)
	if err != nil {
		return nil, err
	}
	dN, err := bigmod.NewNat().SetBytes(d, n)
	if err != nil {
		return nil, err
	}
	pk := &PrivateKey{
		pub: PublicKey{
			N: n, E: e,
		},
		d: dN,
	}
	if err := checkPublicKey(&pk.pub); err != nil {
		return nil, err
	}
	return pk, nil
}

func checkPublicKey(pub *PublicKey) error {
	if pub.N == nil {
		return errors.New("crypto/rsa: missing public modulus")
	}
	if pub.N.Nat().IsOdd() == 0 {
		return errors.New("crypto/rsa: public modulus is even")
	}
	if pub.N.BitLen() < 2048 || pub.N.BitLen() > 16384 {
		fips140.RecordNonApproved()
	}
	if pub.E < 2 {
		return errors.New("crypto/rsa: public exponent too small or negative")
	}
	// e needs to be coprime with p-1 and q-1, since it must be invertible
	// modulo λ(pq). Since p and q are prime, this means e needs to be odd.
	if pub.E&1 == 0 {
		return errors.New("crypto/rsa: public exponent is even")
	}
	// FIPS 186-5, Section 5.5(e): "The exponent e shall be an odd, positive
	// integer such that 2¹⁶ < e < 2²⁵⁶."
	if pub.E <= 1<<16 {
		fips140.RecordNonApproved()
	}
	// We require pub.E to fit into a 32-bit integer so that we
	// do not have different behavior depending on whether
	// int is 32 or 64 bits. See also
	// https://www.imperialviolet.org/2012/03/16/rsae.html.
	if pub.E > 1<<31-1 {
		return errors.New("crypto/rsa: public exponent too large")
	}
	return nil
}

// Encrypt performs the RSA public key operation.
func Encrypt(pub *PublicKey, plaintext []byte) ([]byte, error) {
	fips140.RecordNonApproved()
	if err := checkPublicKey(pub); err != nil {
		return nil, err
	}
	return encrypt(pub, plaintext)
}

func encrypt(pub *PublicKey, plaintext []byte) ([]byte, error) {
	m, err := bigmod.NewNat().SetBytes(plaintext, pub.N)
	if err != nil {
		return nil, err
	}
	return bigmod.NewNat().ExpShortVarTime(m, uint(pub.E), pub.N).Bytes(pub.N), nil
}

var ErrMessageTooLong = errors.New("crypto/rsa: message too long for RSA key size")
var ErrDecryption = errors.New("crypto/rsa: decryption error")
var ErrVerification = errors.New("crypto/rsa: verification error")

const withCheck = true
const noCheck = false

// DecryptWithoutCheck performs the RSA private key operation.
func DecryptWithoutCheck(priv *PrivateKey, ciphertext []byte) ([]byte, error) {
	fips140.RecordNonApproved()
	return decrypt(priv, ciphertext, noCheck)
}

// DecryptWithCheck performs the RSA private key operation and checks the
// result to defend against errors in the CRT computation.
func DecryptWithCheck(priv *PrivateKey, ciphertext []byte) ([]byte, error) {
	fips140.RecordNonApproved()
	return decrypt(priv, ciphertext, withCheck)
}

// decrypt performs an RSA decryption of ciphertext into out. If check is true,
// m^e is calculated and compared with ciphertext, in order to defend against
// errors in the CRT computation.
func decrypt(priv *PrivateKey, ciphertext []byte, check bool) ([]byte, error) {
	var m *bigmod.Nat
	N, E := priv.pub.N, priv.pub.E

	c, err := bigmod.NewNat().SetBytes(ciphertext, N)
	if err != nil {
		return nil, ErrDecryption
	}

	if priv.dP == nil {
		// Legacy codepath for deprecated multi-prime keys.
		fips140.RecordNonApproved()
		m = bigmod.NewNat().Exp(c, priv.d.Bytes(N), N)

	} else {
		P, Q := priv.p, priv.q
		t0 := bigmod.NewNat()
		// m = c ^ Dp mod p
		m = bigmod.NewNat().Exp(t0.Mod(c, P), priv.dP, P)
		// m2 = c ^ Dq mod q
		m2 := bigmod.NewNat().Exp(t0.Mod(c, Q), priv.dQ, Q)
		// m = m - m2 mod p
		m.Sub(t0.Mod(m2, P), P)
		// m = m * Qinv mod p
		m.Mul(priv.qInv, P)
		// m = m * q mod N
		m.ExpandFor(N).Mul(t0.Mod(Q.Nat(), N), N)
		// m = m + m2 mod N
		m.Add(m2.ExpandFor(N), N)
	}

	if check {
		c1 := bigmod.NewNat().ExpShortVarTime(m, uint(E), N)
		if c1.Equal(c) != 1 {
			return nil, ErrDecryption
		}
	}

	return m.Bytes(N), nil
}

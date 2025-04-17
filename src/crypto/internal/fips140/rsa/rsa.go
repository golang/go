// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"bytes"
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
	dP   []byte      // d mod (p - 1)
	dQ   []byte      // d mod (q - 1)
	qInv *bigmod.Nat // qInv = q⁻¹ mod p
	// fipsApproved is false if this key does not comply with FIPS 186-5 or
	// SP 800-56B Rev. 2.
	fipsApproved bool
}

func (priv *PrivateKey) PublicKey() *PublicKey {
	return &priv.pub
}

// NewPrivateKey creates a new RSA private key from the given parameters.
//
// All values are in big-endian byte slice format, and may have leading zeros
// or be shorter if leading zeroes were trimmed.
func NewPrivateKey(N []byte, e int, d, P, Q []byte) (*PrivateKey, error) {
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
	return newPrivateKey(n, e, dN, p, q)
}

func newPrivateKey(n *bigmod.Modulus, e int, d *bigmod.Nat, p, q *bigmod.Modulus) (*PrivateKey, error) {
	pMinusOne := p.Nat().SubOne(p)
	pMinusOneMod, err := bigmod.NewModulus(pMinusOne.Bytes(p))
	if err != nil {
		return nil, err
	}
	dP := bigmod.NewNat().Mod(d, pMinusOneMod).Bytes(pMinusOneMod)

	qMinusOne := q.Nat().SubOne(q)
	qMinusOneMod, err := bigmod.NewModulus(qMinusOne.Bytes(q))
	if err != nil {
		return nil, err
	}
	dQ := bigmod.NewNat().Mod(d, qMinusOneMod).Bytes(qMinusOneMod)

	// Constant-time modular inversion with prime modulus by Fermat's Little
	// Theorem: qInv = q⁻¹ mod p = q^(p-2) mod p.
	if p.Nat().IsOdd() == 0 {
		// [bigmod.Nat.Exp] requires an odd modulus.
		return nil, errors.New("crypto/rsa: p is even")
	}
	pMinusTwo := p.Nat().SubOne(p).SubOne(p).Bytes(p)
	qInv := bigmod.NewNat().Mod(q.Nat(), p)
	qInv.Exp(qInv, pMinusTwo, p)

	pk := &PrivateKey{
		pub: PublicKey{
			N: n, E: e,
		},
		d: d, p: p, q: q,
		dP: dP, dQ: dQ, qInv: qInv,
	}
	if err := checkPrivateKey(pk); err != nil {
		return nil, err
	}
	return pk, nil
}

// NewPrivateKeyWithPrecomputation creates a new RSA private key from the given
// parameters, which include precomputed CRT values.
func NewPrivateKeyWithPrecomputation(N []byte, e int, d, P, Q, dP, dQ, qInv []byte) (*PrivateKey, error) {
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
	qInvNat, err := bigmod.NewNat().SetBytes(qInv, p)
	if err != nil {
		return nil, err
	}

	pk := &PrivateKey{
		pub: PublicKey{
			N: n, E: e,
		},
		d: dN, p: p, q: q,
		dP: dP, dQ: dQ, qInv: qInvNat,
	}
	if err := checkPrivateKey(pk); err != nil {
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
	if err := checkPrivateKey(pk); err != nil {
		return nil, err
	}
	return pk, nil
}

// Export returns the key parameters in big-endian byte slice format.
//
// P, Q, dP, dQ, and qInv may be nil if the key was created with
// NewPrivateKeyWithoutCRT.
func (priv *PrivateKey) Export() (N []byte, e int, d, P, Q, dP, dQ, qInv []byte) {
	N = priv.pub.N.Nat().Bytes(priv.pub.N)
	e = priv.pub.E
	d = priv.d.Bytes(priv.pub.N)
	if priv.dP == nil {
		return
	}
	P = priv.p.Nat().Bytes(priv.p)
	Q = priv.q.Nat().Bytes(priv.q)
	dP = bytes.Clone(priv.dP)
	dQ = bytes.Clone(priv.dQ)
	qInv = priv.qInv.Bytes(priv.p)
	return
}

// checkPrivateKey is called by the NewPrivateKey and GenerateKey functions, and
// is allowed to modify priv.fipsApproved.
func checkPrivateKey(priv *PrivateKey) error {
	priv.fipsApproved = true

	if fipsApproved, err := checkPublicKey(&priv.pub); err != nil {
		return err
	} else if !fipsApproved {
		priv.fipsApproved = false
	}

	if priv.dP == nil {
		// Legacy and deprecated multi-prime keys.
		priv.fipsApproved = false
		return nil
	}

	N := priv.pub.N
	p := priv.p
	q := priv.q

	// FIPS 186-5, Section 5.1 requires "that p and q be of the same bit length."
	if p.BitLen() != q.BitLen() {
		priv.fipsApproved = false
	}

	// Check that pq ≡ 1 mod N (and that p < N and q < N).
	pN := bigmod.NewNat().ExpandFor(N)
	if _, err := pN.SetBytes(p.Nat().Bytes(p), N); err != nil {
		return errors.New("crypto/rsa: invalid prime")
	}
	qN := bigmod.NewNat().ExpandFor(N)
	if _, err := qN.SetBytes(q.Nat().Bytes(q), N); err != nil {
		return errors.New("crypto/rsa: invalid prime")
	}
	if pN.Mul(qN, N).IsZero() != 1 {
		return errors.New("crypto/rsa: p * q != n")
	}

	// Check that de ≡ 1 mod p-1, and de ≡ 1 mod q-1.
	//
	// This implies that e is coprime to each p-1 as e has a multiplicative
	// inverse. Therefore e is coprime to lcm(p-1,q-1) = λ(N).
	// It also implies that a^de ≡ a mod p as a^(p-1) ≡ 1 mod p. Thus a^de ≡ a
	// mod n for all a coprime to n, as required.
	//
	// This checks dP, dQ, and e. We don't check d because it is not actually
	// used in the RSA private key operation.
	pMinus1, err := bigmod.NewModulus(p.Nat().SubOne(p).Bytes(p))
	if err != nil {
		return errors.New("crypto/rsa: invalid prime")
	}
	dP, err := bigmod.NewNat().SetBytes(priv.dP, pMinus1)
	if err != nil {
		return errors.New("crypto/rsa: invalid CRT exponent")
	}
	de := bigmod.NewNat()
	de.SetUint(uint(priv.pub.E)).ExpandFor(pMinus1)
	de.Mul(dP, pMinus1)
	if de.IsOne() != 1 {
		return errors.New("crypto/rsa: invalid CRT exponent")
	}

	qMinus1, err := bigmod.NewModulus(q.Nat().SubOne(q).Bytes(q))
	if err != nil {
		return errors.New("crypto/rsa: invalid prime")
	}
	dQ, err := bigmod.NewNat().SetBytes(priv.dQ, qMinus1)
	if err != nil {
		return errors.New("crypto/rsa: invalid CRT exponent")
	}
	de.SetUint(uint(priv.pub.E)).ExpandFor(qMinus1)
	de.Mul(dQ, qMinus1)
	if de.IsOne() != 1 {
		return errors.New("crypto/rsa: invalid CRT exponent")
	}

	// Check that qInv * q ≡ 1 mod p.
	qP, err := bigmod.NewNat().SetOverflowingBytes(q.Nat().Bytes(q), p)
	if err != nil {
		// q >= 2^⌈log2(p)⌉
		qP = bigmod.NewNat().Mod(q.Nat(), p)
	}
	if qP.Mul(priv.qInv, p).IsOne() != 1 {
		return errors.New("crypto/rsa: invalid CRT coefficient")
	}

	// Check that |p - q| > 2^(nlen/2 - 100).
	//
	// If p and q are very close to each other, then N=pq can be trivially
	// factored using Fermat's factorization method. Broken RSA implementations
	// do generate such keys. See Hanno Böck, Fermat Factorization in the Wild,
	// https://eprint.iacr.org/2023/026.pdf.
	diff := bigmod.NewNat()
	if qP, err := bigmod.NewNat().SetBytes(q.Nat().Bytes(q), p); err != nil {
		// q > p
		pQ, err := bigmod.NewNat().SetBytes(p.Nat().Bytes(p), q)
		if err != nil {
			return errors.New("crypto/rsa: p == q")
		}
		// diff = 0 - p mod q = q - p
		diff.ExpandFor(q).Sub(pQ, q)
	} else {
		// p > q
		// diff = 0 - q mod p = p - q
		diff.ExpandFor(p).Sub(qP, p)
	}
	// A tiny bit of leakage is acceptable because it's not adaptive, an
	// attacker only learns the magnitude of p - q.
	if diff.BitLenVarTime() <= N.BitLen()/2-100 {
		return errors.New("crypto/rsa: |p - q| too small")
	}

	// Check that d > 2^(nlen/2).
	//
	// See section 3 of https://crypto.stanford.edu/~dabo/papers/RSA-survey.pdf
	// for more details about attacks on small d values.
	//
	// Likewise, the leakage of the magnitude of d is not adaptive.
	if priv.d.BitLenVarTime() <= N.BitLen()/2 {
		return errors.New("crypto/rsa: d too small")
	}

	// If the key is still in scope for FIPS mode, perform a Pairwise
	// Consistency Test.
	if priv.fipsApproved {
		if err := fips140.PCT("RSA sign and verify PCT", func() error {
			hash := []byte{
				0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
				0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
				0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
				0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
			}
			sig, err := signPKCS1v15(priv, "SHA-256", hash)
			if err != nil {
				return err
			}
			return verifyPKCS1v15(priv.PublicKey(), "SHA-256", hash, sig)
		}); err != nil {
			return err
		}
	}

	return nil
}

func checkPublicKey(pub *PublicKey) (fipsApproved bool, err error) {
	fipsApproved = true
	if pub.N == nil {
		return false, errors.New("crypto/rsa: missing public modulus")
	}
	if pub.N.Nat().IsOdd() == 0 {
		return false, errors.New("crypto/rsa: public modulus is even")
	}
	// FIPS 186-5, Section 5.1: "This standard specifies the use of a modulus
	// whose bit length is an even integer and greater than or equal to 2048
	// bits."
	if pub.N.BitLen() < 2048 {
		fipsApproved = false
	}
	if pub.N.BitLen()%2 == 1 {
		fipsApproved = false
	}
	if pub.E < 2 {
		return false, errors.New("crypto/rsa: public exponent too small or negative")
	}
	// e needs to be coprime with p-1 and q-1, since it must be invertible
	// modulo λ(pq). Since p and q are prime, this means e needs to be odd.
	if pub.E&1 == 0 {
		return false, errors.New("crypto/rsa: public exponent is even")
	}
	// FIPS 186-5, Section 5.5(e): "The exponent e shall be an odd, positive
	// integer such that 2¹⁶ < e < 2²⁵⁶."
	if pub.E <= 1<<16 {
		fipsApproved = false
	}
	// We require pub.E to fit into a 32-bit integer so that we
	// do not have different behavior depending on whether
	// int is 32 or 64 bits. See also
	// https://www.imperialviolet.org/2012/03/16/rsae.html.
	if pub.E > 1<<31-1 {
		return false, errors.New("crypto/rsa: public exponent too large")
	}
	return fipsApproved, nil
}

// Encrypt performs the RSA public key operation.
func Encrypt(pub *PublicKey, plaintext []byte) ([]byte, error) {
	fips140.RecordNonApproved()
	if _, err := checkPublicKey(pub); err != nil {
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
	if !priv.fipsApproved {
		fips140.RecordNonApproved()
	}

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

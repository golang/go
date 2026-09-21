// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"bytes"
	"crypto/internal/constanttime"
	"crypto/internal/fips140"
	"crypto/internal/fips140/bigmod"
	"crypto/internal/fips140/drbg"
	"crypto/internal/fips140/subtle"
	"errors"
	"hash"
	"io"
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

// TestingOnlyLargeExponentPrivateKey is a variant of [PrivateKey] that supports
// large public exponents. It is only meant for supporting the full ACVP test
// suite. This type must not be used in production code.
type TestingOnlyLargeExponentPrivateKey struct {
	n    *bigmod.Modulus
	e    []byte // big-endian public exponent
	d    *bigmod.Nat
	p, q *bigmod.Modulus
	dP   []byte
	dQ   []byte
	qInv *bigmod.Nat
}

func (priv *TestingOnlyLargeExponentPrivateKey) Size() int {
	return (priv.n.BitLen() + 7) / 8
}

// TestingOnlyNewLargeExponentPrivateKeyWithPrecomputation creates a new RSA private key
// with a large public exponent from the given parameters. It is only meant for ACVP testing.
func TestingOnlyNewLargeExponentPrivateKeyWithPrecomputation(N []byte, e []byte, d, P, Q, dP, dQ, qInv []byte) (*TestingOnlyLargeExponentPrivateKey, error) {
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

	priv := &TestingOnlyLargeExponentPrivateKey{
		n: n, e: e, d: dN, p: p, q: q,
		dP: dP, dQ: dQ, qInv: qInvNat,
	}
	if err := checkLargeExponentPrivateKey(priv); err != nil {
		return nil, err
	}
	return priv, nil
}

func checkLargeExponentPrivateKey(priv *TestingOnlyLargeExponentPrivateKey) error {
	// Check public key portion.
	pub := &TestingOnlyLargeExponentPublicKey{N: priv.n, E: priv.e}
	if err := checkLargeExponentPublicKey(pub); err != nil {
		return err
	}

	N := priv.n
	p := priv.p
	q := priv.q

	// FIPS 186-5, Section 5.1 requires "that p and q be of the same bit length."
	if p.BitLen() != q.BitLen() {
		// We don't enforce this for testing, just note it.
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
	// Uses byte-slice exponent for large exponents.
	pMinus1, err := bigmod.NewModulus(p.Nat().SubOne(p).Bytes(p))
	if err != nil {
		return errors.New("crypto/rsa: invalid prime")
	}
	dP, err := bigmod.NewNat().SetBytes(priv.dP, pMinus1)
	if err != nil {
		return errors.New("crypto/rsa: invalid CRT exponent")
	}
	de := bigmod.NewNat()
	if _, err := de.SetBytes(priv.e, pMinus1); err != nil {
		// Exponent might be larger than p-1, reduce it.
		eNat, _ := bigmod.NewNat().SetBytes(priv.e, priv.n)
		de.Mod(eNat, pMinus1)
	}
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
	if _, err := de.SetBytes(priv.e, qMinus1); err != nil {
		// Exponent might be larger than q-1, reduce it.
		eNat, _ := bigmod.NewNat().SetBytes(priv.e, priv.n)
		de.Mod(eNat, qMinus1)
	}
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

	return nil
}

func decryptLargeExponent(priv *TestingOnlyLargeExponentPrivateKey, ciphertext []byte) ([]byte, error) {
	N := priv.n
	c, err := bigmod.NewNat().SetBytes(ciphertext, N)
	if err != nil {
		return nil, ErrDecryption
	}

	// CRT-based decryption (same as regular decrypt, doesn't use E).
	P, Q := priv.p, priv.q
	t0 := bigmod.NewNat()
	// m = c ^ Dp mod p
	m := bigmod.NewNat().Exp(t0.Mod(c, P), priv.dP, P)
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

	return m.Bytes(N), nil
}

// TestingOnlyLargeExponentDecryptOAEP decrypts ciphertext using RSAES-OAEP with
// a private key that has a large public exponent. It is only meant for ACVP testing.
func TestingOnlyLargeExponentDecryptOAEP(hash, mgfHash hash.Hash, priv *TestingOnlyLargeExponentPrivateKey, ciphertext []byte, label []byte) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	checkApprovedHash(hash)

	k := priv.Size()
	if len(ciphertext) > k || k < hash.Size()*2+2 {
		return nil, ErrDecryption
	}

	em, err := decryptLargeExponent(priv, ciphertext)
	if err != nil {
		return nil, err
	}

	hash.Reset()
	hash.Write(label)
	lHash := hash.Sum(nil)

	firstByteIsZero := constanttime.ByteEq(em[0], 0)

	seed := em[1 : hash.Size()+1]
	db := em[hash.Size()+1:]

	mgf1XOR(seed, mgfHash, db)
	mgf1XOR(db, mgfHash, seed)

	lHash2 := db[0:hash.Size()]

	lHash2Good := subtle.ConstantTimeCompare(lHash, lHash2)

	var lookingForIndex, index, invalid int
	lookingForIndex = 1
	rest := db[hash.Size():]

	for i := 0; i < len(rest); i++ {
		equals0 := constanttime.ByteEq(rest[i], 0)
		equals1 := constanttime.ByteEq(rest[i], 1)
		index = constanttime.Select(lookingForIndex&equals1, i, index)
		lookingForIndex = constanttime.Select(equals1, 0, lookingForIndex)
		invalid = constanttime.Select(lookingForIndex&^equals0, 1, invalid)
	}

	if firstByteIsZero&lHash2Good&^invalid&^lookingForIndex != 1 {
		return nil, ErrDecryption
	}

	return rest[index+1:], nil
}

// TestingOnlyLargeExponentEncryptOAEP encrypts the given message with RSAES-OAEP
// using a public key with a large exponent. It is only meant for ACVP testing.
func TestingOnlyLargeExponentEncryptOAEP(hash, mgfHash hash.Hash, random io.Reader, pub *TestingOnlyLargeExponentPublicKey, msg []byte, label []byte) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	checkApprovedHash(hash)
	if err := checkLargeExponentPublicKey(pub); err != nil {
		return nil, err
	}
	k := pub.Size()
	if len(msg) > k-2*hash.Size()-2 {
		return nil, ErrMessageTooLong
	}

	hash.Reset()
	hash.Write(label)
	lHash := hash.Sum(nil)

	em := make([]byte, k)
	seed := em[1 : 1+hash.Size()]
	db := em[1+hash.Size():]

	copy(db[0:hash.Size()], lHash)
	db[len(db)-len(msg)-1] = 1
	copy(db[len(db)-len(msg):], msg)

	if err := drbg.ReadWithReader(random, seed); err != nil {
		return nil, err
	}

	mgf1XOR(db, mgfHash, seed)
	mgf1XOR(seed, mgfHash, db)

	return encryptLargeExponent(pub, em)
}

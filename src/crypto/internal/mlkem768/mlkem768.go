// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mlkem768 implements the quantum-resistant key encapsulation method
// ML-KEM (formerly known as Kyber), as specified in [NIST FIPS 203].
//
// Only the recommended ML-KEM-768 parameter set is provided.
//
// [NIST FIPS 203]: https://doi.org/10.6028/NIST.FIPS.203
package mlkem768

// This package targets security, correctness, simplicity, readability, and
// reviewability as its primary goals. All critical operations are performed in
// constant time.
//
// Variable and function names, as well as code layout, are selected to
// facilitate reviewing the implementation against the NIST FIPS 203 document.
//
// Reviewers unfamiliar with polynomials or linear algebra might find the
// background at https://words.filippo.io/kyber-math/ useful.

import (
	"crypto/rand"
	"crypto/subtle"
	"errors"

	"golang.org/x/crypto/sha3"
)

const (
	// ML-KEM global constants.
	n = 256
	q = 3329

	// encodingSizeX is the byte size of a ringElement or nttElement encoded
	// by ByteEncode_X (FIPS 203, Algorithm 5).
	encodingSize12 = n * 12 / 8
	encodingSize10 = n * 10 / 8
	encodingSize4  = n * 4 / 8
	encodingSize1  = n * 1 / 8

	messageSize = encodingSize1

	SharedKeySize = 32
	SeedSize      = 32 + 32
)

// ML-KEM-768 parameters.
const (
	k = 3

	decryptionKeySize = k * encodingSize12
	encryptionKeySize = k*encodingSize12 + 32

	CiphertextSize768       = k*encodingSize10 + encodingSize4
	EncapsulationKeySize768 = encryptionKeySize
)

// A DecapsulationKey768 is the secret key used to decapsulate a shared key from a
// ciphertext. It includes various precomputed values.
type DecapsulationKey768 struct {
	d [32]byte // decapsulation key seed
	z [32]byte // implicit rejection sampling seed

	ρ [32]byte // sampleNTT seed for A, stored for the encapsulation key
	h [32]byte // H(ek), stored for ML-KEM.Decaps_internal

	encryptionKey
	decryptionKey
}

// Bytes returns the decapsulation key as a 64-byte seed in the "d || z" form.
//
// The decapsulation key must be kept secret.
func (dk *DecapsulationKey768) Bytes() []byte {
	var b [SeedSize]byte
	copy(b[:], dk.d[:])
	copy(b[32:], dk.z[:])
	return b[:]
}

// EncapsulationKey returns the public encapsulation key necessary to produce
// ciphertexts.
func (dk *DecapsulationKey768) EncapsulationKey() *EncapsulationKey768 {
	return &EncapsulationKey768{
		ρ:             dk.ρ,
		h:             dk.h,
		encryptionKey: dk.encryptionKey,
	}
}

// An EncapsulationKey768 is the public key used to produce ciphertexts to be
// decapsulated by the corresponding [DecapsulationKey768].
type EncapsulationKey768 struct {
	ρ [32]byte // sampleNTT seed for A
	h [32]byte // H(ek)
	encryptionKey
}

// Bytes returns the encapsulation key as a byte slice.
func (ek *EncapsulationKey768) Bytes() []byte {
	// The actual logic is in a separate function to outline this allocation.
	b := make([]byte, 0, EncapsulationKeySize768)
	return ek.bytes(b)
}

func (ek *EncapsulationKey768) bytes(b []byte) []byte {
	for i := range ek.t {
		b = polyByteEncode(b, ek.t[i])
	}
	b = append(b, ek.ρ[:]...)
	return b
}

// encryptionKey is the parsed and expanded form of a PKE encryption key.
type encryptionKey struct {
	t [k]nttElement     // ByteDecode₁₂(ek[:384k])
	a [k * k]nttElement // A[i*k+j] = sampleNTT(ρ, j, i)
}

// decryptionKey is the parsed and expanded form of a PKE decryption key.
type decryptionKey struct {
	s [k]nttElement // ByteDecode₁₂(dk[:decryptionKeySize])
}

// GenerateKey768 generates a new decapsulation key, drawing random bytes from
// crypto/rand. The decapsulation key must be kept secret.
func GenerateKey768() (*DecapsulationKey768, error) {
	// The actual logic is in a separate function to outline this allocation.
	dk := &DecapsulationKey768{}
	return generateKey(dk), nil
}

func generateKey(dk *DecapsulationKey768) *DecapsulationKey768 {
	var d [32]byte
	rand.Read(d[:])
	var z [32]byte
	rand.Read(z[:])
	return kemKeyGen(dk, &d, &z)
}

// NewDecapsulationKey768 parses a decapsulation key from a 64-byte
// seed in the "d || z" form. The seed must be uniformly random.
func NewDecapsulationKey768(seed []byte) (*DecapsulationKey768, error) {
	// The actual logic is in a separate function to outline this allocation.
	dk := &DecapsulationKey768{}
	return newKeyFromSeed(dk, seed)
}

func newKeyFromSeed(dk *DecapsulationKey768, seed []byte) (*DecapsulationKey768, error) {
	if len(seed) != SeedSize {
		return nil, errors.New("mlkem768: invalid seed length")
	}
	d := (*[32]byte)(seed[:32])
	z := (*[32]byte)(seed[32:])
	return kemKeyGen(dk, d, z), nil
}

// kemKeyGen generates a decapsulation key.
//
// It implements ML-KEM.KeyGen_internal according to FIPS 203, Algorithm 16, and
// K-PKE.KeyGen according to FIPS 203, Algorithm 13. The two are merged to save
// copies and allocations.
func kemKeyGen(dk *DecapsulationKey768, d, z *[32]byte) *DecapsulationKey768 {
	if dk == nil {
		dk = &DecapsulationKey768{}
	}
	dk.d = *d
	dk.z = *z

	g := sha3.New512()
	g.Write(d[:])
	g.Write([]byte{k}) // Module dimension as a domain separator.
	G := g.Sum(make([]byte, 0, 64))
	ρ, σ := G[:32], G[32:]
	dk.ρ = [32]byte(ρ)

	A := &dk.a
	for i := byte(0); i < k; i++ {
		for j := byte(0); j < k; j++ {
			A[i*k+j] = sampleNTT(ρ, j, i)
		}
	}

	var N byte
	s := &dk.s
	for i := range s {
		s[i] = ntt(samplePolyCBD(σ, N))
		N++
	}
	e := make([]nttElement, k)
	for i := range e {
		e[i] = ntt(samplePolyCBD(σ, N))
		N++
	}

	t := &dk.t
	for i := range t { // t = A ◦ s + e
		t[i] = e[i]
		for j := range s {
			t[i] = polyAdd(t[i], nttMul(A[i*k+j], s[j]))
		}
	}

	H := sha3.New256()
	ek := dk.EncapsulationKey().Bytes()
	H.Write(ek)
	H.Sum(dk.h[:0])

	return dk
}

// Encapsulate generates a shared key and an associated ciphertext from an
// encapsulation key, drawing random bytes from crypto/rand.
//
// The shared key must be kept secret.
func (ek *EncapsulationKey768) Encapsulate() (ciphertext, sharedKey []byte) {
	// The actual logic is in a separate function to outline this allocation.
	var cc [CiphertextSize768]byte
	return ek.encapsulate(&cc)
}

func (ek *EncapsulationKey768) encapsulate(cc *[CiphertextSize768]byte) (ciphertext, sharedKey []byte) {
	var m [messageSize]byte
	rand.Read(m[:])
	// Note that the modulus check (step 2 of the encapsulation key check from
	// FIPS 203, Section 7.2) is performed by polyByteDecode in parseEK.
	return kemEncaps(cc, ek, &m)
}

// kemEncaps generates a shared key and an associated ciphertext.
//
// It implements ML-KEM.Encaps_internal according to FIPS 203, Algorithm 17.
func kemEncaps(cc *[CiphertextSize768]byte, ek *EncapsulationKey768, m *[messageSize]byte) (c, K []byte) {
	if cc == nil {
		cc = &[CiphertextSize768]byte{}
	}

	g := sha3.New512()
	g.Write(m[:])
	g.Write(ek.h[:])
	G := g.Sum(nil)
	K, r := G[:SharedKeySize], G[SharedKeySize:]
	c = pkeEncrypt(cc, &ek.encryptionKey, m, r)
	return c, K
}

// NewEncapsulationKey768 parses an encapsulation key from its encoded form.
// If the encapsulation key is not valid, NewEncapsulationKey768 returns an error.
func NewEncapsulationKey768(encapsulationKey []byte) (*EncapsulationKey768, error) {
	// The actual logic is in a separate function to outline this allocation.
	ek := &EncapsulationKey768{}
	return parseEK(ek, encapsulationKey)
}

// parseEK parses an encryption key from its encoded form.
//
// It implements the initial stages of K-PKE.Encrypt according to FIPS 203,
// Algorithm 14.
func parseEK(ek *EncapsulationKey768, ekPKE []byte) (*EncapsulationKey768, error) {
	if len(ekPKE) != encryptionKeySize {
		return nil, errors.New("mlkem768: invalid encapsulation key length")
	}

	ek.h = sha3.Sum256(ekPKE[:])

	for i := range ek.t {
		var err error
		ek.t[i], err = polyByteDecode[nttElement](ekPKE[:encodingSize12])
		if err != nil {
			return nil, err
		}
		ekPKE = ekPKE[encodingSize12:]
	}
	copy(ek.ρ[:], ekPKE)

	for i := byte(0); i < k; i++ {
		for j := byte(0); j < k; j++ {
			ek.a[i*k+j] = sampleNTT(ek.ρ[:], j, i)
		}
	}

	return ek, nil
}

// pkeEncrypt encrypt a plaintext message.
//
// It implements K-PKE.Encrypt according to FIPS 203, Algorithm 14, although the
// computation of t and AT is done in parseEK.
func pkeEncrypt(cc *[CiphertextSize768]byte, ex *encryptionKey, m *[messageSize]byte, rnd []byte) []byte {
	var N byte
	r, e1 := make([]nttElement, k), make([]ringElement, k)
	for i := range r {
		r[i] = ntt(samplePolyCBD(rnd, N))
		N++
	}
	for i := range e1 {
		e1[i] = samplePolyCBD(rnd, N)
		N++
	}
	e2 := samplePolyCBD(rnd, N)

	u := make([]ringElement, k) // NTT⁻¹(AT ◦ r) + e1
	for i := range u {
		u[i] = e1[i]
		for j := range r {
			// Note that i and j are inverted, as we need the transposed of A.
			u[i] = polyAdd(u[i], inverseNTT(nttMul(ex.a[j*k+i], r[j])))
		}
	}

	μ := ringDecodeAndDecompress1(m)

	var vNTT nttElement // t⊺ ◦ r
	for i := range ex.t {
		vNTT = polyAdd(vNTT, nttMul(ex.t[i], r[i]))
	}
	v := polyAdd(polyAdd(inverseNTT(vNTT), e2), μ)

	c := cc[:0]
	for _, f := range u {
		c = ringCompressAndEncode10(c, f)
	}
	c = ringCompressAndEncode4(c, v)

	return c
}

// Decapsulate generates a shared key from a ciphertext and a decapsulation key.
// If the ciphertext is not valid, Decapsulate returns an error.
//
// The shared key must be kept secret.
func (dk *DecapsulationKey768) Decapsulate(ciphertext []byte) (sharedKey []byte, err error) {
	if len(ciphertext) != CiphertextSize768 {
		return nil, errors.New("mlkem768: invalid ciphertext length")
	}
	c := (*[CiphertextSize768]byte)(ciphertext)
	// Note that the hash check (step 3 of the decapsulation input check from
	// FIPS 203, Section 7.3) is foregone as a DecapsulationKey is always
	// validly generated by ML-KEM.KeyGen_internal.
	return kemDecaps(dk, c), nil
}

// kemDecaps produces a shared key from a ciphertext.
//
// It implements ML-KEM.Decaps_internal according to FIPS 203, Algorithm 18.
func kemDecaps(dk *DecapsulationKey768, c *[CiphertextSize768]byte) (K []byte) {
	m := pkeDecrypt(&dk.decryptionKey, c)
	g := sha3.New512()
	g.Write(m[:])
	g.Write(dk.h[:])
	G := g.Sum(make([]byte, 0, 64))
	Kprime, r := G[:SharedKeySize], G[SharedKeySize:]
	J := sha3.NewShake256()
	J.Write(dk.z[:])
	J.Write(c[:])
	Kout := make([]byte, SharedKeySize)
	J.Read(Kout)
	var cc [CiphertextSize768]byte
	c1 := pkeEncrypt(&cc, &dk.encryptionKey, (*[32]byte)(m), r)

	subtle.ConstantTimeCopy(subtle.ConstantTimeCompare(c[:], c1), Kout, Kprime)
	return Kout
}

// pkeDecrypt decrypts a ciphertext.
//
// It implements K-PKE.Decrypt according to FIPS 203, Algorithm 15,
// although s is retained from kemKeyGen.
func pkeDecrypt(dx *decryptionKey, c *[CiphertextSize768]byte) []byte {
	u := make([]ringElement, k)
	for i := range u {
		b := (*[encodingSize10]byte)(c[encodingSize10*i : encodingSize10*(i+1)])
		u[i] = ringDecodeAndDecompress10(b)
	}

	b := (*[encodingSize4]byte)(c[encodingSize10*k:])
	v := ringDecodeAndDecompress4(b)

	var mask nttElement // s⊺ ◦ NTT(u)
	for i := range dx.s {
		mask = polyAdd(mask, nttMul(dx.s[i], ntt(u[i])))
	}
	w := polySub(v, inverseNTT(mask))

	return ringCompressAndEncode1(nil, w)
}

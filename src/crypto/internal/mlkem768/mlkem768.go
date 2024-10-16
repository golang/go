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
	"internal/byteorder"

	"golang.org/x/crypto/sha3"
)

const (
	// ML-KEM global constants.
	n = 256
	q = 3329

	log2q = 12

	// ML-KEM-768 parameters. The code makes assumptions based on these values,
	// they can't be changed blindly.
	k  = 3
	η  = 2
	du = 10
	dv = 4

	// encodingSizeX is the byte size of a ringElement or nttElement encoded
	// by ByteEncode_X (FIPS 203, Algorithm 5).
	encodingSize12 = n * log2q / 8
	encodingSize10 = n * du / 8
	encodingSize4  = n * dv / 8
	encodingSize1  = n * 1 / 8

	messageSize       = encodingSize1
	decryptionKeySize = k * encodingSize12
	encryptionKeySize = k*encodingSize12 + 32

	CiphertextSize       = k*encodingSize10 + encodingSize4
	EncapsulationKeySize = encryptionKeySize
	SharedKeySize        = 32
	SeedSize             = 32 + 32
)

// A DecapsulationKey is the secret key used to decapsulate a shared key from a
// ciphertext. It includes various precomputed values.
type DecapsulationKey struct {
	d [32]byte // decapsulation key seed
	z [32]byte // implicit rejection sampling seed

	ρ [32]byte // sampleNTT seed for A, stored for the encapsulation key
	h [32]byte // H(ek), stored for ML-KEM.Decaps_internal

	encryptionKey
	decryptionKey
}

// Bytes returns the decapsulation key as a 64-byte seed in the "d || z" form.
func (dk *DecapsulationKey) Bytes() []byte {
	var b [SeedSize]byte
	copy(b[:], dk.d[:])
	copy(b[32:], dk.z[:])
	return b[:]
}

// EncapsulationKey returns the public encapsulation key necessary to produce
// ciphertexts.
func (dk *DecapsulationKey) EncapsulationKey() []byte {
	// The actual logic is in a separate function to outline this allocation.
	b := make([]byte, 0, EncapsulationKeySize)
	return dk.encapsulationKey(b)
}

func (dk *DecapsulationKey) encapsulationKey(b []byte) []byte {
	for i := range dk.t {
		b = polyByteEncode(b, dk.t[i])
	}
	b = append(b, dk.ρ[:]...)
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

// GenerateKey generates a new decapsulation key, drawing random bytes from
// crypto/rand. The decapsulation key must be kept secret.
func GenerateKey() (*DecapsulationKey, error) {
	// The actual logic is in a separate function to outline this allocation.
	dk := &DecapsulationKey{}
	return generateKey(dk)
}

func generateKey(dk *DecapsulationKey) (*DecapsulationKey, error) {
	var d [32]byte
	if _, err := rand.Read(d[:]); err != nil {
		return nil, errors.New("mlkem768: crypto/rand Read failed: " + err.Error())
	}
	var z [32]byte
	if _, err := rand.Read(z[:]); err != nil {
		return nil, errors.New("mlkem768: crypto/rand Read failed: " + err.Error())
	}
	return kemKeyGen(dk, &d, &z), nil
}

// NewKeyFromSeed deterministically generates a decapsulation key from a 64-byte
// seed in the "d || z" form. The seed must be uniformly random.
func NewKeyFromSeed(seed []byte) (*DecapsulationKey, error) {
	// The actual logic is in a separate function to outline this allocation.
	dk := &DecapsulationKey{}
	return newKeyFromSeed(dk, seed)
}

func newKeyFromSeed(dk *DecapsulationKey, seed []byte) (*DecapsulationKey, error) {
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
func kemKeyGen(dk *DecapsulationKey, d, z *[32]byte) *DecapsulationKey {
	if dk == nil {
		dk = &DecapsulationKey{}
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
	ek := dk.EncapsulationKey()
	H.Write(ek)
	H.Sum(dk.h[:0])

	return dk
}

// Encapsulate generates a shared key and an associated ciphertext from an
// encapsulation key, drawing random bytes from crypto/rand.
// If the encapsulation key is not valid, Encapsulate returns an error.
//
// The shared key must be kept secret.
func Encapsulate(encapsulationKey []byte) (ciphertext, sharedKey []byte, err error) {
	// The actual logic is in a separate function to outline this allocation.
	var cc [CiphertextSize]byte
	return encapsulate(&cc, encapsulationKey)
}

func encapsulate(cc *[CiphertextSize]byte, encapsulationKey []byte) (ciphertext, sharedKey []byte, err error) {
	if len(encapsulationKey) != EncapsulationKeySize {
		return nil, nil, errors.New("mlkem768: invalid encapsulation key length")
	}
	var m [messageSize]byte
	if _, err := rand.Read(m[:]); err != nil {
		return nil, nil, errors.New("mlkem768: crypto/rand Read failed: " + err.Error())
	}
	// Note that the modulus check (step 2 of the encapsulation key check from
	// FIPS 203, Section 7.2) is performed by polyByteDecode in parseEK.
	return kemEncaps(cc, encapsulationKey, &m)
}

// kemEncaps generates a shared key and an associated ciphertext.
//
// It implements ML-KEM.Encaps_internal according to FIPS 203, Algorithm 17.
func kemEncaps(cc *[CiphertextSize]byte, ek []byte, m *[messageSize]byte) (c, K []byte, err error) {
	if cc == nil {
		cc = &[CiphertextSize]byte{}
	}

	H := sha3.Sum256(ek[:])
	g := sha3.New512()
	g.Write(m[:])
	g.Write(H[:])
	G := g.Sum(nil)
	K, r := G[:SharedKeySize], G[SharedKeySize:]
	var ex encryptionKey
	if err := parseEK(&ex, ek[:]); err != nil {
		return nil, nil, err
	}
	c = pkeEncrypt(cc, &ex, m, r)
	return c, K, nil
}

// parseEK parses an encryption key from its encoded form.
//
// It implements the initial stages of K-PKE.Encrypt according to FIPS 203,
// Algorithm 14.
func parseEK(ex *encryptionKey, ekPKE []byte) error {
	if len(ekPKE) != encryptionKeySize {
		return errors.New("mlkem768: invalid encryption key length")
	}

	for i := range ex.t {
		var err error
		ex.t[i], err = polyByteDecode[nttElement](ekPKE[:encodingSize12])
		if err != nil {
			return err
		}
		ekPKE = ekPKE[encodingSize12:]
	}
	ρ := ekPKE

	for i := byte(0); i < k; i++ {
		for j := byte(0); j < k; j++ {
			ex.a[i*k+j] = sampleNTT(ρ, j, i)
		}
	}

	return nil
}

// pkeEncrypt encrypt a plaintext message.
//
// It implements K-PKE.Encrypt according to FIPS 203, Algorithm 14, although the
// computation of t and AT is done in parseEK.
func pkeEncrypt(cc *[CiphertextSize]byte, ex *encryptionKey, m *[messageSize]byte, rnd []byte) []byte {
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
func (dk *DecapsulationKey) Decapsulate(ciphertext []byte) (sharedKey []byte, err error) {
	if len(ciphertext) != CiphertextSize {
		return nil, errors.New("mlkem768: invalid ciphertext length")
	}
	c := (*[CiphertextSize]byte)(ciphertext)
	// Note that the hash check (step 3 of the decapsulation input check from
	// FIPS 203, Section 7.3) is foregone as a DecapsulationKey is always
	// validly generated by ML-KEM.KeyGen_internal.
	return kemDecaps(dk, c), nil
}

// kemDecaps produces a shared key from a ciphertext.
//
// It implements ML-KEM.Decaps_internal according to FIPS 203, Algorithm 18.
func kemDecaps(dk *DecapsulationKey, c *[CiphertextSize]byte) (K []byte) {
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
	var cc [CiphertextSize]byte
	c1 := pkeEncrypt(&cc, &dk.encryptionKey, (*[32]byte)(m), r)

	subtle.ConstantTimeCopy(subtle.ConstantTimeCompare(c[:], c1), Kout, Kprime)
	return Kout
}

// pkeDecrypt decrypts a ciphertext.
//
// It implements K-PKE.Decrypt according to FIPS 203, Algorithm 15,
// although s is retained from kemKeyGen.
func pkeDecrypt(dx *decryptionKey, c *[CiphertextSize]byte) []byte {
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

// fieldElement is an integer modulo q, an element of ℤ_q. It is always reduced.
type fieldElement uint16

// fieldCheckReduced checks that a value a is < q.
func fieldCheckReduced(a uint16) (fieldElement, error) {
	if a >= q {
		return 0, errors.New("unreduced field element")
	}
	return fieldElement(a), nil
}

// fieldReduceOnce reduces a value a < 2q.
func fieldReduceOnce(a uint16) fieldElement {
	x := a - q
	// If x underflowed, then x >= 2¹⁶ - q > 2¹⁵, so the top bit is set.
	x += (x >> 15) * q
	return fieldElement(x)
}

func fieldAdd(a, b fieldElement) fieldElement {
	x := uint16(a + b)
	return fieldReduceOnce(x)
}

func fieldSub(a, b fieldElement) fieldElement {
	x := uint16(a - b + q)
	return fieldReduceOnce(x)
}

const (
	barrettMultiplier = 5039 // 2¹² * 2¹² / q
	barrettShift      = 24   // log₂(2¹² * 2¹²)
)

// fieldReduce reduces a value a < 2q² using Barrett reduction, to avoid
// potentially variable-time division.
func fieldReduce(a uint32) fieldElement {
	quotient := uint32((uint64(a) * barrettMultiplier) >> barrettShift)
	return fieldReduceOnce(uint16(a - quotient*q))
}

func fieldMul(a, b fieldElement) fieldElement {
	x := uint32(a) * uint32(b)
	return fieldReduce(x)
}

// fieldMulSub returns a * (b - c). This operation is fused to save a
// fieldReduceOnce after the subtraction.
func fieldMulSub(a, b, c fieldElement) fieldElement {
	x := uint32(a) * uint32(b-c+q)
	return fieldReduce(x)
}

// fieldAddMul returns a * b + c * d. This operation is fused to save a
// fieldReduceOnce and a fieldReduce.
func fieldAddMul(a, b, c, d fieldElement) fieldElement {
	x := uint32(a) * uint32(b)
	x += uint32(c) * uint32(d)
	return fieldReduce(x)
}

// compress maps a field element uniformly to the range 0 to 2ᵈ-1, according to
// FIPS 203, Definition 4.7.
func compress(x fieldElement, d uint8) uint16 {
	// We want to compute (x * 2ᵈ) / q, rounded to nearest integer, with 1/2
	// rounding up (see FIPS 203, Section 2.3).

	// Barrett reduction produces a quotient and a remainder in the range [0, 2q),
	// such that dividend = quotient * q + remainder.
	dividend := uint32(x) << d // x * 2ᵈ
	quotient := uint32(uint64(dividend) * barrettMultiplier >> barrettShift)
	remainder := dividend - quotient*q

	// Since the remainder is in the range [0, 2q), not [0, q), we need to
	// portion it into three spans for rounding.
	//
	//     [ 0,       q/2     ) -> round to 0
	//     [ q/2,     q + q/2 ) -> round to 1
	//     [ q + q/2, 2q      ) -> round to 2
	//
	// We can convert that to the following logic: add 1 if remainder > q/2,
	// then add 1 again if remainder > q + q/2.
	//
	// Note that if remainder > x, then ⌊x⌋ - remainder underflows, and the top
	// bit of the difference will be set.
	quotient += (q/2 - remainder) >> 31 & 1
	quotient += (q + q/2 - remainder) >> 31 & 1

	// quotient might have overflowed at this point, so reduce it by masking.
	var mask uint32 = (1 << d) - 1
	return uint16(quotient & mask)
}

// decompress maps a number x between 0 and 2ᵈ-1 uniformly to the full range of
// field elements, according to FIPS 203, Definition 4.8.
func decompress(y uint16, d uint8) fieldElement {
	// We want to compute (y * q) / 2ᵈ, rounded to nearest integer, with 1/2
	// rounding up (see FIPS 203, Section 2.3).

	dividend := uint32(y) * q
	quotient := dividend >> d // (y * q) / 2ᵈ

	// The d'th least-significant bit of the dividend (the most significant bit
	// of the remainder) is 1 for the top half of the values that divide to the
	// same quotient, which are the ones that round up.
	quotient += dividend >> (d - 1) & 1

	// quotient is at most (2¹¹-1) * q / 2¹¹ + 1 = 3328, so it didn't overflow.
	return fieldElement(quotient)
}

// ringElement is a polynomial, an element of R_q, represented as an array
// according to FIPS 203, Section 2.4.4.
type ringElement [n]fieldElement

// polyAdd adds two ringElements or nttElements.
func polyAdd[T ~[n]fieldElement](a, b T) (s T) {
	for i := range s {
		s[i] = fieldAdd(a[i], b[i])
	}
	return s
}

// polySub subtracts two ringElements or nttElements.
func polySub[T ~[n]fieldElement](a, b T) (s T) {
	for i := range s {
		s[i] = fieldSub(a[i], b[i])
	}
	return s
}

// polyByteEncode appends the 384-byte encoding of f to b.
//
// It implements ByteEncode₁₂, according to FIPS 203, Algorithm 5.
func polyByteEncode[T ~[n]fieldElement](b []byte, f T) []byte {
	out, B := sliceForAppend(b, encodingSize12)
	for i := 0; i < n; i += 2 {
		x := uint32(f[i]) | uint32(f[i+1])<<12
		B[0] = uint8(x)
		B[1] = uint8(x >> 8)
		B[2] = uint8(x >> 16)
		B = B[3:]
	}
	return out
}

// polyByteDecode decodes the 384-byte encoding of a polynomial, checking that
// all the coefficients are properly reduced. This fulfills the "Modulus check"
// step of ML-KEM Encapsulation.
//
// It implements ByteDecode₁₂, according to FIPS 203, Algorithm 6.
func polyByteDecode[T ~[n]fieldElement](b []byte) (T, error) {
	if len(b) != encodingSize12 {
		return T{}, errors.New("mlkem768: invalid encoding length")
	}
	var f T
	for i := 0; i < n; i += 2 {
		d := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
		const mask12 = 0b1111_1111_1111
		var err error
		if f[i], err = fieldCheckReduced(uint16(d & mask12)); err != nil {
			return T{}, errors.New("mlkem768: invalid polynomial encoding")
		}
		if f[i+1], err = fieldCheckReduced(uint16(d >> 12)); err != nil {
			return T{}, errors.New("mlkem768: invalid polynomial encoding")
		}
		b = b[3:]
	}
	return f, nil
}

// sliceForAppend takes a slice and a requested number of bytes. It returns a
// slice with the contents of the given slice followed by that many bytes and a
// second slice that aliases into it and contains only the extra bytes. If the
// original slice has sufficient capacity then no allocation is performed.
func sliceForAppend(in []byte, n int) (head, tail []byte) {
	if total := len(in) + n; cap(in) >= total {
		head = in[:total]
	} else {
		head = make([]byte, total)
		copy(head, in)
	}
	tail = head[len(in):]
	return
}

// ringCompressAndEncode1 appends a 32-byte encoding of a ring element to s,
// compressing one coefficients per bit.
//
// It implements Compress₁, according to FIPS 203, Definition 4.7,
// followed by ByteEncode₁, according to FIPS 203, Algorithm 5.
func ringCompressAndEncode1(s []byte, f ringElement) []byte {
	s, b := sliceForAppend(s, encodingSize1)
	for i := range b {
		b[i] = 0
	}
	for i := range f {
		b[i/8] |= uint8(compress(f[i], 1) << (i % 8))
	}
	return s
}

// ringDecodeAndDecompress1 decodes a 32-byte slice to a ring element where each
// bit is mapped to 0 or ⌈q/2⌋.
//
// It implements ByteDecode₁, according to FIPS 203, Algorithm 6,
// followed by Decompress₁, according to FIPS 203, Definition 4.8.
func ringDecodeAndDecompress1(b *[encodingSize1]byte) ringElement {
	var f ringElement
	for i := range f {
		b_i := b[i/8] >> (i % 8) & 1
		const halfQ = (q + 1) / 2        // ⌈q/2⌋, rounded up per FIPS 203, Section 2.3
		f[i] = fieldElement(b_i) * halfQ // 0 decompresses to 0, and 1 to ⌈q/2⌋
	}
	return f
}

// ringCompressAndEncode4 appends a 128-byte encoding of a ring element to s,
// compressing two coefficients per byte.
//
// It implements Compress₄, according to FIPS 203, Definition 4.7,
// followed by ByteEncode₄, according to FIPS 203, Algorithm 5.
func ringCompressAndEncode4(s []byte, f ringElement) []byte {
	s, b := sliceForAppend(s, encodingSize4)
	for i := 0; i < n; i += 2 {
		b[i/2] = uint8(compress(f[i], 4) | compress(f[i+1], 4)<<4)
	}
	return s
}

// ringDecodeAndDecompress4 decodes a 128-byte encoding of a ring element where
// each four bits are mapped to an equidistant distribution.
//
// It implements ByteDecode₄, according to FIPS 203, Algorithm 6,
// followed by Decompress₄, according to FIPS 203, Definition 4.8.
func ringDecodeAndDecompress4(b *[encodingSize4]byte) ringElement {
	var f ringElement
	for i := 0; i < n; i += 2 {
		f[i] = fieldElement(decompress(uint16(b[i/2]&0b1111), 4))
		f[i+1] = fieldElement(decompress(uint16(b[i/2]>>4), 4))
	}
	return f
}

// ringCompressAndEncode10 appends a 320-byte encoding of a ring element to s,
// compressing four coefficients per five bytes.
//
// It implements Compress₁₀, according to FIPS 203, Definition 4.7,
// followed by ByteEncode₁₀, according to FIPS 203, Algorithm 5.
func ringCompressAndEncode10(s []byte, f ringElement) []byte {
	s, b := sliceForAppend(s, encodingSize10)
	for i := 0; i < n; i += 4 {
		var x uint64
		x |= uint64(compress(f[i+0], 10))
		x |= uint64(compress(f[i+1], 10)) << 10
		x |= uint64(compress(f[i+2], 10)) << 20
		x |= uint64(compress(f[i+3], 10)) << 30
		b[0] = uint8(x)
		b[1] = uint8(x >> 8)
		b[2] = uint8(x >> 16)
		b[3] = uint8(x >> 24)
		b[4] = uint8(x >> 32)
		b = b[5:]
	}
	return s
}

// ringDecodeAndDecompress10 decodes a 320-byte encoding of a ring element where
// each ten bits are mapped to an equidistant distribution.
//
// It implements ByteDecode₁₀, according to FIPS 203, Algorithm 6,
// followed by Decompress₁₀, according to FIPS 203, Definition 4.8.
func ringDecodeAndDecompress10(bb *[encodingSize10]byte) ringElement {
	b := bb[:]
	var f ringElement
	for i := 0; i < n; i += 4 {
		x := uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32
		b = b[5:]
		f[i] = fieldElement(decompress(uint16(x>>0&0b11_1111_1111), 10))
		f[i+1] = fieldElement(decompress(uint16(x>>10&0b11_1111_1111), 10))
		f[i+2] = fieldElement(decompress(uint16(x>>20&0b11_1111_1111), 10))
		f[i+3] = fieldElement(decompress(uint16(x>>30&0b11_1111_1111), 10))
	}
	return f
}

// samplePolyCBD draws a ringElement from the special Dη distribution given a
// stream of random bytes generated by the PRF function, according to FIPS 203,
// Algorithm 8 and Definition 4.3.
func samplePolyCBD(s []byte, b byte) ringElement {
	prf := sha3.NewShake256()
	prf.Write(s)
	prf.Write([]byte{b})
	B := make([]byte, 64*η)
	prf.Read(B)

	// SamplePolyCBD simply draws four (2η) bits for each coefficient, and adds
	// the first two and subtracts the last two.

	var f ringElement
	for i := 0; i < n; i += 2 {
		b := B[i/2]
		b_7, b_6, b_5, b_4 := b>>7, b>>6&1, b>>5&1, b>>4&1
		b_3, b_2, b_1, b_0 := b>>3&1, b>>2&1, b>>1&1, b&1
		f[i] = fieldSub(fieldElement(b_0+b_1), fieldElement(b_2+b_3))
		f[i+1] = fieldSub(fieldElement(b_4+b_5), fieldElement(b_6+b_7))
	}
	return f
}

// nttElement is an NTT representation, an element of T_q, represented as an
// array according to FIPS 203, Section 2.4.4.
type nttElement [n]fieldElement

// gammas are the values ζ^2BitRev7(i)+1 mod q for each index i, according to
// FIPS 203, Appendix A (with negative values reduced to positive).
var gammas = [128]fieldElement{17, 3312, 2761, 568, 583, 2746, 2649, 680, 1637, 1692, 723, 2606, 2288, 1041, 1100, 2229, 1409, 1920, 2662, 667, 3281, 48, 233, 3096, 756, 2573, 2156, 1173, 3015, 314, 3050, 279, 1703, 1626, 1651, 1678, 2789, 540, 1789, 1540, 1847, 1482, 952, 2377, 1461, 1868, 2687, 642, 939, 2390, 2308, 1021, 2437, 892, 2388, 941, 733, 2596, 2337, 992, 268, 3061, 641, 2688, 1584, 1745, 2298, 1031, 2037, 1292, 3220, 109, 375, 2954, 2549, 780, 2090, 1239, 1645, 1684, 1063, 2266, 319, 3010, 2773, 556, 757, 2572, 2099, 1230, 561, 2768, 2466, 863, 2594, 735, 2804, 525, 1092, 2237, 403, 2926, 1026, 2303, 1143, 2186, 2150, 1179, 2775, 554, 886, 2443, 1722, 1607, 1212, 2117, 1874, 1455, 1029, 2300, 2110, 1219, 2935, 394, 885, 2444, 2154, 1175}

// nttMul multiplies two nttElements.
//
// It implements MultiplyNTTs, according to FIPS 203, Algorithm 11.
func nttMul(f, g nttElement) nttElement {
	var h nttElement
	// We use i += 2 for bounds check elimination. See https://go.dev/issue/66826.
	for i := 0; i < 256; i += 2 {
		a0, a1 := f[i], f[i+1]
		b0, b1 := g[i], g[i+1]
		h[i] = fieldAddMul(a0, b0, fieldMul(a1, b1), gammas[i/2])
		h[i+1] = fieldAddMul(a0, b1, a1, b0)
	}
	return h
}

// zetas are the values ζ^BitRev7(k) mod q for each index k, according to FIPS
// 203, Appendix A.
var zetas = [128]fieldElement{1, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797, 2786, 3260, 569, 1746, 296, 2447, 1339, 1476, 3046, 56, 2240, 1333, 1426, 2094, 535, 2882, 2393, 2879, 1974, 821, 289, 331, 3253, 1756, 1197, 2304, 2277, 2055, 650, 1977, 2513, 632, 2865, 33, 1320, 1915, 2319, 1435, 807, 452, 1438, 2868, 1534, 2402, 2647, 2617, 1481, 648, 2474, 3110, 1227, 910, 17, 2761, 583, 2649, 1637, 723, 2288, 1100, 1409, 2662, 3281, 233, 756, 2156, 3015, 3050, 1703, 1651, 2789, 1789, 1847, 952, 1461, 2687, 939, 2308, 2437, 2388, 733, 2337, 268, 641, 1584, 2298, 2037, 3220, 375, 2549, 2090, 1645, 1063, 319, 2773, 757, 2099, 561, 2466, 2594, 2804, 1092, 403, 1026, 1143, 2150, 2775, 886, 1722, 1212, 1874, 1029, 2110, 2935, 885, 2154}

// ntt maps a ringElement to its nttElement representation.
//
// It implements NTT, according to FIPS 203, Algorithm 9.
func ntt(f ringElement) nttElement {
	k := 1
	for len := 128; len >= 2; len /= 2 {
		for start := 0; start < 256; start += 2 * len {
			zeta := zetas[k]
			k++
			// Bounds check elimination hint.
			f, flen := f[start:start+len], f[start+len:start+len+len]
			for j := 0; j < len; j++ {
				t := fieldMul(zeta, flen[j])
				flen[j] = fieldSub(f[j], t)
				f[j] = fieldAdd(f[j], t)
			}
		}
	}
	return nttElement(f)
}

// inverseNTT maps a nttElement back to the ringElement it represents.
//
// It implements NTT⁻¹, according to FIPS 203, Algorithm 10.
func inverseNTT(f nttElement) ringElement {
	k := 127
	for len := 2; len <= 128; len *= 2 {
		for start := 0; start < 256; start += 2 * len {
			zeta := zetas[k]
			k--
			// Bounds check elimination hint.
			f, flen := f[start:start+len], f[start+len:start+len+len]
			for j := 0; j < len; j++ {
				t := f[j]
				f[j] = fieldAdd(t, flen[j])
				flen[j] = fieldMulSub(zeta, flen[j], t)
			}
		}
	}
	for i := range f {
		f[i] = fieldMul(f[i], 3303) // 3303 = 128⁻¹ mod q
	}
	return ringElement(f)
}

// sampleNTT draws a uniformly random nttElement from a stream of uniformly
// random bytes generated by the XOF function, according to FIPS 203,
// Algorithm 7.
func sampleNTT(rho []byte, ii, jj byte) nttElement {
	B := sha3.NewShake128()
	B.Write(rho)
	B.Write([]byte{ii, jj})

	// SampleNTT essentially draws 12 bits at a time from r, interprets them in
	// little-endian, and rejects values higher than q, until it drew 256
	// values. (The rejection rate is approximately 19%.)
	//
	// To do this from a bytes stream, it draws three bytes at a time, and
	// splits them into two uint16 appropriately masked.
	//
	//               r₀              r₁              r₂
	//       |- - - - - - - -|- - - - - - - -|- - - - - - - -|
	//
	//               Uint16(r₀ || r₁)
	//       |- - - - - - - - - - - - - - - -|
	//       |- - - - - - - - - - - -|
	//                   d₁
	//
	//                                Uint16(r₁ || r₂)
	//                       |- - - - - - - - - - - - - - - -|
	//                               |- - - - - - - - - - - -|
	//                                           d₂
	//
	// Note that in little-endian, the rightmost bits are the most significant
	// bits (dropped with a mask) and the leftmost bits are the least
	// significant bits (dropped with a right shift).

	var a nttElement
	var j int        // index into a
	var buf [24]byte // buffered reads from B
	off := len(buf)  // index into buf, starts in a "buffer fully consumed" state
	for {
		if off >= len(buf) {
			B.Read(buf[:])
			off = 0
		}
		d1 := byteorder.LeUint16(buf[off:]) & 0b1111_1111_1111
		d2 := byteorder.LeUint16(buf[off+1:]) >> 4
		off += 3
		if d1 < q {
			a[j] = fieldElement(d1)
			j++
		}
		if j >= len(a) {
			break
		}
		if d2 < q {
			a[j] = fieldElement(d2)
			j++
		}
		if j >= len(a) {
			break
		}
	}
	return a
}

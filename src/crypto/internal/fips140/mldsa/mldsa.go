// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mldsa

import (
	"bytes"
	"crypto/internal/fips140"
	"crypto/internal/fips140/drbg"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140/subtle"
	"crypto/internal/fips140deps/byteorder"
	"errors"
)

type parameters struct {
	k, l int // dimensions of A
	η    int // bound for secret coefficients
	γ1   int // log₂(γ₁), where [-γ₁+1, γ₁] is the bound of y
	γ2   int // denominator of γ₂ = (q - 1) / γ2
	λ    int // collison strength
	τ    int // number of non-zero coefficients in challenge
	ω    int // max number of hints in MakeHint
}

var (
	params44 = parameters{k: 4, l: 4, η: 2, γ1: 17, γ2: 88, λ: 128, τ: 39, ω: 80}
	params65 = parameters{k: 6, l: 5, η: 4, γ1: 19, γ2: 32, λ: 192, τ: 49, ω: 55}
	params87 = parameters{k: 8, l: 7, η: 2, γ1: 19, γ2: 32, λ: 256, τ: 60, ω: 75}
)

func pubKeySize(p parameters) int {
	// ρ + k × n × 10-bit coefficients of t₁
	return 32 + p.k*n*10/8
}

func sigSize(p parameters) int {
	// challenge + l × n × (γ₁+1)-bit coefficients of z + hint
	return (p.λ / 4) + p.l*n*(p.γ1+1)/8 + p.ω + p.k
}

const (
	PrivateKeySize = 32

	PublicKeySize44 = 32 + 4*n*10/8
	PublicKeySize65 = 32 + 6*n*10/8
	PublicKeySize87 = 32 + 8*n*10/8

	SignatureSize44 = 128/4 + 4*n*(17+1)/8 + 80 + 4
	SignatureSize65 = 192/4 + 5*n*(19+1)/8 + 55 + 6
	SignatureSize87 = 256/4 + 7*n*(19+1)/8 + 75 + 8
)

const maxK, maxL, maxλ, maxγ1 = 8, 7, 256, 19
const maxPubKeySize = PublicKeySize87

type PrivateKey struct {
	seed [32]byte
	pub  PublicKey
	s1   [maxL]nttElement
	s2   [maxK]nttElement
	t0   [maxK]nttElement
	k    [32]byte
}

func (priv *PrivateKey) Equal(x *PrivateKey) bool {
	return priv.pub.p == x.pub.p && subtle.ConstantTimeCompare(priv.seed[:], x.seed[:]) == 1
}

func (priv *PrivateKey) Bytes() []byte {
	seed := priv.seed
	return seed[:]
}

func (priv *PrivateKey) PublicKey() *PublicKey {
	// Note that this is likely to keep the entire PrivateKey reachable for
	// the lifetime of the PublicKey, which may be undesirable.
	return &priv.pub
}

type PublicKey struct {
	raw [maxPubKeySize]byte
	p   parameters
	a   [maxK * maxL]nttElement
	t1  [maxK]nttElement // NTT(t₁ ⋅ 2ᵈ)
	tr  [64]byte         // public key hash
}

func (pub *PublicKey) Equal(x *PublicKey) bool {
	size := pubKeySize(pub.p)
	return pub.p == x.p && subtle.ConstantTimeCompare(pub.raw[:size], x.raw[:size]) == 1
}

func (pub *PublicKey) Bytes() []byte {
	size := pubKeySize(pub.p)
	return bytes.Clone(pub.raw[:size])
}

func (pub *PublicKey) Parameters() string {
	switch pub.p {
	case params44:
		return "ML-DSA-44"
	case params65:
		return "ML-DSA-65"
	case params87:
		return "ML-DSA-87"
	default:
		panic("mldsa: internal error: unknown parameters")
	}
}

func GenerateKey44() *PrivateKey {
	fipsSelfTest()
	fips140.RecordApproved()
	var seed [32]byte
	drbg.Read(seed[:])
	priv := newPrivateKey(&seed, params44)
	fipsPCT(priv)
	return priv
}

func GenerateKey65() *PrivateKey {
	fipsSelfTest()
	fips140.RecordApproved()
	var seed [32]byte
	drbg.Read(seed[:])
	priv := newPrivateKey(&seed, params65)
	fipsPCT(priv)
	return priv
}

func GenerateKey87() *PrivateKey {
	fipsSelfTest()
	fips140.RecordApproved()
	var seed [32]byte
	drbg.Read(seed[:])
	priv := newPrivateKey(&seed, params87)
	fipsPCT(priv)
	return priv
}

var errInvalidSeedLength = errors.New("mldsa: invalid seed length")

func NewPrivateKey44(seed []byte) (*PrivateKey, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	if len(seed) != 32 {
		return nil, errInvalidSeedLength
	}
	return newPrivateKey((*[32]byte)(seed), params44), nil
}

func NewPrivateKey65(seed []byte) (*PrivateKey, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	if len(seed) != 32 {
		return nil, errInvalidSeedLength
	}
	return newPrivateKey((*[32]byte)(seed), params65), nil
}

func NewPrivateKey87(seed []byte) (*PrivateKey, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	if len(seed) != 32 {
		return nil, errInvalidSeedLength
	}
	return newPrivateKey((*[32]byte)(seed), params87), nil
}

func newPrivateKey(seed *[32]byte, p parameters) *PrivateKey {
	k, l := p.k, p.l

	priv := &PrivateKey{pub: PublicKey{p: p}}
	priv.seed = *seed

	ξ := sha3.NewShake256()
	ξ.Write(seed[:])
	ξ.Write([]byte{byte(k), byte(l)})
	ρ, ρs := make([]byte, 32), make([]byte, 64)
	ξ.Read(ρ)
	ξ.Read(ρs)
	ξ.Read(priv.k[:])

	A := priv.pub.a[:k*l]
	computeMatrixA(A, ρ, p)

	s1 := priv.s1[:l]
	for r := range l {
		s1[r] = ntt(sampleBoundedPoly(ρs, byte(r), p))
	}
	s2 := priv.s2[:k]
	for r := range k {
		s2[r] = ntt(sampleBoundedPoly(ρs, byte(l+r), p))
	}

	// ˆt = Â ∘ ŝ₁ + ŝ₂
	tHat := make([]nttElement, k, maxK)
	for i := range tHat {
		tHat[i] = s2[i]
		for j := range s1 {
			tHat[i] = polyAdd(tHat[i], nttMul(A[i*l+j], s1[j]))
		}
	}
	// t = NTT⁻¹(ˆt)
	t := make([]ringElement, k, maxK)
	for i := range tHat {
		t[i] = inverseNTT(tHat[i])
	}
	// (t₁, _) = Power2Round(t)
	// (_, ˆt₀) = NTT(Power2Round(t))
	t1, t0 := make([][n]uint16, k, maxK), priv.t0[:k]
	for i := range t {
		var w ringElement
		for j := range t[i] {
			t1[i][j], w[j] = power2Round(t[i][j])
		}
		t0[i] = ntt(w)
	}

	// The computations below (and their storage in the PrivateKey struct) are
	// not strictly necessary and could be deferred to PrivateKey.PublicKey().
	// That would require keeping or re-deriving ρ and t/t1, though.

	pk := pkEncode(priv.pub.raw[:0], ρ, t1, p)
	priv.pub.tr = computePublicKeyHash(pk)
	computeT1Hat(priv.pub.t1[:k], t1) // NTT(t₁ ⋅ 2ᵈ)

	return priv
}

func computeMatrixA(A []nttElement, ρ []byte, p parameters) {
	k, l := p.k, p.l
	for r := range k {
		for s := range l {
			A[r*l+s] = sampleNTT(ρ, byte(s), byte(r))
		}
	}
}

func computePublicKeyHash(pk []byte) [64]byte {
	H := sha3.NewShake256()
	H.Write(pk)
	var tr [64]byte
	H.Read(tr[:])
	return tr
}

func computeT1Hat(t1Hat []nttElement, t1 [][n]uint16) {
	for i := range t1 {
		var w ringElement
		for j := range t1[i] {
			// t₁ <= 2¹⁰ - 1
			// t₁ ⋅ 2ᵈ <= 2ᵈ(2¹⁰ - 1) = 2²³ - 2¹³ < q = 2²³ - 2¹³ + 1
			z, _ := fieldToMontgomery(uint32(t1[i][j]) << 13)
			w[j] = z
		}
		t1Hat[i] = ntt(w)
	}
}

func pkEncode(buf []byte, ρ []byte, t1 [][n]uint16, p parameters) []byte {
	pk := append(buf, ρ...)
	for _, w := range t1[:p.k] {
		// Encode four at a time into 4 * 10 bits = 5 bytes.
		for i := 0; i < n; i += 4 {
			c0 := w[i]
			c1 := w[i+1]
			c2 := w[i+2]
			c3 := w[i+3]
			b0 := byte(c0 >> 0)
			b1 := byte((c0 >> 8) | (c1 << 2))
			b2 := byte((c1 >> 6) | (c2 << 4))
			b3 := byte((c2 >> 4) | (c3 << 6))
			b4 := byte(c3 >> 2)
			pk = append(pk, b0, b1, b2, b3, b4)
		}
	}
	return pk
}

func pkDecode(pk []byte, t1 [][n]uint16, p parameters) (ρ []byte, err error) {
	if len(pk) != pubKeySize(p) {
		return nil, errInvalidPublicKeyLength
	}
	ρ, pk = pk[:32], pk[32:]
	for r := range t1 {
		// Decode four at a time from 4 * 10 bits = 5 bytes.
		for i := 0; i < n; i += 4 {
			b0, b1, b2, b3, b4 := pk[0], pk[1], pk[2], pk[3], pk[4]
			t1[r][i+0] = uint16(b0>>0) | uint16(b1&0b0000_0011)<<8
			t1[r][i+1] = uint16(b1>>2) | uint16(b2&0b0000_1111)<<6
			t1[r][i+2] = uint16(b2>>4) | uint16(b3&0b0011_1111)<<4
			t1[r][i+3] = uint16(b3>>6) | uint16(b4&0b1111_1111)<<2
			pk = pk[5:]
		}
	}
	return ρ, nil
}

var errInvalidPublicKeyLength = errors.New("mldsa: invalid public key length")

func NewPublicKey44(pk []byte) (*PublicKey, error) {
	return newPublicKey(pk, params44)
}

func NewPublicKey65(pk []byte) (*PublicKey, error) {
	return newPublicKey(pk, params65)
}

func NewPublicKey87(pk []byte) (*PublicKey, error) {
	return newPublicKey(pk, params87)
}

func newPublicKey(pk []byte, p parameters) (*PublicKey, error) {
	k, l := p.k, p.l

	t1 := make([][n]uint16, k, maxK)
	ρ, err := pkDecode(pk, t1, p)
	if err != nil {
		return nil, err
	}

	pub := &PublicKey{p: p}
	copy(pub.raw[:], pk)
	computeMatrixA(pub.a[:k*l], ρ, p)
	pub.tr = computePublicKeyHash(pk)
	computeT1Hat(pub.t1[:k], t1) // NTT(t₁ ⋅ 2ᵈ)

	return pub, nil
}

var (
	errContextTooLong    = errors.New("mldsa: context too long")
	errMessageHashLength = errors.New("mldsa: invalid message hash length")
	errRandomLength      = errors.New("mldsa: invalid random length")
)

func Sign(priv *PrivateKey, msg []byte, context string) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	var random [32]byte
	drbg.Read(random[:])
	μ, err := computeMessageHash(priv.pub.tr[:], msg, context)
	if err != nil {
		return nil, err
	}
	return signInternal(priv, &μ, &random), nil
}

func SignDeterministic(priv *PrivateKey, msg []byte, context string) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	var random [32]byte
	μ, err := computeMessageHash(priv.pub.tr[:], msg, context)
	if err != nil {
		return nil, err
	}
	return signInternal(priv, &μ, &random), nil
}

func TestingOnlySignWithRandom(priv *PrivateKey, msg []byte, context string, random []byte) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	μ, err := computeMessageHash(priv.pub.tr[:], msg, context)
	if err != nil {
		return nil, err
	}
	if len(random) != 32 {
		return nil, errRandomLength
	}
	return signInternal(priv, &μ, (*[32]byte)(random)), nil
}

func SignExternalMu(priv *PrivateKey, μ []byte) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	var random [32]byte
	drbg.Read(random[:])
	if len(μ) != 64 {
		return nil, errMessageHashLength
	}
	return signInternal(priv, (*[64]byte)(μ), &random), nil
}

func SignExternalMuDeterministic(priv *PrivateKey, μ []byte) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	var random [32]byte
	if len(μ) != 64 {
		return nil, errMessageHashLength
	}
	return signInternal(priv, (*[64]byte)(μ), &random), nil
}

func TestingOnlySignExternalMuWithRandom(priv *PrivateKey, μ []byte, random []byte) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	if len(μ) != 64 {
		return nil, errMessageHashLength
	}
	if len(random) != 32 {
		return nil, errRandomLength
	}
	return signInternal(priv, (*[64]byte)(μ), (*[32]byte)(random)), nil
}

func computeMessageHash(tr []byte, msg []byte, context string) ([64]byte, error) {
	if len(context) > 255 {
		return [64]byte{}, errContextTooLong
	}
	H := sha3.NewShake256()
	H.Write(tr)
	H.Write([]byte{0}) // ML-DSA / HashML-DSA domain separator
	H.Write([]byte{byte(len(context))})
	H.Write([]byte(context))
	H.Write(msg)
	var μ [64]byte
	H.Read(μ[:])
	return μ, nil
}

func signInternal(priv *PrivateKey, μ *[64]byte, random *[32]byte) []byte {
	p, k, l := priv.pub.p, priv.pub.p.k, priv.pub.p.l
	A, s1, s2, t0 := priv.pub.a[:k*l], priv.s1[:l], priv.s2[:k], priv.t0[:k]

	β := p.τ * p.η
	γ1 := uint32(1 << p.γ1)
	γ1β := γ1 - uint32(β)
	γ2 := (q - 1) / uint32(p.γ2)
	γ2β := γ2 - uint32(β)

	H := sha3.NewShake256()
	H.Write(priv.k[:])
	H.Write(random[:])
	H.Write(μ[:])
	nonce := make([]byte, 64)
	H.Read(nonce)

	κ := 0
sign:
	for {
		// Main rejection sampling loop. Note that leaking rejected signatures
		// leaks information about the private key. However, as explained in
		// https://pq-crystals.org/dilithium/data/dilithium-specification-round3.pdf
		// Section 5.5, we are free to leak rejected ch values, as well as which
		// check causes the rejection and which coefficient failed the check
		// (but not the value or sign of the coefficient).

		y := make([]ringElement, l, maxL)
		for r := range y {
			counter := make([]byte, 2)
			byteorder.LEPutUint16(counter, uint16(κ))
			κ++

			H.Reset()
			H.Write(nonce)
			H.Write(counter)
			v := make([]byte, (p.γ1+1)*n/8, (maxγ1+1)*n/8)
			H.Read(v)

			y[r] = bitUnpack(v, p)
		}

		// w = NTT⁻¹(Â ∘ NTT(y))
		yHat := make([]nttElement, l, maxL)
		for i := range y {
			yHat[i] = ntt(y[i])
		}
		w := make([]ringElement, k, maxK)
		for i := range w {
			var wHat nttElement
			for j := range l {
				wHat = polyAdd(wHat, nttMul(A[i*l+j], yHat[j]))
			}
			w[i] = inverseNTT(wHat)
		}

		H.Reset()
		H.Write(μ[:])
		for i := range w {
			w1Encode(H, highBits(w[i], p), p)
		}
		ch := make([]byte, p.λ/4, maxλ/4)
		H.Read(ch)

		// sampleInBall is not constant time, but see comment above about
		// leaking rejected ch values being acceptable.
		c := ntt(sampleInBall(ch, p))

		cs1 := make([]ringElement, l, maxL)
		for i := range cs1 {
			cs1[i] = inverseNTT(nttMul(c, s1[i]))
		}
		cs2 := make([]ringElement, k, maxK)
		for i := range cs2 {
			cs2[i] = inverseNTT(nttMul(c, s2[i]))
		}

		z := make([]ringElement, l, maxL)
		for i := range y {
			z[i] = polyAdd(y[i], cs1[i])

			// Reject if ||z||∞ ≥ γ1 − β
			if coefficientsExceedBound(z[i], γ1β) {
				if testingOnlyRejectionReason != nil {
					testingOnlyRejectionReason("z")
				}
				continue sign
			}
		}

		for i := range w {
			r0 := polySub(w[i], cs2[i])

			// Reject if ||LowBits(r0)||∞ ≥ γ2 − β
			if lowBitsExceedBound(r0, γ2β, p) {
				if testingOnlyRejectionReason != nil {
					testingOnlyRejectionReason("r0")
				}
				continue sign
			}
		}

		ct0 := make([]ringElement, k, maxK)
		for i := range ct0 {
			ct0[i] = inverseNTT(nttMul(c, t0[i]))

			// Reject if ||ct0||∞ ≥ γ2
			if coefficientsExceedBound(ct0[i], γ2) {
				if testingOnlyRejectionReason != nil {
					testingOnlyRejectionReason("ct0")
				}
				continue sign
			}
		}

		count1s := 0
		h := make([][n]byte, k, maxK)
		for i := range w {
			var count int
			h[i], count = makeHint(ct0[i], w[i], cs2[i], p)
			count1s += count
		}
		// Reject if number of hints > ω
		if count1s > p.ω {
			if testingOnlyRejectionReason != nil {
				testingOnlyRejectionReason("h")
			}
			continue sign
		}

		return sigEncode(ch, z, h, p)
	}
}

// testingOnlyRejectionReason is set in tests, to ensure that all rejection
// paths are covered. If not nil, it is called with a string describing the
// reason for rejection: "z", "r0", "ct0", or "h".
var testingOnlyRejectionReason func(reason string)

// w1Encode implements w1Encode from FIPS 204, writing directly into H.
func w1Encode(H *sha3.SHAKE, w [n]byte, p parameters) {
	switch p.γ2 {
	case 32:
		// Coefficients are <= (q − 1)/(2γ2) − 1 = 15, four bits each.
		buf := make([]byte, 4*n/8)
		for i := 0; i < n; i += 2 {
			b0 := w[i]
			b1 := w[i+1]
			buf[i/2] = b0 | b1<<4
		}
		H.Write(buf)
	case 88:
		// Coefficients are <= (q − 1)/(2γ2) − 1 = 43, six bits each.
		buf := make([]byte, 6*n/8)
		for i := 0; i < n; i += 4 {
			b0 := w[i]
			b1 := w[i+1]
			b2 := w[i+2]
			b3 := w[i+3]
			buf[3*i/4+0] = (b0 >> 0) | (b1 << 6)
			buf[3*i/4+1] = (b1 >> 2) | (b2 << 4)
			buf[3*i/4+2] = (b2 >> 4) | (b3 << 2)
		}
		H.Write(buf)
	default:
		panic("mldsa: internal error: unsupported γ2")
	}
}

func coefficientsExceedBound(w ringElement, bound uint32) bool {
	// If this function appears in profiles, it might be possible to deduplicate
	// the work of fieldFromMontgomery inside fieldInfinityNorm with the
	// subsequent encoding of w.
	for i := range w {
		if fieldInfinityNorm(w[i]) >= bound {
			return true
		}
	}
	return false
}

func lowBitsExceedBound(w ringElement, bound uint32, p parameters) bool {
	switch p.γ2 {
	case 32:
		for i := range w {
			_, r0 := decompose32(w[i])
			if constantTimeAbs(r0) >= bound {
				return true
			}
		}
	case 88:
		for i := range w {
			_, r0 := decompose88(w[i])
			if constantTimeAbs(r0) >= bound {
				return true
			}
		}
	default:
		panic("mldsa: internal error: unsupported γ2")
	}
	return false
}

var (
	errInvalidSignatureLength           = errors.New("mldsa: invalid signature length")
	errInvalidSignatureCoeffBounds      = errors.New("mldsa: invalid signature")
	errInvalidSignatureChallenge        = errors.New("mldsa: invalid signature")
	errInvalidSignatureHintLimits       = errors.New("mldsa: invalid signature encoding")
	errInvalidSignatureHintIndexOrder   = errors.New("mldsa: invalid signature encoding")
	errInvalidSignatureHintExtraIndices = errors.New("mldsa: invalid signature encoding")
)

func Verify(pub *PublicKey, msg, sig []byte, context string) error {
	fipsSelfTest()
	fips140.RecordApproved()
	μ, err := computeMessageHash(pub.tr[:], msg, context)
	if err != nil {
		return err
	}
	return verifyInternal(pub, &μ, sig)
}

func VerifyExternalMu(pub *PublicKey, μ []byte, sig []byte) error {
	fipsSelfTest()
	fips140.RecordApproved()
	if len(μ) != 64 {
		return errMessageHashLength
	}
	return verifyInternal(pub, (*[64]byte)(μ), sig)
}

func verifyInternal(pub *PublicKey, μ *[64]byte, sig []byte) error {
	p, k, l := pub.p, pub.p.k, pub.p.l
	t1, A := pub.t1[:k], pub.a[:k*l]

	β := p.τ * p.η
	γ1 := uint32(1 << p.γ1)
	γ1β := γ1 - uint32(β)

	z := make([]ringElement, l, maxL)
	h := make([][n]byte, k, maxK)
	ch, err := sigDecode(sig, z, h, p)
	if err != nil {
		return err
	}

	c := ntt(sampleInBall(ch, p))

	// w = Â ∘ NTT(z) − NTT(c) ∘ NTT(t₁ ⋅ 2ᵈ)
	zHat := make([]nttElement, l, maxL)
	for i := range zHat {
		zHat[i] = ntt(z[i])
	}
	w := make([]ringElement, k, maxK)
	for i := range w {
		var wHat nttElement
		for j := range l {
			wHat = polyAdd(wHat, nttMul(A[i*l+j], zHat[j]))
		}
		wHat = polySub(wHat, nttMul(c, t1[i]))
		w[i] = inverseNTT(wHat)
	}

	// Use hints h to compute w₁ from w(approx).
	w1 := make([][n]byte, k, maxK)
	for i := range w {
		w1[i] = useHint(w[i], h[i], p)
	}

	H := sha3.NewShake256()
	H.Write(μ[:])
	for i := range w {
		w1Encode(H, w1[i], p)
	}
	computedCH := make([]byte, p.λ/4, maxλ/4)
	H.Read(computedCH)

	for i := range z {
		if coefficientsExceedBound(z[i], γ1β) {
			return errInvalidSignatureCoeffBounds
		}
	}

	if !bytes.Equal(ch, computedCH) {
		return errInvalidSignatureChallenge
	}

	return nil
}

func sigEncode(ch []byte, z []ringElement, h [][n]byte, p parameters) []byte {
	sig := make([]byte, 0, sigSize(p))
	sig = append(sig, ch...)
	for i := range z {
		sig = bitPack(sig, z[i], p)
	}
	sig = hintEncode(sig, h, p)
	return sig
}

func sigDecode(sig []byte, z []ringElement, h [][n]byte, p parameters) (ch []byte, err error) {
	if len(sig) != sigSize(p) {
		return nil, errInvalidSignatureLength
	}
	ch, sig = sig[:p.λ/4], sig[p.λ/4:]
	for i := range z {
		length := (p.γ1 + 1) * n / 8
		z[i] = bitUnpack(sig[:length], p)
		sig = sig[length:]
	}
	if err := hintDecode(sig, h, p); err != nil {
		return nil, err
	}
	return ch, nil
}

func hintEncode(buf []byte, h [][n]byte, p parameters) []byte {
	ω, k := p.ω, p.k
	out, y := sliceForAppend(buf, ω+k)
	var idx byte
	for i := range k {
		for j := range n {
			if h[i][j] != 0 {
				y[idx] = byte(j)
				idx++
			}
		}
		y[ω+i] = idx
	}
	return out
}

func hintDecode(y []byte, h [][n]byte, p parameters) error {
	ω, k := p.ω, p.k
	if len(y) != ω+k {
		return errors.New("mldsa: internal error: invalid signature hint length")
	}
	var idx byte
	for i := range k {
		limit := y[ω+i]
		if limit < idx || limit > byte(ω) {
			return errInvalidSignatureHintLimits
		}
		first := idx
		for idx < limit {
			if idx > first && y[idx-1] >= y[idx] {
				return errInvalidSignatureHintIndexOrder
			}
			h[i][y[idx]] = 1
			idx++
		}
	}
	for i := idx; i < byte(ω); i++ {
		if y[i] != 0 {
			return errInvalidSignatureHintExtraIndices
		}
	}
	return nil
}

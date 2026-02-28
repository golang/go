// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mldsa

import (
	"crypto/internal/fips140/drbg"
	"errors"
	"math/bits"
)

// FIPS 204 defines a needless semi-expanded format for private keys. This is
// not a good format for key storage and exchange, because it is large and
// requires careful parsing to reject malformed keys. Seeds instead are just 32
// bytes, are always valid, and always expand to valid keys in memory. It is
// *also* a poor in-memory format, because it defers computing the NTT of s1,
// s2, and t0 and the expansion of A until signing time, which is inefficient.
// For a hot second, it looked like we could have all agreed to only use seeds,
// but unfortunately OpenSSL and BouncyCastle lobbied hard against that during
// the WGLC of the LAMPS IETF working group. Also, ACVP tests provide and expect
// semi-expanded keys, so we implement them here for testing purposes.

func semiExpandedPrivKeySize(p parameters) int {
	k, l := p.k, p.l
	ηBitlen := bits.Len(uint(p.η)) + 1
	// ρ + K + tr + l × n × η-bit coefficients of s₁ +
	// k × n × η-bit coefficients of s₂ + k × n × 13-bit coefficients of t₀
	return 32 + 32 + 64 + l*n*ηBitlen/8 + k*n*ηBitlen/8 + k*n*13/8
}

// TestingOnlyNewPrivateKeyFromSemiExpanded creates a PrivateKey from a
// semi-expanded private key encoding, for testing purposes. It rejects
// inconsistent keys.
//
// [PrivateKey.Bytes] must NOT be called on the resulting key, as it will
// produce a random value.
func TestingOnlyNewPrivateKeyFromSemiExpanded(sk []byte) (*PrivateKey, error) {
	var p parameters
	switch len(sk) {
	case semiExpandedPrivKeySize(params44):
		p = params44
	case semiExpandedPrivKeySize(params65):
		p = params65
	case semiExpandedPrivKeySize(params87):
		p = params87
	default:
		return nil, errors.New("mldsa: invalid semi-expanded private key size")
	}
	k, l := p.k, p.l

	ρ, K, tr, s1, s2, t0, err := skDecode(sk, p)
	if err != nil {
		return nil, err
	}

	priv := &PrivateKey{pub: PublicKey{p: p}}
	priv.k = K
	priv.pub.tr = tr
	A := priv.pub.a[:k*l]
	computeMatrixA(A, ρ[:], p)
	for r := range l {
		priv.s1[r] = ntt(s1[r])
	}
	for r := range k {
		priv.s2[r] = ntt(s2[r])
	}
	for r := range k {
		priv.t0[r] = ntt(t0[r])
	}

	// We need to put something in priv.seed, and putting random bytes feels
	// safer than putting anything predictable.
	drbg.Read(priv.seed[:])

	// Making this format *even more* annoying, we need to recompute t1 from ρ,
	// s1, and s2 if we want to generate the public key. This is essentially as
	// much work as regenerating everything from seed.
	//
	// You might also notice that the semi-expanded format also stores t0 and a
	// hash of the public key, though. How are we supposed to check they are
	// consistent without regenerating the public key? Do we even need to check?
	// Who knows! FIPS 204 says
	//
	//  > Note that there exist malformed inputs that can cause skDecode to
	//  > return values that are not in the correct range. Hence, skDecode
	//  > should only be run on inputs that come from trusted sources.
	//
	// so it sounds like it doesn't even want us to check the coefficients are
	// within bounds, but especially if using this format for key exchange, that
	// sounds like a bad idea. So we check everything.

	t1 := make([][n]uint16, k, maxK)
	for i := range k {
		tHat := priv.s2[i]
		for j := range l {
			tHat = polyAdd(tHat, nttMul(A[i*l+j], priv.s1[j]))
		}
		t := inverseNTT(tHat)
		for j := range n {
			r1, r0 := power2Round(t[j])
			t1[i][j] = r1
			if r0 != t0[i][j] {
				return nil, errors.New("mldsa: semi-expanded private key inconsistent with t0")
			}
		}
	}

	pk := pkEncode(priv.pub.raw[:0], ρ[:], t1, p)
	if computePublicKeyHash(pk) != tr {
		return nil, errors.New("mldsa: semi-expanded private key inconsistent with public key hash")
	}
	computeT1Hat(priv.pub.t1[:k], t1) // NTT(t₁ ⋅ 2ᵈ)

	return priv, nil
}

func TestingOnlyPrivateKeySemiExpandedBytes(priv *PrivateKey) []byte {
	k, l, η := priv.pub.p.k, priv.pub.p.l, priv.pub.p.η
	sk := make([]byte, 0, semiExpandedPrivKeySize(priv.pub.p))
	sk = append(sk, priv.pub.raw[:32]...) // ρ
	sk = append(sk, priv.k[:]...)         // K
	sk = append(sk, priv.pub.tr[:]...)    // tr
	for i := range l {
		sk = bitPackSlow(sk, inverseNTT(priv.s1[i]), η, η)
	}
	for i := range k {
		sk = bitPackSlow(sk, inverseNTT(priv.s2[i]), η, η)
	}
	const bound = 1 << (13 - 1) // 2^(d-1)
	for i := range k {
		sk = bitPackSlow(sk, inverseNTT(priv.t0[i]), bound-1, bound)
	}
	return sk
}

func skDecode(sk []byte, p parameters) (ρ, K [32]byte, tr [64]byte, s1, s2, t0 []ringElement, err error) {
	k, l, η := p.k, p.l, p.η
	if len(sk) != semiExpandedPrivKeySize(p) {
		err = errors.New("mldsa: invalid semi-expanded private key size")
		return
	}
	copy(ρ[:], sk[:32])
	sk = sk[32:]
	copy(K[:], sk[:32])
	sk = sk[32:]
	copy(tr[:], sk[:64])
	sk = sk[64:]

	s1 = make([]ringElement, l)
	for i := range l {
		length := n * bits.Len(uint(η)*2) / 8
		s1[i], err = bitUnpackSlow(sk[:length], η, η)
		if err != nil {
			return
		}
		sk = sk[length:]
	}

	s2 = make([]ringElement, k)
	for i := range k {
		length := n * bits.Len(uint(η)*2) / 8
		s2[i], err = bitUnpackSlow(sk[:length], η, η)
		if err != nil {
			return
		}
		sk = sk[length:]
	}

	const bound = 1 << (13 - 1) // 2^(d-1)
	t0 = make([]ringElement, k)
	for i := range k {
		length := n * 13 / 8
		t0[i], err = bitUnpackSlow(sk[:length], bound-1, bound)
		if err != nil {
			return
		}
		sk = sk[length:]
	}

	return
}

func bitPackSlow(buf []byte, r ringElement, a, b int) []byte {
	bitlen := bits.Len(uint(a + b))
	if bitlen <= 0 || bitlen > 16 {
		panic("mldsa: internal error: invalid bitlen")
	}
	out, v := sliceForAppend(buf, n*bitlen/8)
	var acc uint32
	var accBits uint
	for i := range r {
		w := int32(b) - fieldCenteredMod(r[i])
		acc |= uint32(w) << accBits
		accBits += uint(bitlen)
		for accBits >= 8 {
			v[0] = byte(acc)
			v = v[1:]
			acc >>= 8
			accBits -= 8
		}
	}
	if accBits > 0 {
		v[0] = byte(acc)
	}
	return out
}

func bitUnpackSlow(v []byte, a, b int) (ringElement, error) {
	bitlen := bits.Len(uint(a + b))
	if bitlen <= 0 || bitlen > 16 {
		panic("mldsa: internal error: invalid bitlen")
	}
	if len(v) != n*bitlen/8 {
		return ringElement{}, errors.New("mldsa: invalid input length for bitUnpackSlow")
	}

	mask := uint32((1 << bitlen) - 1)
	maxValue := uint32(a + b)

	var r ringElement
	var acc uint32
	var accBits uint
	vIdx := 0

	for i := range r {
		for accBits < uint(bitlen) {
			if vIdx < len(v) {
				acc |= uint32(v[vIdx]) << accBits
				vIdx++
				accBits += 8
			}
		}
		w := acc & mask
		if w > maxValue {
			return ringElement{}, errors.New("mldsa: coefficient out of range")
		}
		r[i] = fieldSubToMontgomery(uint32(b), w)
		acc >>= bitlen
		accBits -= uint(bitlen)
	}

	return r, nil
}

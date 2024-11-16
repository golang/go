// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa

import (
	"bytes"
	"crypto/internal/fips"
	"crypto/internal/fips/bigmod"
	"crypto/internal/fips/drbg"
	"crypto/internal/fips/nistec"
	"errors"
	"io"
	"sync"
)

// PrivateKey and PublicKey are not generic to make it possible to use them
// in other types without instantiating them with a specific point type.
// They are tied to one of the Curve types below through the curveID field.

type PrivateKey struct {
	pub PublicKey
	d   []byte // bigmod.(*Nat).Bytes output (fixed length)
}

func (priv *PrivateKey) Bytes() []byte {
	return priv.d
}

func (priv *PrivateKey) PublicKey() *PublicKey {
	return &priv.pub
}

type PublicKey struct {
	curve curveID
	q     []byte // uncompressed nistec Point.Bytes output
}

func (pub *PublicKey) Bytes() []byte {
	return pub.q
}

type curveID string

const (
	p224 curveID = "P-224"
	p256 curveID = "P-256"
	p384 curveID = "P-384"
	p521 curveID = "P-521"
)

type Curve[P Point[P]] struct {
	curve      curveID
	newPoint   func() P
	ordInverse func([]byte) ([]byte, error)
	N          *bigmod.Modulus
	nMinus2    []byte
}

// Point is a generic constraint for the [nistec] Point types.
type Point[P any] interface {
	*nistec.P224Point | *nistec.P256Point | *nistec.P384Point | *nistec.P521Point
	Bytes() []byte
	BytesX() ([]byte, error)
	SetBytes([]byte) (P, error)
	ScalarMult(P, []byte) (P, error)
	ScalarBaseMult([]byte) (P, error)
	Add(p1, p2 P) P
}

func precomputeParams[P Point[P]](c *Curve[P], order []byte) {
	var err error
	c.N, err = bigmod.NewModulus(order)
	if err != nil {
		panic(err)
	}
	two, _ := bigmod.NewNat().SetBytes([]byte{2}, c.N)
	c.nMinus2 = bigmod.NewNat().ExpandFor(c.N).Sub(two, c.N).Bytes(c.N)
}

func P224() *Curve[*nistec.P224Point] { return _P224() }

var _P224 = sync.OnceValue(func() *Curve[*nistec.P224Point] {
	c := &Curve[*nistec.P224Point]{
		curve:    p224,
		newPoint: nistec.NewP224Point,
	}
	precomputeParams(c, p224Order)
	return c
})

var p224Order = []byte{
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x16, 0xa2,
	0xe0, 0xb8, 0xf0, 0x3e, 0x13, 0xdd, 0x29, 0x45,
	0x5c, 0x5c, 0x2a, 0x3d,
}

func P256() *Curve[*nistec.P256Point] { return _P256() }

var _P256 = sync.OnceValue(func() *Curve[*nistec.P256Point] {
	c := &Curve[*nistec.P256Point]{
		curve:      p256,
		newPoint:   nistec.NewP256Point,
		ordInverse: nistec.P256OrdInverse,
	}
	precomputeParams(c, p256Order)
	return c
})

var p256Order = []byte{
	0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xbc, 0xe6, 0xfa, 0xad, 0xa7, 0x17, 0x9e, 0x84,
	0xf3, 0xb9, 0xca, 0xc2, 0xfc, 0x63, 0x25, 0x51}

func P384() *Curve[*nistec.P384Point] { return _P384() }

var _P384 = sync.OnceValue(func() *Curve[*nistec.P384Point] {
	c := &Curve[*nistec.P384Point]{
		curve:    p384,
		newPoint: nistec.NewP384Point,
	}
	precomputeParams(c, p384Order)
	return c
})

var p384Order = []byte{
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xc7, 0x63, 0x4d, 0x81, 0xf4, 0x37, 0x2d, 0xdf,
	0x58, 0x1a, 0x0d, 0xb2, 0x48, 0xb0, 0xa7, 0x7a,
	0xec, 0xec, 0x19, 0x6a, 0xcc, 0xc5, 0x29, 0x73}

func P521() *Curve[*nistec.P521Point] { return _P521() }

var _P521 = sync.OnceValue(func() *Curve[*nistec.P521Point] {
	c := &Curve[*nistec.P521Point]{
		curve:    p521,
		newPoint: nistec.NewP521Point,
	}
	precomputeParams(c, p521Order)
	return c
})

var p521Order = []byte{0x01, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfa,
	0x51, 0x86, 0x87, 0x83, 0xbf, 0x2f, 0x96, 0x6b,
	0x7f, 0xcc, 0x01, 0x48, 0xf7, 0x09, 0xa5, 0xd0,
	0x3b, 0xb5, 0xc9, 0xb8, 0x89, 0x9c, 0x47, 0xae,
	0xbb, 0x6f, 0xb7, 0x1e, 0x91, 0x38, 0x64, 0x09}

func NewPrivateKey[P Point[P]](c *Curve[P], D, Q []byte) (*PrivateKey, error) {
	fips.RecordApproved()
	pub, err := NewPublicKey(c, Q)
	if err != nil {
		return nil, err
	}
	d, err := bigmod.NewNat().SetBytes(D, c.N)
	if err != nil {
		return nil, err
	}
	priv := &PrivateKey{pub: *pub, d: d.Bytes(c.N)}
	fips.CAST("ECDSA PCT", func() error { return fipsPCT(c, priv) })
	return priv, nil
}

func NewPublicKey[P Point[P]](c *Curve[P], Q []byte) (*PublicKey, error) {
	// SetBytes checks that Q is a valid point on the curve, and that its
	// coordinates are reduced modulo p, fulfilling the requirements of SP
	// 800-89, Section 5.3.2.
	_, err := c.newPoint().SetBytes(Q)
	if err != nil {
		return nil, err
	}
	return &PublicKey{curve: c.curve, q: Q}, nil
}

// GenerateKey generates a new ECDSA private key pair for the specified curve.
//
// In FIPS mode, rand is ignored.
func GenerateKey[P Point[P]](c *Curve[P], rand io.Reader) (*PrivateKey, error) {
	fips.RecordApproved()
	k, Q, err := randomPoint(c, rand)
	if err != nil {
		return nil, err
	}
	priv := &PrivateKey{
		pub: PublicKey{
			curve: c.curve,
			q:     Q.Bytes(),
		},
		d: k.Bytes(c.N),
	}
	fips.CAST("ECDSA PCT", func() error { return fipsPCT(c, priv) })
	return priv, nil
}

func fipsPCT[P Point[P]](c *Curve[P], k *PrivateKey) error {
	hash := []byte{
		0x32, 0xb3, 0xda, 0x19, 0xf9, 0x44, 0xb0, 0x48,
		0x66, 0xd9, 0x31, 0xa6, 0x6c, 0x30, 0xb9, 0x4a,
		0xe7, 0x28, 0xcc, 0xe7, 0x00, 0xe4, 0xb0, 0xa5,
		0xb4, 0xfe, 0xae, 0x8f, 0x43, 0x8f, 0xde, 0xc2,
	}
	sig, err := Sign(c, k, nil, hash)
	if err != nil {
		return err
	}
	return Verify(c, &k.pub, hash, sig)
}

type testingOnlyFixedRandomPointReader struct{}

func (testingOnlyFixedRandomPointReader) Read(p []byte) (int, error) { panic("not implemented") }

// testingOnlyFixedRandomPoint is a signal to Sign that it should use a fixed
// random point to enable the CAST. This is extremely dangerous because if it
// were used with a production private key it would leak the private key.
// We use this rather than a global variable to avoid accidentally using it in
// production code. Still wish we didn't have to do this.
var testingOnlyFixedRandomPoint = testingOnlyFixedRandomPointReader{}

// randomPoint returns a random scalar and the corresponding point using a
// procedure equivalent to FIPS 186-5, Appendix A.2.2 (ECDSA Key Pair Generation
// by Rejection Sampling) and to Appendix A.2.2 (Per-Message Secret Number
// Generation of Private Keys by Rejection Sampling) followed by Step 5 of
// Section 6.4.1.
func randomPoint[P Point[P]](c *Curve[P], rand io.Reader) (k *bigmod.Nat, p P, err error) {
	for {
		b := make([]byte, c.N.Size())
		if fips.Enabled {
			if rand == testingOnlyFixedRandomPoint {
				b[len(b)-1] = 1
			} else {
				drbg.Read(b)
			}
		} else {
			if _, err := io.ReadFull(rand, b); err != nil {
				return nil, nil, err
			}
		}

		// Mask off any excess bits to increase the chance of hitting a value in
		// (0, N). These are the most dangerous lines in the package and maybe in
		// the library: a single bit of bias in the selection of nonces would likely
		// lead to key recovery, but no tests would fail. Look but DO NOT TOUCH.
		if excess := len(b)*8 - c.N.BitLen(); excess > 0 {
			// Just to be safe, assert that this only happens for the one curve that
			// doesn't have a round number of bits.
			if excess != 0 && c.curve != p521 {
				panic("ecdsa: internal error: unexpectedly masking off bits")
			}
			b[0] >>= excess
		}

		// FIPS 186-5, Appendix A.4.2 makes us check x <= N - 2 and then return
		// x + 1. Note that it follows that 0 < x + 1 < N. Instead, SetBytes
		// checks that k < N, and we explicitly check 0 != k. Since k can't be
		// negative, this is strictly equivalent. None of this matters anyway
		// because the chance of selecting zero is cryptographically negligible.
		if k, err := bigmod.NewNat().SetBytes(b, c.N); err == nil && k.IsZero() == 0 {
			p, err := c.newPoint().ScalarBaseMult(k.Bytes(c.N))
			return k, p, err
		}

		if testingOnlyRejectionSamplingLooped != nil {
			testingOnlyRejectionSamplingLooped()
		}
	}
}

// testingOnlyRejectionSamplingLooped is called when rejection sampling in
// randomPoint rejects a candidate for being higher than the modulus.
var testingOnlyRejectionSamplingLooped func()

// Signature is an ECDSA signature, where r and s are represented as big-endian
// fixed-length byte slices.
type Signature struct {
	R, S []byte
}

// Sign signs a hash (which should be the result of hashing a larger message)
// using the private key, priv. If the hash is longer than the bit-length of the
// private key's curve order, the hash will be truncated to that length.
//
// The signature is randomized. If FIPS mode is enabled, csprng is ignored.
func Sign[P Point[P]](c *Curve[P], priv *PrivateKey, csprng io.Reader, hash []byte) (*Signature, error) {
	if priv.pub.curve != c.curve {
		return nil, errors.New("ecdsa: private key does not match curve")
	}
	fips.RecordApproved()
	fipsSelfTest()
	return sign(c, priv, csprng, hash)
}

func signGeneric[P Point[P]](c *Curve[P], priv *PrivateKey, csprng io.Reader, hash []byte) (*Signature, error) {
	// FIPS 186-5, Section 6.4.1

	k, R, err := randomPoint(c, csprng)
	if err != nil {
		return nil, err
	}

	// kInv = k⁻¹
	kInv := bigmod.NewNat()
	inverse(c, kInv, k)

	Rx, err := R.BytesX()
	if err != nil {
		return nil, err
	}
	r, err := bigmod.NewNat().SetOverflowingBytes(Rx, c.N)
	if err != nil {
		return nil, err
	}

	// The spec wants us to retry here, but the chance of hitting this condition
	// on a large prime-order group like the NIST curves we support is
	// cryptographically negligible. If we hit it, something is awfully wrong.
	if r.IsZero() == 1 {
		return nil, errors.New("ecdsa: internal error: r is zero")
	}

	e := bigmod.NewNat()
	hashToNat(c, e, hash)

	s, err := bigmod.NewNat().SetBytes(priv.d, c.N)
	if err != nil {
		return nil, err
	}
	s.Mul(r, c.N)
	s.Add(e, c.N)
	s.Mul(kInv, c.N)

	// Again, the chance of this happening is cryptographically negligible.
	if s.IsZero() == 1 {
		return nil, errors.New("ecdsa: internal error: s is zero")
	}

	return &Signature{r.Bytes(c.N), s.Bytes(c.N)}, nil
}

// inverse sets kInv to the inverse of k modulo the order of the curve.
func inverse[P Point[P]](c *Curve[P], kInv, k *bigmod.Nat) {
	if c.ordInverse != nil {
		kBytes, err := c.ordInverse(k.Bytes(c.N))
		// Some platforms don't implement ordInverse, and always return an error.
		if err == nil {
			_, err := kInv.SetBytes(kBytes, c.N)
			if err != nil {
				panic("ecdsa: internal error: ordInverse produced an invalid value")
			}
			return
		}
	}

	// Calculate the inverse of s in GF(N) using Fermat's method
	// (exponentiation modulo P - 2, per Euler's theorem)
	kInv.Exp(k, c.nMinus2, c.N)
}

// hashToNat sets e to the left-most bits of hash, according to
// FIPS 186-5, Section 6.4.1, point 2 and Section 6.4.2, point 3.
func hashToNat[P Point[P]](c *Curve[P], e *bigmod.Nat, hash []byte) {
	// ECDSA asks us to take the left-most log2(N) bits of hash, and use them as
	// an integer modulo N. This is the absolute worst of all worlds: we still
	// have to reduce, because the result might still overflow N, but to take
	// the left-most bits for P-521 we have to do a right shift.
	if size := c.N.Size(); len(hash) >= size {
		hash = hash[:size]
		if excess := len(hash)*8 - c.N.BitLen(); excess > 0 {
			hash = bytes.Clone(hash)
			for i := len(hash) - 1; i >= 0; i-- {
				hash[i] >>= excess
				if i > 0 {
					hash[i] |= hash[i-1] << (8 - excess)
				}
			}
		}
	}
	_, err := e.SetOverflowingBytes(hash, c.N)
	if err != nil {
		panic("ecdsa: internal error: truncated hash is too long")
	}
}

// Verify verifies the signature, sig, of hash (which should be the result of
// hashing a larger message) using the public key, pub. If the hash is longer
// than the bit-length of the private key's curve order, the hash will be
// truncated to that length.
//
// The inputs are not considered confidential, and may leak through timing side
// channels, or if an attacker has control of part of the inputs.
func Verify[P Point[P]](c *Curve[P], pub *PublicKey, hash []byte, sig *Signature) error {
	if pub.curve != c.curve {
		return errors.New("ecdsa: public key does not match curve")
	}
	fips.RecordApproved()
	fipsSelfTest()
	return verify(c, pub, hash, sig)
}

func verifyGeneric[P Point[P]](c *Curve[P], pub *PublicKey, hash []byte, sig *Signature) error {
	// FIPS 186-5, Section 6.4.2

	Q, err := c.newPoint().SetBytes(pub.q)
	if err != nil {
		return err
	}

	r, err := bigmod.NewNat().SetBytes(sig.R, c.N)
	if err != nil {
		return err
	}
	if r.IsZero() == 1 {
		return errors.New("ecdsa: invalid signature: r is zero")
	}
	s, err := bigmod.NewNat().SetBytes(sig.S, c.N)
	if err != nil {
		return err
	}
	if s.IsZero() == 1 {
		return errors.New("ecdsa: invalid signature: s is zero")
	}

	e := bigmod.NewNat()
	hashToNat(c, e, hash)

	// w = s⁻¹
	w := bigmod.NewNat()
	inverse(c, w, s)

	// p₁ = [e * s⁻¹]G
	p1, err := c.newPoint().ScalarBaseMult(e.Mul(w, c.N).Bytes(c.N))
	if err != nil {
		return err
	}
	// p₂ = [r * s⁻¹]Q
	p2, err := Q.ScalarMult(Q, w.Mul(r, c.N).Bytes(c.N))
	if err != nil {
		return err
	}
	// BytesX returns an error for the point at infinity.
	Rx, err := p1.Add(p1, p2).BytesX()
	if err != nil {
		return err
	}

	v, err := bigmod.NewNat().SetOverflowingBytes(Rx, c.N)
	if err != nil {
		return err
	}

	if v.Equal(r) != 1 {
		return errors.New("ecdsa: signature did not verify")
	}
	return nil
}

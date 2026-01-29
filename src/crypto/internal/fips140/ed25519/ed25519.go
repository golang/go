// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ed25519

import (
	"bytes"
	"crypto/internal/fips140"
	"crypto/internal/fips140/drbg"
	"crypto/internal/fips140/edwards25519"
	"crypto/internal/fips140/sha512"
	"errors"
	"strconv"
)

// See https://blog.mozilla.org/warner/2011/11/29/ed25519-keys/ for the
// components of the keys and the moving parts of the algorithm.

const (
	seedSize       = 32
	publicKeySize  = 32
	privateKeySize = seedSize + publicKeySize
	signatureSize  = 64
	sha512Size     = 64
)

type PrivateKey struct {
	seed   [seedSize]byte
	pub    [publicKeySize]byte
	s      edwards25519.Scalar
	prefix [sha512Size / 2]byte
}

func (priv *PrivateKey) Bytes() []byte {
	k := make([]byte, 0, privateKeySize)
	k = append(k, priv.seed[:]...)
	k = append(k, priv.pub[:]...)
	return k
}

func (priv *PrivateKey) Seed() []byte {
	seed := priv.seed
	return seed[:]
}

func (priv *PrivateKey) PublicKey() []byte {
	pub := priv.pub
	return pub[:]
}

type PublicKey struct {
	a      edwards25519.Point
	aBytes [32]byte
}

func (pub *PublicKey) Bytes() []byte {
	a := pub.aBytes
	return a[:]
}

// GenerateKey generates a new Ed25519 private key pair.
func GenerateKey() (*PrivateKey, error) {
	priv := &PrivateKey{}
	return generateKey(priv)
}

func generateKey(priv *PrivateKey) (*PrivateKey, error) {
	fips140.RecordApproved()
	drbg.Read(priv.seed[:])
	precomputePrivateKey(priv)
	fipsPCT(priv)
	return priv, nil
}

func NewPrivateKeyFromSeed(seed []byte) (*PrivateKey, error) {
	priv := &PrivateKey{}
	return newPrivateKeyFromSeed(priv, seed)
}

func newPrivateKeyFromSeed(priv *PrivateKey, seed []byte) (*PrivateKey, error) {
	fips140.RecordApproved()
	if l := len(seed); l != seedSize {
		return nil, errors.New("ed25519: bad seed length: " + strconv.Itoa(l))
	}
	copy(priv.seed[:], seed)
	precomputePrivateKey(priv)
	return priv, nil
}

func precomputePrivateKey(priv *PrivateKey) {
	hs := sha512.New()
	hs.Write(priv.seed[:])
	h := hs.Sum(make([]byte, 0, sha512Size))

	s, err := priv.s.SetBytesWithClamping(h[:32])
	if err != nil {
		panic("ed25519: internal error: setting scalar failed")
	}
	A := (&edwards25519.Point{}).ScalarBaseMult(s)
	copy(priv.pub[:], A.Bytes())

	copy(priv.prefix[:], h[32:])
}

func NewPrivateKey(priv []byte) (*PrivateKey, error) {
	p := &PrivateKey{}
	return newPrivateKey(p, priv)
}

func newPrivateKey(priv *PrivateKey, privBytes []byte) (*PrivateKey, error) {
	fips140.RecordApproved()
	if l := len(privBytes); l != privateKeySize {
		return nil, errors.New("ed25519: bad private key length: " + strconv.Itoa(l))
	}

	copy(priv.seed[:], privBytes[:32])

	hs := sha512.New()
	hs.Write(priv.seed[:])
	h := hs.Sum(make([]byte, 0, sha512Size))

	if _, err := priv.s.SetBytesWithClamping(h[:32]); err != nil {
		panic("ed25519: internal error: setting scalar failed")
	}
	// Note that we are not decompressing the public key point here,
	// because it takes > 20% of the time of a signature generation.
	// Signing doesn't use it as a point anyway.
	copy(priv.pub[:], privBytes[32:])

	copy(priv.prefix[:], h[32:])

	return priv, nil
}

func NewPublicKey(pub []byte) (*PublicKey, error) {
	p := &PublicKey{}
	return newPublicKey(p, pub)
}

func newPublicKey(pub *PublicKey, pubBytes []byte) (*PublicKey, error) {
	if l := len(pubBytes); l != publicKeySize {
		return nil, errors.New("ed25519: bad public key length: " + strconv.Itoa(l))
	}
	// SetBytes checks that the point is on the curve.
	if _, err := pub.a.SetBytes(pubBytes); err != nil {
		return nil, errors.New("ed25519: bad public key")
	}
	copy(pub.aBytes[:], pubBytes)
	return pub, nil
}

// Domain separation prefixes used to disambiguate Ed25519/Ed25519ph/Ed25519ctx.
// See RFC 8032, Section 2 and Section 5.1.
const (
	// domPrefixPure is empty for pure Ed25519.
	domPrefixPure = ""
	// domPrefixPh is dom2(phflag=1) for Ed25519ph. It must be followed by the
	// uint8-length prefixed context.
	domPrefixPh = "SigEd25519 no Ed25519 collisions\x01"
	// domPrefixCtx is dom2(phflag=0) for Ed25519ctx. It must be followed by the
	// uint8-length prefixed context.
	domPrefixCtx = "SigEd25519 no Ed25519 collisions\x00"
)

func Sign(priv *PrivateKey, message []byte) []byte {
	// Outline the function body so that the returned signature can be
	// stack-allocated.
	signature := make([]byte, signatureSize)
	return sign(signature, priv, message)
}

func sign(signature []byte, priv *PrivateKey, message []byte) []byte {
	fipsSelfTest()
	fips140.RecordApproved()
	return signWithDom(signature, priv, message, domPrefixPure, "")
}

func SignPH(priv *PrivateKey, message []byte, context string) ([]byte, error) {
	// Outline the function body so that the returned signature can be
	// stack-allocated.
	signature := make([]byte, signatureSize)
	return signPH(signature, priv, message, context)
}

func signPH(signature []byte, priv *PrivateKey, message []byte, context string) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	if l := len(message); l != sha512Size {
		return nil, errors.New("ed25519: bad Ed25519ph message hash length: " + strconv.Itoa(l))
	}
	if l := len(context); l > 255 {
		return nil, errors.New("ed25519: bad Ed25519ph context length: " + strconv.Itoa(l))
	}
	return signWithDom(signature, priv, message, domPrefixPh, context), nil
}

func SignCtx(priv *PrivateKey, message []byte, context string) ([]byte, error) {
	// Outline the function body so that the returned signature can be
	// stack-allocated.
	signature := make([]byte, signatureSize)
	return signCtx(signature, priv, message, context)
}

func signCtx(signature []byte, priv *PrivateKey, message []byte, context string) ([]byte, error) {
	fipsSelfTest()
	// FIPS 186-5 specifies Ed25519 and Ed25519ph (with context), but not Ed25519ctx.
	fips140.RecordNonApproved()
	// Note that per RFC 8032, Section 5.1, the context SHOULD NOT be empty.
	if l := len(context); l > 255 {
		return nil, errors.New("ed25519: bad Ed25519ctx context length: " + strconv.Itoa(l))
	}
	return signWithDom(signature, priv, message, domPrefixCtx, context), nil
}

func signWithDom(signature []byte, priv *PrivateKey, message []byte, domPrefix, context string) []byte {
	mh := sha512.New()
	if domPrefix != domPrefixPure {
		mh.Write([]byte(domPrefix))
		mh.Write([]byte{byte(len(context))})
		mh.Write([]byte(context))
	}
	mh.Write(priv.prefix[:])
	mh.Write(message)
	messageDigest := make([]byte, 0, sha512Size)
	messageDigest = mh.Sum(messageDigest)
	r, err := edwards25519.NewScalar().SetUniformBytes(messageDigest)
	if err != nil {
		panic("ed25519: internal error: setting scalar failed")
	}

	R := (&edwards25519.Point{}).ScalarBaseMult(r)

	kh := sha512.New()
	if domPrefix != domPrefixPure {
		kh.Write([]byte(domPrefix))
		kh.Write([]byte{byte(len(context))})
		kh.Write([]byte(context))
	}
	kh.Write(R.Bytes())
	kh.Write(priv.pub[:])
	kh.Write(message)
	hramDigest := make([]byte, 0, sha512Size)
	hramDigest = kh.Sum(hramDigest)
	k, err := edwards25519.NewScalar().SetUniformBytes(hramDigest)
	if err != nil {
		panic("ed25519: internal error: setting scalar failed")
	}

	S := edwards25519.NewScalar().MultiplyAdd(k, &priv.s, r)

	copy(signature[:32], R.Bytes())
	copy(signature[32:], S.Bytes())

	return signature
}

func Verify(pub *PublicKey, message, sig []byte) error {
	return verify(pub, message, sig)
}

func verify(pub *PublicKey, message, sig []byte) error {
	fipsSelfTest()
	fips140.RecordApproved()
	return verifyWithDom(pub, message, sig, domPrefixPure, "")
}

func VerifyPH(pub *PublicKey, message []byte, sig []byte, context string) error {
	fipsSelfTest()
	fips140.RecordApproved()
	if l := len(message); l != sha512Size {
		return errors.New("ed25519: bad Ed25519ph message hash length: " + strconv.Itoa(l))
	}
	if l := len(context); l > 255 {
		return errors.New("ed25519: bad Ed25519ph context length: " + strconv.Itoa(l))
	}
	return verifyWithDom(pub, message, sig, domPrefixPh, context)
}

func VerifyCtx(pub *PublicKey, message []byte, sig []byte, context string) error {
	fipsSelfTest()
	// FIPS 186-5 specifies Ed25519 and Ed25519ph (with context), but not Ed25519ctx.
	fips140.RecordNonApproved()
	if l := len(context); l > 255 {
		return errors.New("ed25519: bad Ed25519ctx context length: " + strconv.Itoa(l))
	}
	return verifyWithDom(pub, message, sig, domPrefixCtx, context)
}

func verifyWithDom(pub *PublicKey, message, sig []byte, domPrefix, context string) error {
	if l := len(sig); l != signatureSize {
		return errors.New("ed25519: bad signature length: " + strconv.Itoa(l))
	}

	if sig[63]&224 != 0 {
		return errors.New("ed25519: invalid signature")
	}

	kh := sha512.New()
	if domPrefix != domPrefixPure {
		kh.Write([]byte(domPrefix))
		kh.Write([]byte{byte(len(context))})
		kh.Write([]byte(context))
	}
	kh.Write(sig[:32])
	kh.Write(pub.aBytes[:])
	kh.Write(message)
	hramDigest := make([]byte, 0, sha512Size)
	hramDigest = kh.Sum(hramDigest)
	k, err := edwards25519.NewScalar().SetUniformBytes(hramDigest)
	if err != nil {
		panic("ed25519: internal error: setting scalar failed")
	}

	S, err := edwards25519.NewScalar().SetCanonicalBytes(sig[32:])
	if err != nil {
		return errors.New("ed25519: invalid signature")
	}

	// [S]B = R + [k]A --> [k](-A) + [S]B = R
	minusA := (&edwards25519.Point{}).Negate(&pub.a)
	R := (&edwards25519.Point{}).VarTimeDoubleScalarBaseMult(k, minusA, S)

	if !bytes.Equal(sig[:32], R.Bytes()) {
		return errors.New("ed25519: invalid signature")
	}
	return nil
}

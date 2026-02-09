// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpke

import (
	"crypto/ecdh"
	"crypto/internal/rand"
	"errors"
	"internal/byteorder"
	"slices"
)

// A KEM is a Key Encapsulation Mechanism, one of the three components of an
// HPKE ciphersuite.
type KEM interface {
	// ID returns the HPKE KEM identifier.
	ID() uint16

	// GenerateKey generates a new key pair.
	GenerateKey() (PrivateKey, error)

	// NewPublicKey deserializes a public key from bytes.
	//
	// It implements DeserializePublicKey, as defined in RFC 9180.
	NewPublicKey([]byte) (PublicKey, error)

	// NewPrivateKey deserializes a private key from bytes.
	//
	// It implements DeserializePrivateKey, as defined in RFC 9180.
	NewPrivateKey([]byte) (PrivateKey, error)

	// DeriveKeyPair derives a key pair from the given input keying material.
	//
	// It implements DeriveKeyPair, as defined in RFC 9180.
	DeriveKeyPair(ikm []byte) (PrivateKey, error)

	encSize() int
}

// NewKEM returns the KEM implementation for the given KEM ID.
//
// Applications are encouraged to use specific implementations like [DHKEM] or
// [MLKEM768X25519] instead, unless runtime agility is required.
func NewKEM(id uint16) (KEM, error) {
	switch id {
	case 0x0010: // DHKEM(P-256, HKDF-SHA256)
		return DHKEM(ecdh.P256()), nil
	case 0x0011: // DHKEM(P-384, HKDF-SHA384)
		return DHKEM(ecdh.P384()), nil
	case 0x0012: // DHKEM(P-521, HKDF-SHA512)
		return DHKEM(ecdh.P521()), nil
	case 0x0020: // DHKEM(X25519, HKDF-SHA256)
		return DHKEM(ecdh.X25519()), nil
	case 0x0041: // ML-KEM-768
		return MLKEM768(), nil
	case 0x0042: // ML-KEM-1024
		return MLKEM1024(), nil
	case 0x647a: // MLKEM768-X25519
		return MLKEM768X25519(), nil
	case 0x0050: // MLKEM768-P256
		return MLKEM768P256(), nil
	case 0x0051: // MLKEM1024-P384
		return MLKEM1024P384(), nil
	default:
		return nil, errors.New("unsupported KEM")
	}
}

// A PublicKey is an instantiation of a KEM (one of the three components of an
// HPKE ciphersuite) with an encapsulation key (i.e. the public key).
//
// A PublicKey is usually obtained from a method of the corresponding [KEM] or
// [PrivateKey], such as [KEM.NewPublicKey] or [PrivateKey.PublicKey].
type PublicKey interface {
	// KEM returns the instantiated KEM.
	KEM() KEM

	// Bytes returns the public key as the output of SerializePublicKey.
	Bytes() []byte

	encap() (sharedSecret, enc []byte, err error)
}

// A PrivateKey is an instantiation of a KEM (one of the three components of
// an HPKE ciphersuite) with a decapsulation key (i.e. the secret key).
//
// A PrivateKey is usually obtained from a method of the corresponding [KEM],
// such as [KEM.GenerateKey] or [KEM.NewPrivateKey].
type PrivateKey interface {
	// KEM returns the instantiated KEM.
	KEM() KEM

	// Bytes returns the private key as the output of SerializePrivateKey, as
	// defined in RFC 9180.
	//
	// Note that for X25519 this might not match the input to NewPrivateKey.
	// This is a requirement of RFC 9180, Section 7.1.2.
	Bytes() ([]byte, error)

	// PublicKey returns the corresponding PublicKey.
	PublicKey() PublicKey

	decap(enc []byte) (sharedSecret []byte, err error)
}

type dhKEM struct {
	kdf     KDF
	id      uint16
	curve   ecdh.Curve
	Nsecret uint16
	Nsk     uint16
	Nenc    int
}

func (kem *dhKEM) extractAndExpand(dhKey, kemContext []byte) ([]byte, error) {
	suiteID := byteorder.BEAppendUint16([]byte("KEM"), kem.id)
	eaePRK, err := kem.kdf.labeledExtract(suiteID, nil, "eae_prk", dhKey)
	if err != nil {
		return nil, err
	}
	return kem.kdf.labeledExpand(suiteID, eaePRK, "shared_secret", kemContext, kem.Nsecret)
}

func (kem *dhKEM) ID() uint16 {
	return kem.id
}

func (kem *dhKEM) encSize() int {
	return kem.Nenc
}

var dhKEMP256 = &dhKEM{HKDFSHA256(), 0x0010, ecdh.P256(), 32, 32, 65}
var dhKEMP384 = &dhKEM{HKDFSHA384(), 0x0011, ecdh.P384(), 48, 48, 97}
var dhKEMP521 = &dhKEM{HKDFSHA512(), 0x0012, ecdh.P521(), 64, 66, 133}
var dhKEMX25519 = &dhKEM{HKDFSHA256(), 0x0020, ecdh.X25519(), 32, 32, 32}

// DHKEM returns a KEM implementing one of
//
//   - DHKEM(P-256, HKDF-SHA256)
//   - DHKEM(P-384, HKDF-SHA384)
//   - DHKEM(P-521, HKDF-SHA512)
//   - DHKEM(X25519, HKDF-SHA256)
//
// depending on curve.
func DHKEM(curve ecdh.Curve) KEM {
	switch curve {
	case ecdh.P256():
		return dhKEMP256
	case ecdh.P384():
		return dhKEMP384
	case ecdh.P521():
		return dhKEMP521
	case ecdh.X25519():
		return dhKEMX25519
	default:
		// The set of ecdh.Curve implementations is closed, because the
		// interface has unexported methods. Therefore, this default case is
		// only hit if a new curve is added that DHKEM doesn't support.
		return unsupportedCurveKEM{}
	}
}

type unsupportedCurveKEM struct{}

func (unsupportedCurveKEM) ID() uint16 {
	return 0
}
func (unsupportedCurveKEM) GenerateKey() (PrivateKey, error) {
	return nil, errors.New("unsupported curve")
}
func (unsupportedCurveKEM) NewPublicKey([]byte) (PublicKey, error) {
	return nil, errors.New("unsupported curve")
}
func (unsupportedCurveKEM) NewPrivateKey([]byte) (PrivateKey, error) {
	return nil, errors.New("unsupported curve")
}
func (unsupportedCurveKEM) DeriveKeyPair([]byte) (PrivateKey, error) {
	return nil, errors.New("unsupported curve")
}
func (unsupportedCurveKEM) encSize() int {
	return 0
}

type dhKEMPublicKey struct {
	kem *dhKEM
	pub *ecdh.PublicKey
}

// NewDHKEMPublicKey returns a PublicKey implementing
//
//   - DHKEM(P-256, HKDF-SHA256)
//   - DHKEM(P-384, HKDF-SHA384)
//   - DHKEM(P-521, HKDF-SHA512)
//   - DHKEM(X25519, HKDF-SHA256)
//
// depending on the underlying curve of pub ([ecdh.X25519], [ecdh.P256],
// [ecdh.P384], or [ecdh.P521]).
//
// This function is meant for applications that already have an instantiated
// crypto/ecdh public key. Otherwise, applications should use the
// [KEM.NewPublicKey] method of [DHKEM].
func NewDHKEMPublicKey(pub *ecdh.PublicKey) (PublicKey, error) {
	kem, ok := DHKEM(pub.Curve()).(*dhKEM)
	if !ok {
		return nil, errors.New("unsupported curve")
	}
	return &dhKEMPublicKey{
		kem: kem,
		pub: pub,
	}, nil
}

func (kem *dhKEM) NewPublicKey(data []byte) (PublicKey, error) {
	pub, err := kem.curve.NewPublicKey(data)
	if err != nil {
		return nil, err
	}
	return NewDHKEMPublicKey(pub)
}

func (pk *dhKEMPublicKey) KEM() KEM {
	return pk.kem
}

func (pk *dhKEMPublicKey) Bytes() []byte {
	return pk.pub.Bytes()
}

// testingOnlyGenerateKey is only used during testing, to provide
// a fixed test key to use when checking the RFC 9180 vectors.
var testingOnlyGenerateKey func() *ecdh.PrivateKey

func (pk *dhKEMPublicKey) encap() (sharedSecret []byte, encapPub []byte, err error) {
	privEph, err := pk.pub.Curve().GenerateKey(rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	if testingOnlyGenerateKey != nil {
		privEph = testingOnlyGenerateKey()
	}
	dhVal, err := privEph.ECDH(pk.pub)
	if err != nil {
		return nil, nil, err
	}
	encPubEph := privEph.PublicKey().Bytes()

	encPubRecip := pk.pub.Bytes()
	kemContext := append(encPubEph, encPubRecip...)
	sharedSecret, err = pk.kem.extractAndExpand(dhVal, kemContext)
	if err != nil {
		return nil, nil, err
	}
	return sharedSecret, encPubEph, nil
}

type dhKEMPrivateKey struct {
	kem  *dhKEM
	priv ecdh.KeyExchanger
}

// NewDHKEMPrivateKey returns a PrivateKey implementing
//
//   - DHKEM(P-256, HKDF-SHA256)
//   - DHKEM(P-384, HKDF-SHA384)
//   - DHKEM(P-521, HKDF-SHA512)
//   - DHKEM(X25519, HKDF-SHA256)
//
// depending on the underlying curve of priv ([ecdh.X25519], [ecdh.P256],
// [ecdh.P384], or [ecdh.P521]).
//
// This function is meant for applications that already have an instantiated
// crypto/ecdh private key, or another implementation of a [ecdh.KeyExchanger]
// (e.g. a hardware key). Otherwise, applications should use the
// [KEM.NewPrivateKey] method of [DHKEM].
func NewDHKEMPrivateKey(priv ecdh.KeyExchanger) (PrivateKey, error) {
	kem, ok := DHKEM(priv.Curve()).(*dhKEM)
	if !ok {
		return nil, errors.New("unsupported curve")
	}
	return &dhKEMPrivateKey{
		kem:  kem,
		priv: priv,
	}, nil
}

func (kem *dhKEM) GenerateKey() (PrivateKey, error) {
	priv, err := kem.curve.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	return NewDHKEMPrivateKey(priv)
}

func (kem *dhKEM) NewPrivateKey(ikm []byte) (PrivateKey, error) {
	priv, err := kem.curve.NewPrivateKey(ikm)
	if err != nil {
		return nil, err
	}
	return NewDHKEMPrivateKey(priv)
}

func (kem *dhKEM) DeriveKeyPair(ikm []byte) (PrivateKey, error) {
	// DeriveKeyPair from RFC 9180 Section 7.1.3.
	suiteID := byteorder.BEAppendUint16([]byte("KEM"), kem.id)
	prk, err := kem.kdf.labeledExtract(suiteID, nil, "dkp_prk", ikm)
	if err != nil {
		return nil, err
	}
	if kem == dhKEMX25519 {
		s, err := kem.kdf.labeledExpand(suiteID, prk, "sk", nil, kem.Nsk)
		if err != nil {
			return nil, err
		}
		return kem.NewPrivateKey(s)
	}
	var counter uint8
	for counter < 4 {
		s, err := kem.kdf.labeledExpand(suiteID, prk, "candidate", []byte{counter}, kem.Nsk)
		if err != nil {
			return nil, err
		}
		if kem == dhKEMP521 {
			s[0] &= 0x01
		}
		r, err := kem.NewPrivateKey(s)
		if err != nil {
			counter++
			continue
		}
		return r, nil
	}
	panic("chance of four rejections is < 2^-128")
}

func (k *dhKEMPrivateKey) KEM() KEM {
	return k.kem
}

func (k *dhKEMPrivateKey) Bytes() ([]byte, error) {
	// Bizarrely, RFC 9180, Section 7.1.2 says SerializePrivateKey MUST clamp
	// the output, which I thought we all agreed to instead do as part of the DH
	// function, letting private keys be random bytes.
	//
	// At the same time, it says DeserializePrivateKey MUST also clamp, implying
	// that the input doesn't have to be clamped, so Bytes by spec doesn't
	// necessarily match the NewPrivateKey input.
	//
	// I'm sure this will not lead to any unexpected behavior or interop issue.
	priv, ok := k.priv.(*ecdh.PrivateKey)
	if !ok {
		return nil, errors.New("ecdh: private key does not support Bytes")
	}
	if k.kem == dhKEMX25519 {
		b := priv.Bytes()
		b[0] &= 248
		b[31] &= 127
		b[31] |= 64
		return b, nil
	}
	return priv.Bytes(), nil
}

func (k *dhKEMPrivateKey) PublicKey() PublicKey {
	return &dhKEMPublicKey{
		kem: k.kem,
		pub: k.priv.PublicKey(),
	}
}

func (k *dhKEMPrivateKey) decap(encPubEph []byte) ([]byte, error) {
	pubEph, err := k.priv.Curve().NewPublicKey(encPubEph)
	if err != nil {
		return nil, err
	}
	dhVal, err := k.priv.ECDH(pubEph)
	if err != nil {
		return nil, err
	}
	kemContext := append(slices.Clip(encPubEph), k.priv.PublicKey().Bytes()...)
	return k.kem.extractAndExpand(dhVal, kemContext)
}

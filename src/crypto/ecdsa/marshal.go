// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa

import (
	"bytes"
	"crypto/elliptic"
	"encoding/asn1"
	"encoding/pem"
	"errors"
	"fmt"
)

// MarshalText implements [encoding.TextMarshaler]. The result is a PEM
// encoded public key in PKIX format.
func (pub *PublicKey) MarshalText() ([]byte, error) {
	der, err := marshalPKIXPublicKey(pub)
	if err != nil {
		return nil, err
	}
	return pem.EncodeToMemory(&pem.Block{Type: "PUBLIC KEY", Bytes: der}), nil
}

// UnmarshalText implements [encoding.TextUnmarshaler]. The input must be a
// PEM encoded public key in PKIX format.
func (pub *PublicKey) UnmarshalText(text []byte) error {
	block, rest := pem.Decode(text)
	if block == nil {
		return errors.New("ecdsa: failed to decode PEM block")
	}
	if block.Type != "PUBLIC KEY" {
		return errors.New("ecdsa: PEM block is not a public key")
	}
	if len(bytes.TrimSpace(rest)) != 0 {
		return errors.New("ecdsa: trailing data after PEM block")
	}
	p, err := parsePKIXPublicKey(block.Bytes)
	if err != nil {
		return err
	}
	*pub = *p
	return nil
}

// MarshalText implements [encoding.TextMarshaler]. The result is a PEM
// encoded private key in PKCS #8 format.
func (priv *PrivateKey) MarshalText() ([]byte, error) {
	der, err := marshalPKCS8PrivateKey(priv)
	if err != nil {
		return nil, err
	}
	return pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der}), nil
}

// UnmarshalText implements [encoding.TextUnmarshaler]. The input must be a
// PEM encoded private key in PKCS #8 format.
func (priv *PrivateKey) UnmarshalText(text []byte) error {
	block, rest := pem.Decode(text)
	if block == nil {
		return errors.New("ecdsa: failed to decode PEM block")
	}
	if block.Type != "PRIVATE KEY" {
		return errors.New("ecdsa: PEM block is not a private key")
	}
	if len(bytes.TrimSpace(rest)) != 0 {
		return errors.New("ecdsa: trailing data after PEM block")
	}
	p, err := parsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return err
	}
	*priv = *p
	return nil
}

// These types mirror the small ASN.1 structures used by crypto/x509. They
// are kept here to avoid introducing a dependency cycle through crypto/x509.
type algorithmIdentifier struct {
	Algorithm  asn1.ObjectIdentifier
	Parameters asn1.RawValue `asn1:"optional"`
}

type publicKeyInfo struct {
	Algorithm algorithmIdentifier
	PublicKey asn1.BitString
}

type pkcs8 struct {
	Version    int
	Algorithm  algorithmIdentifier
	PrivateKey []byte
}

type ecPrivateKey struct {
	Version       int
	PrivateKey    []byte
	NamedCurveOID asn1.ObjectIdentifier `asn1:"optional,explicit,tag:0"`
	PublicKey     asn1.BitString        `asn1:"optional,explicit,tag:1"`
}

const ecPrivKeyVersion = 1

var (
	oidPublicKeyECDSA = asn1.ObjectIdentifier{1, 2, 840, 10045, 2, 1}
	oidNamedCurveP224 = asn1.ObjectIdentifier{1, 3, 132, 0, 33}
	oidNamedCurveP256 = asn1.ObjectIdentifier{1, 2, 840, 10045, 3, 1, 7}
	oidNamedCurveP384 = asn1.ObjectIdentifier{1, 3, 132, 0, 34}
	oidNamedCurveP521 = asn1.ObjectIdentifier{1, 3, 132, 0, 35}
)

func oidFromNamedCurve(curve elliptic.Curve) (asn1.ObjectIdentifier, bool) {
	switch curve {
	case elliptic.P224():
		return oidNamedCurveP224, true
	case elliptic.P256():
		return oidNamedCurveP256, true
	case elliptic.P384():
		return oidNamedCurveP384, true
	case elliptic.P521():
		return oidNamedCurveP521, true
	default:
		return nil, false
	}
}

func namedCurveFromOID(oid asn1.ObjectIdentifier) elliptic.Curve {
	switch {
	case oid.Equal(oidNamedCurveP224):
		return elliptic.P224()
	case oid.Equal(oidNamedCurveP256):
		return elliptic.P256()
	case oid.Equal(oidNamedCurveP384):
		return elliptic.P384()
	case oid.Equal(oidNamedCurveP521):
		return elliptic.P521()
	default:
		return nil
	}
}

func marshalPKIXPublicKey(pub *PublicKey) ([]byte, error) {
	publicKey, err := pub.Bytes()
	if err != nil {
		return nil, err
	}
	oid, ok := oidFromNamedCurve(pub.Curve)
	if !ok {
		return nil, errors.New("ecdsa: unsupported elliptic curve")
	}
	params, err := asn1.Marshal(oid)
	if err != nil {
		return nil, err
	}
	der, err := asn1.Marshal(publicKeyInfo{
		Algorithm: algorithmIdentifier{
			Algorithm:  oidPublicKeyECDSA,
			Parameters: asn1.RawValue{FullBytes: params},
		},
		PublicKey: asn1.BitString{Bytes: publicKey, BitLength: 8 * len(publicKey)},
	})
	if err != nil {
		return nil, err
	}
	return der, nil
}

func parsePKIXPublicKey(der []byte) (*PublicKey, error) {
	var pki publicKeyInfo
	rest, err := asn1.Unmarshal(der, &pki)
	if err != nil {
		return nil, err
	}
	if len(rest) != 0 {
		return nil, errors.New("ecdsa: trailing data after public key")
	}
	if !pki.Algorithm.Algorithm.Equal(oidPublicKeyECDSA) {
		return nil, errors.New("ecdsa: not an ECDSA public key")
	}
	var curveOID asn1.ObjectIdentifier
	if _, err := asn1.Unmarshal(pki.Algorithm.Parameters.FullBytes, &curveOID); err != nil {
		return nil, errors.New("ecdsa: invalid curve parameters")
	}
	curve := namedCurveFromOID(curveOID)
	if curve == nil {
		return nil, errors.New("ecdsa: unsupported elliptic curve")
	}
	return ParseUncompressedPublicKey(curve, pki.PublicKey.RightAlign())
}

func marshalPKCS8PrivateKey(priv *PrivateKey) ([]byte, error) {
	oid, ok := oidFromNamedCurve(priv.Curve)
	if !ok {
		return nil, errors.New("ecdsa: unsupported elliptic curve")
	}
	params, err := asn1.Marshal(oid)
	if err != nil {
		return nil, err
	}
	privateKey, err := marshalECPrivateKeyWithOID(priv, nil)
	if err != nil {
		return nil, err
	}
	return asn1.Marshal(pkcs8{
		Version: 0,
		Algorithm: algorithmIdentifier{
			Algorithm:  oidPublicKeyECDSA,
			Parameters: asn1.RawValue{FullBytes: params},
		},
		PrivateKey: privateKey,
	})
}

func marshalECPrivateKeyWithOID(priv *PrivateKey, oid asn1.ObjectIdentifier) ([]byte, error) {
	privateKey, err := priv.Bytes()
	if err != nil {
		return nil, err
	}
	publicKey, err := priv.PublicKey.Bytes()
	if err != nil {
		return nil, err
	}
	return asn1.Marshal(ecPrivateKey{
		Version:       ecPrivKeyVersion,
		PrivateKey:    privateKey,
		NamedCurveOID: oid,
		PublicKey:     asn1.BitString{Bytes: publicKey, BitLength: 8 * len(publicKey)},
	})
}

func parsePKCS8PrivateKey(der []byte) (*PrivateKey, error) {
	var p8 pkcs8
	if rest, err := asn1.Unmarshal(der, &p8); err != nil {
		return nil, err
	} else if len(rest) != 0 {
		return nil, errors.New("ecdsa: trailing data after private key")
	}
	if !p8.Algorithm.Algorithm.Equal(oidPublicKeyECDSA) {
		return nil, errors.New("ecdsa: not an ECDSA private key")
	}
	var curveOID asn1.ObjectIdentifier
	if _, err := asn1.Unmarshal(p8.Algorithm.Parameters.FullBytes, &curveOID); err != nil {
		return nil, errors.New("ecdsa: invalid curve parameters")
	}
	return parseECPrivateKey(&curveOID, p8.PrivateKey)
}

func parseECPrivateKey(curveOID *asn1.ObjectIdentifier, der []byte) (*PrivateKey, error) {
	var ecKey ecPrivateKey
	if rest, err := asn1.Unmarshal(der, &ecKey); err != nil {
		return nil, err
	} else if len(rest) != 0 {
		return nil, errors.New("ecdsa: trailing data after EC private key")
	}
	if ecKey.Version != ecPrivKeyVersion {
		return nil, fmt.Errorf("ecdsa: unknown EC private key version %d", ecKey.Version)
	}
	if curveOID == nil {
		curveOID = &ecKey.NamedCurveOID
	}
	curve := namedCurveFromOID(*curveOID)
	if curve == nil {
		return nil, errors.New("ecdsa: unsupported elliptic curve")
	}
	return ParseRawPrivateKey(curve, ecKey.PrivateKey)
}

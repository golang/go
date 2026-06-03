// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"crypto/ecdh"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/mldsa"
	"crypto/rsa"
	"crypto/x509/pkix"
	"encoding/asn1"
	"errors"
	"fmt"
)

// pkcs8 reflects an ASN.1, PKCS #8 PrivateKey. See
// ftp://ftp.rsasecurity.com/pub/pkcs/pkcs-8/pkcs-8v1_2.asn
// and RFC 5208.
type pkcs8 struct {
	Version    int
	Algo       pkix.AlgorithmIdentifier
	PrivateKey []byte
	// optional attributes omitted.
}

// ParsePKCS8PrivateKey parses an unencrypted private key in PKCS #8, ASN.1 DER form.
//
// It returns a *[rsa.PrivateKey], an *[ecdsa.PrivateKey], an [ed25519.PrivateKey] (not
// a pointer), a *[mldsa.PrivateKey], or an *[ecdh.PrivateKey] (for X25519).
// More types might be supported in the future.
//
// This kind of key is commonly encoded in PEM blocks of type "PRIVATE KEY".
//
// Before Go 1.24, the CRT parameters of RSA keys were ignored and recomputed.
// To restore the old behavior, use the GODEBUG=x509rsacrt=0 environment variable.
func ParsePKCS8PrivateKey(der []byte) (key any, err error) {
	var privKey pkcs8
	if _, err := asn1.Unmarshal(der, &privKey); err != nil {
		if _, err := asn1.Unmarshal(der, &ecPrivateKey{}); err == nil {
			return nil, errors.New("x509: failed to parse private key (use ParseECPrivateKey instead for this key format)")
		}
		if _, err := asn1.Unmarshal(der, &pkcs1PrivateKey{}); err == nil {
			return nil, errors.New("x509: failed to parse private key (use ParsePKCS1PrivateKey instead for this key format)")
		}
		return nil, err
	}
	switch {
	case privKey.Algo.Algorithm.Equal(oidPublicKeyRSA):
		key, err = ParsePKCS1PrivateKey(privKey.PrivateKey)
		if err != nil {
			return nil, errors.New("x509: failed to parse RSA private key embedded in PKCS#8: " + err.Error())
		}
		return key, nil

	case privKey.Algo.Algorithm.Equal(oidPublicKeyECDSA):
		bytes := privKey.Algo.Parameters.FullBytes
		namedCurveOID := new(asn1.ObjectIdentifier)
		if _, err := asn1.Unmarshal(bytes, namedCurveOID); err != nil {
			namedCurveOID = nil
		}
		key, err = parseECPrivateKey(namedCurveOID, privKey.PrivateKey)
		if err != nil {
			return nil, errors.New("x509: failed to parse EC private key embedded in PKCS#8: " + err.Error())
		}
		return key, nil

	case privKey.Algo.Algorithm.Equal(oidPublicKeyEd25519):
		if l := len(privKey.Algo.Parameters.FullBytes); l != 0 {
			return nil, errors.New("x509: invalid Ed25519 private key parameters")
		}
		var curvePrivateKey []byte
		if _, err := asn1.Unmarshal(privKey.PrivateKey, &curvePrivateKey); err != nil {
			return nil, fmt.Errorf("x509: invalid Ed25519 private key: %v", err)
		}
		if l := len(curvePrivateKey); l != ed25519.SeedSize {
			return nil, fmt.Errorf("x509: invalid Ed25519 private key length: %d", l)
		}
		return ed25519.NewKeyFromSeed(curvePrivateKey), nil

	case privKey.Algo.Algorithm.Equal(oidPublicKeyMLDSA44),
		privKey.Algo.Algorithm.Equal(oidPublicKeyMLDSA65),
		privKey.Algo.Algorithm.Equal(oidPublicKeyMLDSA87):
		if l := len(privKey.Algo.Parameters.FullBytes); l != 0 {
			return nil, errors.New("x509: invalid ML-DSA private key parameters")
		}
		if l := len(privKey.PrivateKey); l == 0 {
			return nil, fmt.Errorf("x509: invalid ML-DSA private key length: %d", l)
		}
		switch privKey.PrivateKey[0] {
		case 0x80: // IMPLICIT [0] OCTET STRING (seed)
		case 0x04: // OCTET STRING (expandedKey)
			return nil, errors.New("x509: semi-expanded ML-DSA private keys without seed are not supported")
		case 0x30: // SEQUENCE (both)
			return nil, errors.New(`x509: ML-DSA private keys with both seed and expanded key are not supported, use e.g. "openssl pkey -provparam ml-dsa.output_formats=seed-only" to convert to a seed-only key`)
		default:
			return nil, fmt.Errorf("x509: invalid ML-DSA private key: invalid ASN.1 tag %02x", privKey.PrivateKey[0])
		}
		if l := len(privKey.PrivateKey); l != 2+mldsa.PrivateKeySize {
			return nil, fmt.Errorf("x509: invalid ML-DSA private key length: %d", l)
		}
		if privKey.PrivateKey[1] != mldsa.PrivateKeySize {
			return nil, fmt.Errorf("x509: invalid ML-DSA private key ASN.1 encoding")
		}
		params, ok := mldsaParametersFromOID(privKey.Algo.Algorithm)
		if !ok {
			return nil, errors.New("x509: unknown ML-DSA parameters")
		}
		return mldsa.NewPrivateKey(params, privKey.PrivateKey[2:])

	case privKey.Algo.Algorithm.Equal(oidPublicKeyX25519):
		if l := len(privKey.Algo.Parameters.FullBytes); l != 0 {
			return nil, errors.New("x509: invalid X25519 private key parameters")
		}
		var curvePrivateKey []byte
		if _, err := asn1.Unmarshal(privKey.PrivateKey, &curvePrivateKey); err != nil {
			return nil, fmt.Errorf("x509: invalid X25519 private key: %v", err)
		}
		return ecdh.X25519().NewPrivateKey(curvePrivateKey)

	default:
		return nil, fmt.Errorf("x509: PKCS#8 wrapping contained private key with unknown algorithm: %v", privKey.Algo.Algorithm)
	}
}

// MarshalPKCS8PrivateKey converts a private key to PKCS #8, ASN.1 DER form.
//
// The following key types are currently supported: *[rsa.PrivateKey],
// *[ecdsa.PrivateKey], [ed25519.PrivateKey] (not a pointer), *[mldsa.PrivateKey],
// and *[ecdh.PrivateKey]. Unsupported key types result in an error.
//
// This kind of key is commonly encoded in PEM blocks of type "PRIVATE KEY".
//
// MarshalPKCS8PrivateKey runs [rsa.PrivateKey.Precompute] on RSA keys.
func MarshalPKCS8PrivateKey(key any) ([]byte, error) {
	var privKey pkcs8

	switch k := key.(type) {
	case *rsa.PrivateKey:
		privKey.Algo = pkix.AlgorithmIdentifier{
			Algorithm:  oidPublicKeyRSA,
			Parameters: asn1.NullRawValue,
		}
		k.Precompute()
		if err := k.Validate(); err != nil {
			return nil, err
		}
		privKey.PrivateKey = MarshalPKCS1PrivateKey(k)

	case *ecdsa.PrivateKey:
		oid, ok := oidFromNamedCurve(k.Curve)
		if !ok {
			return nil, errors.New("x509: unknown curve while marshaling to PKCS#8")
		}
		oidBytes, err := asn1.Marshal(oid)
		if err != nil {
			return nil, errors.New("x509: failed to marshal curve OID: " + err.Error())
		}
		privKey.Algo = pkix.AlgorithmIdentifier{
			Algorithm: oidPublicKeyECDSA,
			Parameters: asn1.RawValue{
				FullBytes: oidBytes,
			},
		}
		if privKey.PrivateKey, err = marshalECPrivateKeyWithOID(k, nil); err != nil {
			return nil, errors.New("x509: failed to marshal EC private key while building PKCS#8: " + err.Error())
		}

	case ed25519.PrivateKey:
		privKey.Algo = pkix.AlgorithmIdentifier{
			Algorithm: oidPublicKeyEd25519,
		}
		curvePrivateKey, err := asn1.Marshal(k.Seed())
		if err != nil {
			return nil, fmt.Errorf("x509: failed to marshal private key: %v", err)
		}
		privKey.PrivateKey = curvePrivateKey

	case *mldsa.PrivateKey:
		oid, ok := oidFromMLDSAParameters(k.PublicKey().Parameters())
		if !ok {
			return nil, errors.New("x509: unknown ML-DSA parameters while marshaling to PKCS#8")
		}
		privKey.Algo = pkix.AlgorithmIdentifier{
			Algorithm: oid,
		}
		privKey.PrivateKey = append([]byte{0x80, mldsa.PrivateKeySize}, k.Bytes()...)

	case *ecdh.PrivateKey:
		if k.Curve() == ecdh.X25519() {
			privKey.Algo = pkix.AlgorithmIdentifier{
				Algorithm: oidPublicKeyX25519,
			}
			var err error
			if privKey.PrivateKey, err = asn1.Marshal(k.Bytes()); err != nil {
				return nil, fmt.Errorf("x509: failed to marshal private key: %v", err)
			}
		} else {
			oid, ok := oidFromECDHCurve(k.Curve())
			if !ok {
				return nil, errors.New("x509: unknown curve while marshaling to PKCS#8")
			}
			oidBytes, err := asn1.Marshal(oid)
			if err != nil {
				return nil, errors.New("x509: failed to marshal curve OID: " + err.Error())
			}
			privKey.Algo = pkix.AlgorithmIdentifier{
				Algorithm: oidPublicKeyECDSA,
				Parameters: asn1.RawValue{
					FullBytes: oidBytes,
				},
			}
			if privKey.PrivateKey, err = marshalECDHPrivateKey(k); err != nil {
				return nil, errors.New("x509: failed to marshal EC private key while building PKCS#8: " + err.Error())
			}
		}

	default:
		return nil, fmt.Errorf("x509: unknown key type while marshaling PKCS#8: %T", key)
	}

	return asn1.Marshal(privKey)
}

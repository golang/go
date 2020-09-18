// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"crypto/rsa"
	"encoding/asn1"
	"errors"
	"math/big"
)

// pkcs1PrivateKey is a structure which mirrors the PKCS #1 ASN.1 for an RSA private key.
type pkcs1PrivateKey struct {
	Version int
	N       *big.Int
	E       int
	D       *big.Int
	P       *big.Int
	Q       *big.Int
	// We ignore these values, if present, because rsa will calculate them.
	Dp   *big.Int `asn1:"optional"`
	Dq   *big.Int `asn1:"optional"`
	Qinv *big.Int `asn1:"optional"`

	AdditionalPrimes []pkcs1AdditionalRSAPrime `asn1:"optional,omitempty"`
}

type pkcs1AdditionalRSAPrime struct {
	Prime *big.Int

	// We ignore these values because rsa will calculate them.
	Exp   *big.Int
	Coeff *big.Int
}

// pkcs1PublicKey reflects the ASN.1 structure of a PKCS #1 public key.
type pkcs1PublicKey struct {
	N *big.Int
	E int
}

// ParsePKCS1PrivateKey parses an RSA private key in PKCS #1, ASN.1 DER form.
//
// This kind of key is commonly encoded in PEM blocks of type "RSA PRIVATE KEY".
func ParsePKCS1PrivateKey(der []byte) (*rsa.PrivateKey, error) {
	var priv pkcs1PrivateKey
	rest, err := asn1.Unmarshal(der, &priv)
	if len(rest) > 0 {
		return nil, asn1.SyntaxError{Msg: "trailing data"}
	}
	if err != nil {
		if _, err := asn1.Unmarshal(der, &ecPrivateKey{}); err == nil {
			return nil, errors.New("x509: failed to parse private key (use ParseECPrivateKey instead for this key format)")
		}
		if _, err := asn1.Unmarshal(der, &pkcs8{}); err == nil {
			return nil, errors.New("x509: failed to parse private key (use ParsePKCS8PrivateKey instead for this key format)")
		}
		return nil, err
	}

	if priv.Version > 1 {
		return nil, errors.New("x509: unsupported private key version")
	}

	if priv.N.Sign() <= 0 || priv.D.Sign() <= 0 || priv.P.Sign() <= 0 || priv.Q.Sign() <= 0 {
		return nil, errors.New("x509: private key contains zero or negative value")
	}

	key := new(rsa.PrivateKey)
	key.PublicKey = rsa.PublicKey{
		E: priv.E,
		N: priv.N,
	}

	key.D = priv.D
	key.Primes = make([]*big.Int, 2+len(priv.AdditionalPrimes))
	key.Primes[0] = priv.P
	key.Primes[1] = priv.Q
	for i, a := range priv.AdditionalPrimes {
		if a.Prime.Sign() <= 0 {
			return nil, errors.New("x509: private key contains zero or negative prime")
		}
		key.Primes[i+2] = a.Prime
		// We ignore the other two values because rsa will calculate
		// them as needed.
	}

	err = key.Validate()
	if err != nil {
		return nil, err
	}
	key.Precompute()

	return key, nil
}

// MarshalPKCS1PrivateKey converts an RSA private key to PKCS #1, ASN.1 DER form.
//
// This kind of key is commonly encoded in PEM blocks of type "RSA PRIVATE KEY".
// For a more flexible key format which is not RSA specific, use
// MarshalPKCS8PrivateKey.
func MarshalPKCS1PrivateKey(key *rsa.PrivateKey) []byte {
	key.Precompute()

	version := 0
	if len(key.Primes) > 2 {
		version = 1
	}

	priv := pkcs1PrivateKey{
		Version: version,
		N:       key.N,
		E:       key.PublicKey.E,
		D:       key.D,
		P:       key.Primes[0],
		Q:       key.Primes[1],
		Dp:      key.Precomputed.Dp,
		Dq:      key.Precomputed.Dq,
		Qinv:    key.Precomputed.Qinv,
	}

	priv.AdditionalPrimes = make([]pkcs1AdditionalRSAPrime, len(key.Precomputed.CRTValues))
	for i, values := range key.Precomputed.CRTValues {
		priv.AdditionalPrimes[i].Prime = key.Primes[2+i]
		priv.AdditionalPrimes[i].Exp = values.Exp
		priv.AdditionalPrimes[i].Coeff = values.Coeff
	}

	b, _ := asn1.Marshal(priv)
	return b
}

// ParsePKCS1PublicKey parses an RSA public key in PKCS #1, ASN.1 DER form.
//
// This kind of key is commonly encoded in PEM blocks of type "RSA PUBLIC KEY".
func ParsePKCS1PublicKey(der []byte) (*rsa.PublicKey, error) {
	var pub pkcs1PublicKey
	rest, err := asn1.Unmarshal(der, &pub)
	if err != nil {
		if _, err := asn1.Unmarshal(der, &publicKeyInfo{}); err == nil {
			return nil, errors.New("x509: failed to parse public key (use ParsePKIXPublicKey instead for this key format)")
		}
		return nil, err
	}
	if len(rest) > 0 {
		return nil, asn1.SyntaxError{Msg: "trailing data"}
	}

	if pub.N.Sign() <= 0 || pub.E <= 0 {
		return nil, errors.New("x509: public key contains zero or negative value")
	}
	if pub.E > 1<<31-1 {
		return nil, errors.New("x509: public key contains large public exponent")
	}

	return &rsa.PublicKey{
		E: pub.E,
		N: pub.N,
	}, nil
}

// MarshalPKCS1PublicKey converts an RSA public key to PKCS #1, ASN.1 DER form.
//
// This kind of key is commonly encoded in PEM blocks of type "RSA PUBLIC KEY".
func MarshalPKCS1PublicKey(key *rsa.PublicKey) []byte {
	derBytes, _ := asn1.Marshal(pkcs1PublicKey{
		N: key.N,
		E: key.E,
	})
	return derBytes
}

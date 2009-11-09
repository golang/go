// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//	NOTE: PACKAGE UNDER CONSTRUCTION.
//
// This package parses X.509-encoded keys and certificates.
package x509

import (
	"asn1";
	"big";
	"crypto/rsa";
	"os";
)

// pkcs1PrivateKey is a structure which mirrors the PKCS#1 ASN.1 for an RSA private key.
type pkcs1PrivateKey struct {
	Version	int;
	N	asn1.RawValue;
	E	int;
	D	asn1.RawValue;
	P	asn1.RawValue;
	Q	asn1.RawValue;
}

// rawValueIsInteger returns true iff the given ASN.1 RawValue is an INTEGER type.
func rawValueIsInteger(raw *asn1.RawValue) bool {
	return raw.Class == 0 && raw.Tag == 2 && raw.IsCompound == false
}

// ParsePKCS1PrivateKey returns an RSA private key from its ASN.1 PKCS#1 DER encoded form.
func ParsePKCS1PrivateKey(der []byte) (key *rsa.PrivateKey, err os.Error) {
	var priv pkcs1PrivateKey;
	err = asn1.Unmarshal(&priv, der);
	if err != nil {
		return
	}

	if !rawValueIsInteger(&priv.N) ||
		!rawValueIsInteger(&priv.D) ||
		!rawValueIsInteger(&priv.P) ||
		!rawValueIsInteger(&priv.Q) {
		err = asn1.StructuralError{"tags don't match"};
		return;
	}

	key = &rsa.PrivateKey{
		PublicKey: rsa.PublicKey{
			E: priv.E,
			N: new(big.Int).SetBytes(priv.N.Bytes),
		},
		D: new(big.Int).SetBytes(priv.D.Bytes),
		P: new(big.Int).SetBytes(priv.P.Bytes),
		Q: new(big.Int).SetBytes(priv.Q.Bytes),
	};

	err = key.Validate();
	if err != nil {
		return nil, err
	}
	return;
}

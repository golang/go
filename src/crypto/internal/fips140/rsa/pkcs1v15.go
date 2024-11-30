// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

// This file implements signing and verification using PKCS #1 v1.5 signatures.

import (
	"bytes"
	"crypto/internal/fips140"
	"errors"
)

// These are ASN1 DER structures:
//
//	DigestInfo ::= SEQUENCE {
//	  digestAlgorithm AlgorithmIdentifier,
//	  digest OCTET STRING
//	}
//
// For performance, we don't use the generic ASN1 encoder. Rather, we
// precompute a prefix of the digest value that makes a valid ASN1 DER string
// with the correct contents.
var hashPrefixes = map[string][]byte{
	"MD5":         {0x30, 0x20, 0x30, 0x0c, 0x06, 0x08, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x02, 0x05, 0x05, 0x00, 0x04, 0x10},
	"SHA-1":       {0x30, 0x21, 0x30, 0x09, 0x06, 0x05, 0x2b, 0x0e, 0x03, 0x02, 0x1a, 0x05, 0x00, 0x04, 0x14},
	"SHA-224":     {0x30, 0x2d, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x04, 0x05, 0x00, 0x04, 0x1c},
	"SHA-256":     {0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04, 0x20},
	"SHA-384":     {0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02, 0x05, 0x00, 0x04, 0x30},
	"SHA-512":     {0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03, 0x05, 0x00, 0x04, 0x40},
	"SHA-512/224": {0x30, 0x2d, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x05, 0x05, 0x00, 0x04, 0x1C},
	"SHA-512/256": {0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x06, 0x05, 0x00, 0x04, 0x20},
	"SHA3-224":    {0x30, 0x2d, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x07, 0x05, 0x00, 0x04, 0x1C},
	"SHA3-256":    {0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x08, 0x05, 0x00, 0x04, 0x20},
	"SHA3-384":    {0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x09, 0x05, 0x00, 0x04, 0x30},
	"SHA3-512":    {0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x0a, 0x05, 0x00, 0x04, 0x40},
	"MD5+SHA1":    {}, // A special TLS case which doesn't use an ASN1 prefix.
	"RIPEMD-160":  {0x30, 0x20, 0x30, 0x08, 0x06, 0x06, 0x28, 0xcf, 0x06, 0x03, 0x00, 0x31, 0x04, 0x14},
}

// SignPKCS1v15 calculates an RSASSA-PKCS1-v1.5 signature.
//
// hash is the name of the hash function as returned by [crypto.Hash.String]
// or the empty string to indicate that the message is signed directly.
func SignPKCS1v15(priv *PrivateKey, hash string, hashed []byte) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	checkApprovedHashName(hash)

	return signPKCS1v15(priv, hash, hashed)
}

func signPKCS1v15(priv *PrivateKey, hash string, hashed []byte) ([]byte, error) {
	em, err := pkcs1v15ConstructEM(&priv.pub, hash, hashed)
	if err != nil {
		return nil, err
	}

	return decrypt(priv, em, withCheck)
}

func pkcs1v15ConstructEM(pub *PublicKey, hash string, hashed []byte) ([]byte, error) {
	// Special case: "" is used to indicate that the data is signed directly.
	var prefix []byte
	if hash != "" {
		var ok bool
		prefix, ok = hashPrefixes[hash]
		if !ok {
			return nil, errors.New("crypto/rsa: unsupported hash function")
		}
	}

	// EM = 0x00 || 0x01 || PS || 0x00 || T
	k := pub.Size()
	if k < len(prefix)+len(hashed)+2+8+1 {
		return nil, ErrMessageTooLong
	}
	em := make([]byte, k)
	em[1] = 1
	for i := 2; i < k-len(prefix)-len(hashed)-1; i++ {
		em[i] = 0xff
	}
	copy(em[k-len(prefix)-len(hashed):], prefix)
	copy(em[k-len(hashed):], hashed)
	return em, nil
}

// VerifyPKCS1v15 verifies an RSASSA-PKCS1-v1.5 signature.
//
// hash is the name of the hash function as returned by [crypto.Hash.String]
// or the empty string to indicate that the message is signed directly.
func VerifyPKCS1v15(pub *PublicKey, hash string, hashed []byte, sig []byte) error {
	fipsSelfTest()
	fips140.RecordApproved()
	checkApprovedHashName(hash)

	return verifyPKCS1v15(pub, hash, hashed, sig)
}

func verifyPKCS1v15(pub *PublicKey, hash string, hashed []byte, sig []byte) error {
	if fipsApproved, err := checkPublicKey(pub); err != nil {
		return err
	} else if !fipsApproved {
		fips140.RecordNonApproved()
	}

	// RFC 8017 Section 8.2.2: If the length of the signature S is not k
	// octets (where k is the length in octets of the RSA modulus n), output
	// "invalid signature" and stop.
	if pub.Size() != len(sig) {
		return ErrVerification
	}

	em, err := encrypt(pub, sig)
	if err != nil {
		return ErrVerification
	}

	expected, err := pkcs1v15ConstructEM(pub, hash, hashed)
	if err != nil {
		return ErrVerification
	}
	if !bytes.Equal(em, expected) {
		return ErrVerification
	}

	return nil
}

func checkApprovedHashName(hash string) {
	switch hash {
	case "SHA-224", "SHA-256", "SHA-384", "SHA-512", "SHA-512/224", "SHA-512/256",
		"SHA3-224", "SHA3-256", "SHA3-384", "SHA3-512":
	default:
		fips140.RecordNonApproved()
	}
}

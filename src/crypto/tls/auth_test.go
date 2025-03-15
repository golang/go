// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto"
	"crypto/tls/internal/fips140tls"
	"testing"
)

func TestSignatureSelection(t *testing.T) {
	rsaCert := &Certificate{
		Certificate: [][]byte{testRSACertificate},
		PrivateKey:  testRSAPrivateKey,
	}
	pkcs1Cert := &Certificate{
		Certificate:                  [][]byte{testRSACertificate},
		PrivateKey:                   testRSAPrivateKey,
		SupportedSignatureAlgorithms: []SignatureScheme{PKCS1WithSHA1, PKCS1WithSHA256},
	}
	ecdsaCert := &Certificate{
		Certificate: [][]byte{testP256Certificate},
		PrivateKey:  testP256PrivateKey,
	}
	ed25519Cert := &Certificate{
		Certificate: [][]byte{testEd25519Certificate},
		PrivateKey:  testEd25519PrivateKey,
	}

	tests := []struct {
		cert        *Certificate
		peerSigAlgs []SignatureScheme
		tlsVersion  uint16

		expectedSigAlg  SignatureScheme
		expectedSigType uint8
		expectedHash    crypto.Hash
	}{
		{rsaCert, []SignatureScheme{PKCS1WithSHA1, PKCS1WithSHA256}, VersionTLS12, PKCS1WithSHA1, signaturePKCS1v15, crypto.SHA1},
		{rsaCert, []SignatureScheme{PKCS1WithSHA512, PKCS1WithSHA1}, VersionTLS12, PKCS1WithSHA512, signaturePKCS1v15, crypto.SHA512},
		{rsaCert, []SignatureScheme{PSSWithSHA256, PKCS1WithSHA256}, VersionTLS12, PSSWithSHA256, signatureRSAPSS, crypto.SHA256},
		{pkcs1Cert, []SignatureScheme{PSSWithSHA256, PKCS1WithSHA256}, VersionTLS12, PKCS1WithSHA256, signaturePKCS1v15, crypto.SHA256},
		{rsaCert, []SignatureScheme{PSSWithSHA384, PKCS1WithSHA1}, VersionTLS13, PSSWithSHA384, signatureRSAPSS, crypto.SHA384},
		{ecdsaCert, []SignatureScheme{ECDSAWithSHA1}, VersionTLS12, ECDSAWithSHA1, signatureECDSA, crypto.SHA1},
		{ecdsaCert, []SignatureScheme{ECDSAWithP256AndSHA256}, VersionTLS12, ECDSAWithP256AndSHA256, signatureECDSA, crypto.SHA256},
		{ecdsaCert, []SignatureScheme{ECDSAWithP256AndSHA256}, VersionTLS13, ECDSAWithP256AndSHA256, signatureECDSA, crypto.SHA256},
		{ed25519Cert, []SignatureScheme{Ed25519}, VersionTLS12, Ed25519, signatureEd25519, directSigning},
		{ed25519Cert, []SignatureScheme{Ed25519}, VersionTLS13, Ed25519, signatureEd25519, directSigning},

		// TLS 1.2 without signature_algorithms extension
		{rsaCert, nil, VersionTLS12, PKCS1WithSHA1, signaturePKCS1v15, crypto.SHA1},
		{ecdsaCert, nil, VersionTLS12, ECDSAWithSHA1, signatureECDSA, crypto.SHA1},

		// TLS 1.2 does not restrict the ECDSA curve (our ecdsaCert is P-256)
		{ecdsaCert, []SignatureScheme{ECDSAWithP384AndSHA384}, VersionTLS12, ECDSAWithP384AndSHA384, signatureECDSA, crypto.SHA384},
	}

	for testNo, test := range tests {
		if fips140tls.Required() && (test.expectedHash == crypto.SHA1 || test.expectedSigAlg == Ed25519) {
			t.Logf("skipping test[%d] - not compatible with TLS FIPS mode", testNo)
			continue
		}

		sigAlg, err := selectSignatureScheme(test.tlsVersion, test.cert, test.peerSigAlgs)
		if err != nil {
			t.Errorf("test[%d]: unexpected selectSignatureScheme error: %v", testNo, err)
		}
		if test.expectedSigAlg != sigAlg {
			t.Errorf("test[%d]: expected signature scheme %v, got %v", testNo, test.expectedSigAlg, sigAlg)
		}
		sigType, hashFunc, err := typeAndHashFromSignatureScheme(sigAlg)
		if err != nil {
			t.Errorf("test[%d]: unexpected typeAndHashFromSignatureScheme error: %v", testNo, err)
		}
		if test.expectedSigType != sigType {
			t.Errorf("test[%d]: expected signature algorithm %#x, got %#x", testNo, test.expectedSigType, sigType)
		}
		if test.expectedHash != hashFunc {
			t.Errorf("test[%d]: expected hash function %#x, got %#x", testNo, test.expectedHash, hashFunc)
		}
	}

	brokenCert := &Certificate{
		Certificate:                  [][]byte{testRSACertificate},
		PrivateKey:                   testRSAPrivateKey,
		SupportedSignatureAlgorithms: []SignatureScheme{Ed25519},
	}

	badTests := []struct {
		cert        *Certificate
		peerSigAlgs []SignatureScheme
		tlsVersion  uint16
	}{
		{rsaCert, []SignatureScheme{ECDSAWithP256AndSHA256, ECDSAWithSHA1}, VersionTLS12},
		{ecdsaCert, []SignatureScheme{PKCS1WithSHA256, PKCS1WithSHA1}, VersionTLS12},
		{rsaCert, []SignatureScheme{0}, VersionTLS12},
		{ed25519Cert, []SignatureScheme{ECDSAWithP256AndSHA256, ECDSAWithSHA1}, VersionTLS12},
		{ecdsaCert, []SignatureScheme{Ed25519}, VersionTLS12},
		{brokenCert, []SignatureScheme{Ed25519}, VersionTLS12},
		{brokenCert, []SignatureScheme{PKCS1WithSHA256}, VersionTLS12},
		// RFC 5246, Section 7.4.1.4.1, says to only consider {sha1,ecdsa} as
		// default when the extension is missing, and RFC 8422 does not update
		// it. Anyway, if a stack supports Ed25519 it better support sigalgs.
		{ed25519Cert, nil, VersionTLS12},
		// TLS 1.3 has no default signature_algorithms.
		{rsaCert, nil, VersionTLS13},
		{ecdsaCert, nil, VersionTLS13},
		{ed25519Cert, nil, VersionTLS13},
		// Wrong curve, which TLS 1.3 checks
		{ecdsaCert, []SignatureScheme{ECDSAWithP384AndSHA384}, VersionTLS13},
		// TLS 1.3 does not support PKCS1v1.5 or SHA-1.
		{rsaCert, []SignatureScheme{PKCS1WithSHA256}, VersionTLS13},
		{pkcs1Cert, []SignatureScheme{PSSWithSHA256, PKCS1WithSHA256}, VersionTLS13},
		{ecdsaCert, []SignatureScheme{ECDSAWithSHA1}, VersionTLS13},
		// The key can be too small for the hash.
		{rsaCert, []SignatureScheme{PSSWithSHA512}, VersionTLS12},
	}

	for testNo, test := range badTests {
		sigAlg, err := selectSignatureScheme(test.tlsVersion, test.cert, test.peerSigAlgs)
		if err == nil {
			t.Errorf("test[%d]: unexpected success, got %v", testNo, sigAlg)
		}
	}
}

func TestLegacyTypeAndHash(t *testing.T) {
	sigType, hashFunc, err := legacyTypeAndHashFromPublicKey(testRSAPrivateKey.Public())
	if err != nil {
		t.Errorf("RSA: unexpected error: %v", err)
	}
	if expectedSigType := signaturePKCS1v15; expectedSigType != sigType {
		t.Errorf("RSA: expected signature type %#x, got %#x", expectedSigType, sigType)
	}
	if expectedHashFunc := crypto.MD5SHA1; expectedHashFunc != hashFunc {
		t.Errorf("RSA: expected hash %#x, got %#x", expectedHashFunc, hashFunc)
	}

	sigType, hashFunc, err = legacyTypeAndHashFromPublicKey(testECDSAPrivateKey.Public())
	if err != nil {
		t.Errorf("ECDSA: unexpected error: %v", err)
	}
	if expectedSigType := signatureECDSA; expectedSigType != sigType {
		t.Errorf("ECDSA: expected signature type %#x, got %#x", expectedSigType, sigType)
	}
	if expectedHashFunc := crypto.SHA1; expectedHashFunc != hashFunc {
		t.Errorf("ECDSA: expected hash %#x, got %#x", expectedHashFunc, hashFunc)
	}

	// Ed25519 is not supported by TLS 1.0 and 1.1.
	_, _, err = legacyTypeAndHashFromPublicKey(testEd25519PrivateKey.Public())
	if err == nil {
		t.Errorf("Ed25519: unexpected success")
	}
}

// TestSupportedSignatureAlgorithms checks that all supportedSignatureAlgorithms
// have valid type and hash information.
func TestSupportedSignatureAlgorithms(t *testing.T) {
	for _, sigAlg := range supportedSignatureAlgorithms(VersionTLS12) {
		sigType, hash, err := typeAndHashFromSignatureScheme(sigAlg)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", sigAlg, err)
		}
		if sigType == 0 {
			t.Errorf("%v: missing signature type", sigAlg)
		}
		if hash == 0 && sigAlg != Ed25519 {
			t.Errorf("%v: missing hash", sigAlg)
		}
	}
}

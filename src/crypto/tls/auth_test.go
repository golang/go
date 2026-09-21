// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto"
	"crypto/fips140"
	"crypto/internal/cryptotest"
	"crypto/mldsa"
	"crypto/tls/internal/fips140tls"
	"internal/testenv"
	"strconv"
	"testing"
)

func TestSignatureSelection(t *testing.T) {
	pkcs1Cert := testRSA2048Cert
	pkcs1Cert.SupportedSignatureAlgorithms = []SignatureScheme{PKCS1WithSHA1, PKCS1WithSHA256}

	tests := []struct {
		cert        Certificate
		peerSigAlgs []SignatureScheme
		tlsVersion  uint16
		godebug     string

		expectedSigAlg  SignatureScheme
		expectedSigType uint8
		expectedHash    crypto.Hash
	}{
		{testRSA2048Cert, []SignatureScheme{PKCS1WithSHA1, PKCS1WithSHA256}, VersionTLS12, "", PKCS1WithSHA256, signaturePKCS1v15, crypto.SHA256},
		{testRSA2048Cert, []SignatureScheme{PKCS1WithSHA1, PKCS1WithSHA256}, VersionTLS12, "tlssha1=1", PKCS1WithSHA1, signaturePKCS1v15, crypto.SHA1},
		{testRSA2048Cert, []SignatureScheme{PKCS1WithSHA512, PKCS1WithSHA1}, VersionTLS12, "", PKCS1WithSHA512, signaturePKCS1v15, crypto.SHA512},
		{testRSA2048Cert, []SignatureScheme{PSSWithSHA256, PKCS1WithSHA256}, VersionTLS12, "", PSSWithSHA256, signatureRSAPSS, crypto.SHA256},
		{pkcs1Cert, []SignatureScheme{PSSWithSHA256, PKCS1WithSHA256}, VersionTLS12, "", PKCS1WithSHA256, signaturePKCS1v15, crypto.SHA256},
		{testRSA2048Cert, []SignatureScheme{PSSWithSHA384, PKCS1WithSHA1}, VersionTLS13, "", PSSWithSHA384, signatureRSAPSS, crypto.SHA384},
		{testRSA2048Cert, []SignatureScheme{PKCS1WithSHA1, PSSWithSHA384}, VersionTLS13, "", PSSWithSHA384, signatureRSAPSS, crypto.SHA384},
		{testECDSAP256Cert, []SignatureScheme{ECDSAWithSHA1, ECDSAWithP256AndSHA256}, VersionTLS12, "", ECDSAWithP256AndSHA256, signatureECDSA, crypto.SHA256},
		{testECDSAP256Cert, []SignatureScheme{ECDSAWithSHA1}, VersionTLS12, "tlssha1=1", ECDSAWithSHA1, signatureECDSA, crypto.SHA1},
		{testECDSAP256Cert, []SignatureScheme{ECDSAWithP256AndSHA256}, VersionTLS12, "", ECDSAWithP256AndSHA256, signatureECDSA, crypto.SHA256},
		{testECDSAP256Cert, []SignatureScheme{ECDSAWithP256AndSHA256}, VersionTLS13, "", ECDSAWithP256AndSHA256, signatureECDSA, crypto.SHA256},
		{testEd25519Cert, []SignatureScheme{Ed25519}, VersionTLS12, "", Ed25519, signatureEd25519, directSigning},
		{testEd25519Cert, []SignatureScheme{Ed25519}, VersionTLS13, "", Ed25519, signatureEd25519, directSigning},
		{testMLDSA44Cert, []SignatureScheme{MLDSA44}, VersionTLS13, "", MLDSA44, signatureMLDSA, directSigning},
		{testMLDSA65Cert, []SignatureScheme{MLDSA65}, VersionTLS13, "", MLDSA65, signatureMLDSA, directSigning},
		{testMLDSA87Cert, []SignatureScheme{MLDSA87}, VersionTLS13, "", MLDSA87, signatureMLDSA, directSigning},

		// TLS 1.2 without signature_algorithms extension
		{testRSA2048Cert, nil, VersionTLS12, "tlssha1=1", PKCS1WithSHA1, signaturePKCS1v15, crypto.SHA1},
		{testECDSAP256Cert, nil, VersionTLS12, "tlssha1=1", ECDSAWithSHA1, signatureECDSA, crypto.SHA1},

		// TLS 1.2 does not restrict the ECDSA curve
		{testECDSAP256Cert, []SignatureScheme{ECDSAWithP384AndSHA384}, VersionTLS12, "", ECDSAWithP384AndSHA384, signatureECDSA, crypto.SHA384},
	}

	for testNo, test := range tests {
		t.Run(strconv.Itoa(testNo), func(t *testing.T) {
			if fips140tls.Required() && test.expectedHash == crypto.SHA1 {
				t.Skip("skipping test not compatible with TLS FIPS mode")
			}
			switch test.expectedSigAlg {
			case MLDSA44, MLDSA65, MLDSA87:
				cryptotest.MustMinimumFIPS140ModuleVersion(t, "v1.26.0")
			}
			if test.godebug != "" {
				testenv.SetGODEBUG(t, test.godebug)
			} else {
				t.Parallel()
			}

			sigAlg, err := selectSignatureScheme(test.tlsVersion, &test.cert, test.peerSigAlgs)
			if err != nil {
				t.Errorf("unexpected selectSignatureScheme error: %v", err)
			}
			if test.expectedSigAlg != sigAlg {
				t.Errorf("expected signature scheme %v, got %v", test.expectedSigAlg, sigAlg)
			}
			sigType, hashFunc, err := typeAndHashFromSignatureScheme(sigAlg)
			if err != nil {
				t.Errorf("unexpected typeAndHashFromSignatureScheme error: %v", err)
			}
			if test.expectedSigType != sigType {
				t.Errorf("expected signature algorithm %#x, got %#x", test.expectedSigType, sigType)
			}
			if test.expectedHash != hashFunc {
				t.Errorf("expected hash function %#x, got %#x", test.expectedHash, hashFunc)
			}
		})
	}

	brokenCert := testRSA2048Cert
	brokenCert.SupportedSignatureAlgorithms = []SignatureScheme{Ed25519}

	badTests := []struct {
		cert        Certificate
		peerSigAlgs []SignatureScheme
		tlsVersion  uint16
	}{
		{testRSA2048Cert, []SignatureScheme{ECDSAWithP256AndSHA256, ECDSAWithSHA1}, VersionTLS12},
		{testECDSAP256Cert, []SignatureScheme{PKCS1WithSHA256, PKCS1WithSHA1}, VersionTLS12},
		{testRSA2048Cert, []SignatureScheme{0}, VersionTLS12},
		{testEd25519Cert, []SignatureScheme{ECDSAWithP256AndSHA256, ECDSAWithSHA1}, VersionTLS12},
		{testECDSAP256Cert, []SignatureScheme{Ed25519}, VersionTLS12},
		{brokenCert, []SignatureScheme{Ed25519}, VersionTLS12},
		{brokenCert, []SignatureScheme{PKCS1WithSHA256}, VersionTLS12},
		// RFC 5246, Section 7.4.1.4.1, says to only consider {sha1,ecdsa} as
		// default when the extension is missing, and RFC 8422 does not update
		// it. Anyway, if a stack supports Ed25519 it better support sigalgs.
		{testEd25519Cert, nil, VersionTLS12},
		// TLS 1.3 has no default signature_algorithms.
		{testRSA2048Cert, nil, VersionTLS13},
		{testECDSAP256Cert, nil, VersionTLS13},
		{testEd25519Cert, nil, VersionTLS13},
		// Wrong curve, which TLS 1.3 checks
		{testECDSAP256Cert, []SignatureScheme{ECDSAWithP384AndSHA384}, VersionTLS13},
		// TLS 1.3 does not support PKCS1v1.5 or SHA-1.
		{testRSA2048Cert, []SignatureScheme{PKCS1WithSHA256}, VersionTLS13},
		{pkcs1Cert, []SignatureScheme{PSSWithSHA256, PKCS1WithSHA256}, VersionTLS13},
		{testECDSAP256Cert, []SignatureScheme{ECDSAWithSHA1}, VersionTLS13},
		// The key can be too small for the hash.
		{testRSA1024Cert, []SignatureScheme{PSSWithSHA512}, VersionTLS12},
		// SHA-1 requires tlssha1=1
		{testRSA2048Cert, []SignatureScheme{PKCS1WithSHA1}, VersionTLS12},
		{testECDSAP256Cert, []SignatureScheme{ECDSAWithSHA1}, VersionTLS12},
		{testRSA2048Cert, nil, VersionTLS12},
		{testECDSAP256Cert, nil, VersionTLS12},
		// ML-DSA requires TLS 1.3.
		{testMLDSA44Cert, []SignatureScheme{MLDSA44}, VersionTLS12},
		{testMLDSA65Cert, []SignatureScheme{MLDSA65}, VersionTLS12},
		{testMLDSA87Cert, []SignatureScheme{MLDSA87}, VersionTLS12},
		// ML-DSA parameter sets don't cross-match.
		{testMLDSA44Cert, []SignatureScheme{MLDSA65}, VersionTLS13},
		{testMLDSA65Cert, []SignatureScheme{MLDSA87}, VersionTLS13},
		{testMLDSA87Cert, []SignatureScheme{MLDSA44}, VersionTLS13},
		// ML-DSA cert with non-ML-DSA peer sig algs and vice versa.
		{testMLDSA44Cert, []SignatureScheme{Ed25519}, VersionTLS13},
		{testRSA2048Cert, []SignatureScheme{MLDSA44}, VersionTLS13},
		{testECDSAP256Cert, []SignatureScheme{MLDSA44}, VersionTLS13},
	}

	for testNo, test := range badTests {
		sigAlg, err := selectSignatureScheme(test.tlsVersion, &test.cert, test.peerSigAlgs)
		if err == nil {
			t.Errorf("test[%d]: unexpected success, got %v", testNo, sigAlg)
		}
	}
}

func TestLegacyTypeAndHash(t *testing.T) {
	sigType, hashFunc, err := legacyTypeAndHashFromPublicKey(testRSA2048Key.Public())
	if err != nil {
		t.Errorf("RSA: unexpected error: %v", err)
	}
	if expectedSigType := signaturePKCS1v15; expectedSigType != sigType {
		t.Errorf("RSA: expected signature type %#x, got %#x", expectedSigType, sigType)
	}
	if expectedHashFunc := crypto.MD5SHA1; expectedHashFunc != hashFunc {
		t.Errorf("RSA: expected hash %#x, got %#x", expectedHashFunc, hashFunc)
	}

	sigType, hashFunc, err = legacyTypeAndHashFromPublicKey(testECDSAP256Key.Public())
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
	_, _, err = legacyTypeAndHashFromPublicKey(testEd25519Key.Public())
	if err == nil {
		t.Errorf("Ed25519: unexpected success")
	}

	// ML-DSA is not supported by TLS 1.0 and 1.1. Skip under FIPS 140-3 module
	// v1.0.0 which doesn't support ML-DSA public keys.
	if fips140.Version() != "v1.0.0" {
		for _, key := range []*mldsa.PrivateKey{
			testMLDSA44Key, testMLDSA65Key, testMLDSA87Key,
		} {
			if _, _, err := legacyTypeAndHashFromPublicKey(key.PublicKey()); err == nil {
				t.Errorf("%s: unexpected success", key.PublicKey().Parameters())
			}
		}
	}
}

// TestSupportedSignatureAlgorithms checks that all supportedSignatureAlgorithms
// have valid type and hash information.
func TestSupportedSignatureAlgorithms(t *testing.T) {
	for _, sigAlg := range supportedSignatureAlgorithms(VersionTLS12, VersionTLS13) {
		sigType, hash, err := typeAndHashFromSignatureScheme(sigAlg)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", sigAlg, err)
		}
		if sigType == 0 {
			t.Errorf("%v: missing signature type", sigAlg)
		}
		if hash == 0 && sigAlg != Ed25519 && sigAlg != MLDSA44 && sigAlg != MLDSA65 && sigAlg != MLDSA87 {
			t.Errorf("%v: missing hash", sigAlg)
		}
	}
}

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto"
	"testing"
)

func TestSignatureSelection(t *testing.T) {
	rsaCert := &testRSAPrivateKey.PublicKey
	ecdsaCert := &testECDSAPrivateKey.PublicKey
	sigsPKCS1WithSHA := []SignatureScheme{PKCS1WithSHA256, PKCS1WithSHA1}
	sigsPSSWithSHA := []SignatureScheme{PSSWithSHA256, PSSWithSHA384}
	sigsECDSAWithSHA := []SignatureScheme{ECDSAWithP256AndSHA256, ECDSAWithSHA1}

	tests := []struct {
		pubkey      crypto.PublicKey
		peerSigAlgs []SignatureScheme
		ourSigAlgs  []SignatureScheme
		tlsVersion  uint16

		expectedSigAlg  SignatureScheme // or 0 if ignored
		expectedSigType uint8
		expectedHash    crypto.Hash
	}{
		// Hash is fixed for RSA in TLS 1.1 and before.
		// https://tools.ietf.org/html/rfc4346#page-44
		{rsaCert, nil, nil, VersionTLS11, 0, signaturePKCS1v15, crypto.MD5SHA1},
		{rsaCert, nil, nil, VersionTLS10, 0, signaturePKCS1v15, crypto.MD5SHA1},
		{rsaCert, nil, nil, VersionSSL30, 0, signaturePKCS1v15, crypto.MD5SHA1},

		// Before TLS 1.2, there is no signature_algorithms extension
		// nor field in CertificateRequest and digitally-signed and thus
		// it should be ignored.
		{rsaCert, sigsPKCS1WithSHA, nil, VersionTLS11, 0, signaturePKCS1v15, crypto.MD5SHA1},
		{rsaCert, sigsPKCS1WithSHA, sigsPKCS1WithSHA, VersionTLS11, 0, signaturePKCS1v15, crypto.MD5SHA1},
		// Use SHA-1 for TLS 1.0 and 1.1 with ECDSA, see https://tools.ietf.org/html/rfc4492#page-20
		{ecdsaCert, sigsPKCS1WithSHA, sigsPKCS1WithSHA, VersionTLS11, 0, signatureECDSA, crypto.SHA1},
		{ecdsaCert, sigsPKCS1WithSHA, sigsPKCS1WithSHA, VersionTLS10, 0, signatureECDSA, crypto.SHA1},

		// TLS 1.2 without signature_algorithms extension
		// https://tools.ietf.org/html/rfc5246#page-47
		{rsaCert, nil, sigsPKCS1WithSHA, VersionTLS12, PKCS1WithSHA1, signaturePKCS1v15, crypto.SHA1},
		{ecdsaCert, nil, sigsPKCS1WithSHA, VersionTLS12, ECDSAWithSHA1, signatureECDSA, crypto.SHA1},

		{rsaCert, []SignatureScheme{PKCS1WithSHA1}, sigsPKCS1WithSHA, VersionTLS12, PKCS1WithSHA1, signaturePKCS1v15, crypto.SHA1},
		{rsaCert, []SignatureScheme{PKCS1WithSHA256}, sigsPKCS1WithSHA, VersionTLS12, PKCS1WithSHA256, signaturePKCS1v15, crypto.SHA256},
		// "sha_hash" may denote hashes other than SHA-1
		// https://tools.ietf.org/html/draft-ietf-tls-rfc4492bis-17#page-17
		{ecdsaCert, []SignatureScheme{ECDSAWithSHA1}, sigsECDSAWithSHA, VersionTLS12, ECDSAWithSHA1, signatureECDSA, crypto.SHA1},
		{ecdsaCert, []SignatureScheme{ECDSAWithP256AndSHA256}, sigsECDSAWithSHA, VersionTLS12, ECDSAWithP256AndSHA256, signatureECDSA, crypto.SHA256},

		// RSASSA-PSS is defined in TLS 1.3 for TLS 1.2
		// https://tools.ietf.org/html/draft-ietf-tls-tls13-21#page-45
		{rsaCert, []SignatureScheme{PSSWithSHA256}, sigsPSSWithSHA, VersionTLS12, PSSWithSHA256, signatureRSAPSS, crypto.SHA256},
	}

	for testNo, test := range tests {
		sigAlg, sigType, hashFunc, err := pickSignatureAlgorithm(test.pubkey, test.peerSigAlgs, test.ourSigAlgs, test.tlsVersion)
		if err != nil {
			t.Errorf("test[%d]: unexpected error: %v", testNo, err)
		}
		if test.expectedSigAlg != 0 && test.expectedSigAlg != sigAlg {
			t.Errorf("test[%d]: expected signature scheme %#x, got %#x", testNo, test.expectedSigAlg, sigAlg)
		}
		if test.expectedSigType != sigType {
			t.Errorf("test[%d]: expected signature algorithm %#x, got %#x", testNo, test.expectedSigType, sigType)
		}
		if test.expectedHash != hashFunc {
			t.Errorf("test[%d]: expected hash function %#x, got %#x", testNo, test.expectedHash, hashFunc)
		}
	}

	badTests := []struct {
		pubkey      crypto.PublicKey
		peerSigAlgs []SignatureScheme
		ourSigAlgs  []SignatureScheme
		tlsVersion  uint16
	}{
		{rsaCert, sigsECDSAWithSHA, sigsPKCS1WithSHA, VersionTLS12},
		{ecdsaCert, sigsPKCS1WithSHA, sigsPKCS1WithSHA, VersionTLS12},
		{ecdsaCert, sigsECDSAWithSHA, sigsPKCS1WithSHA, VersionTLS12},
		{rsaCert, []SignatureScheme{0}, sigsPKCS1WithSHA, VersionTLS12},

		// ECDSA is unspecified for SSL 3.0 in RFC 4492.
		// TODO a SSL 3.0 client cannot advertise signature_algorithms,
		// but if an application feeds an ECDSA certificate anyway, it
		// will be accepted rather than trigger a handshake failure. Ok?
		//{ecdsaCert, nil, nil, VersionSSL30},
	}

	for testNo, test := range badTests {
		sigAlg, sigType, hashFunc, err := pickSignatureAlgorithm(test.pubkey, test.peerSigAlgs, test.ourSigAlgs, test.tlsVersion)
		if err == nil {
			t.Errorf("test[%d]: unexpected success, got %#x %#x %#x", testNo, sigAlg, sigType, hashFunc)
		}
	}
}

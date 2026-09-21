// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build fips140v1.0

package x509

import "testing"

// TestParseMLDSACertificateFIPS140v1_0 verifies that ML-DSA certificates can
// still be parsed under the FIPS 140-3 module v1.0.0, which doesn't support
// ML-DSA. The parsed certificate has PublicKeyAlgorithm set to
// UnknownPublicKeyAlgorithm and a nil PublicKey, so callers can inspect the
// rest of the certificate without erroring.
func TestParseMLDSACertificateFIPS140v1_0(t *testing.T) {
	for _, tt := range []struct {
		name string
		pem  string
	}{
		{"ML-DSA-44", rfc9881ExampleCertificateMLDSA44},
		{"ML-DSA-65", rfc9881ExampleCertificateMLDSA65},
		{"ML-DSA-87", rfc9881ExampleCertificateMLDSA87},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cert, err := ParseCertificate(pemDecode(t, tt.pem))
			if err != nil {
				t.Fatalf("ParseCertificate failed: %v", err)
			}
			if cert.PublicKeyAlgorithm != UnknownPublicKeyAlgorithm {
				t.Errorf("PublicKeyAlgorithm = %v, want UnknownPublicKeyAlgorithm", cert.PublicKeyAlgorithm)
			}
			if cert.PublicKey != nil {
				t.Errorf("PublicKey = %v, want nil", cert.PublicKey)
			}
			// The rest of the certificate should still be inspectable.
			if cert.Subject.CommonName == "" {
				t.Error("Subject.CommonName is empty; expected the certificate to be parsed")
			}
		})
	}
}

// TestMLDSAUnavailableErrorsNotPanics asserts that the public x509 entry
// points return errors (rather than panicking) when ML-DSA is unavailable.
// The mldsa package documents that "methods are unreachable" on v1.0.0; this
// test ensures x509 callers stay on the error path.
func TestMLDSAUnavailableErrorsNotPanics(t *testing.T) {
	// ParsePKIXPublicKey: extracts the raw SPKI from a parsed cert and parses
	// the public key directly. Should return an error, not panic.
	cert, err := ParseCertificate(pemDecode(t, rfc9881ExampleCertificateMLDSA44))
	if err != nil {
		t.Fatalf("ParseCertificate failed: %v", err)
	}
	if _, err := ParsePKIXPublicKey(cert.RawSubjectPublicKeyInfo); err == nil {
		t.Error("ParsePKIXPublicKey: expected error, got nil")
	}
	// ParsePKCS8PrivateKey: ML-DSA seed-only private keys.
	for _, tt := range []struct {
		name string
		pem  string
	}{
		{"ML-DSA-44", rfc9881ExamplePrivateKeyMLDSA44},
		{"ML-DSA-65", rfc9881ExamplePrivateKeyMLDSA65},
		{"ML-DSA-87", rfc9881ExamplePrivateKeyMLDSA87},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ParsePKCS8PrivateKey(pemDecode(t, tt.pem)); err == nil {
				t.Error("ParsePKCS8PrivateKey: expected error, got nil")
			}
		})
	}
}

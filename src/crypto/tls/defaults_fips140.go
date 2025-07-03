// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !boringcrypto

package tls

import (
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/x509"
)

// These FIPS 140-3 policies allow anything approved by SP 800-140C
// and SP 800-140D, and tested as part of the Go Cryptographic Module.
//
// Notably, not SHA-1, 3DES, RC4, ChaCha20Poly1305, RSA PKCS #1 v1.5 key
// transport, or TLS 1.0—1.1 (because we don't test its KDF).
//
// These are not default lists, but filters to apply to the default or
// configured lists. Missing items are treated as if they were not implemented.
//
// They are applied when the fips140 GODEBUG is "on" or "only".

var (
	allowedSupportedVersionsFIPS = []uint16{
		VersionTLS12,
		VersionTLS13,
	}
	allowedCurvePreferencesFIPS = []CurveID{
		X25519MLKEM768,
		CurveP256,
		CurveP384,
		CurveP521,
	}
	allowedSignatureAlgorithmsFIPS = []SignatureScheme{
		PSSWithSHA256,
		ECDSAWithP256AndSHA256,
		Ed25519,
		PSSWithSHA384,
		PSSWithSHA512,
		PKCS1WithSHA256,
		PKCS1WithSHA384,
		PKCS1WithSHA512,
		ECDSAWithP384AndSHA384,
		ECDSAWithP521AndSHA512,
	}
	allowedCipherSuitesFIPS = []uint16{
		TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
		TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
		TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256,
		TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256,
	}
	allowedCipherSuitesTLS13FIPS = []uint16{
		TLS_AES_128_GCM_SHA256,
		TLS_AES_256_GCM_SHA384,
	}
)

func isCertificateAllowedFIPS(c *x509.Certificate) bool {
	switch k := c.PublicKey.(type) {
	case *rsa.PublicKey:
		return k.N.BitLen() >= 2048
	case *ecdsa.PublicKey:
		return k.Curve == elliptic.P256() || k.Curve == elliptic.P384() || k.Curve == elliptic.P521()
	case ed25519.PublicKey:
		return true
	default:
		return false
	}
}

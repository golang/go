// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/x509"
)

// These Go+BoringCrypto policies mostly match BoringSSL's
// ssl_compliance_policy_fips_202205, which is based on NIST SP 800-52r2.
// https://cs.opensource.google/boringssl/boringssl/+/master:ssl/ssl_lib.cc;l=3289;drc=ea7a88fa
//
// P-521 is allowed per https://go.dev/issue/71757.
//
// They are applied when crypto/tls/fipsonly is imported with GOEXPERIMENT=boringcrypto.

var (
	allowedSupportedVersionsFIPS = []uint16{
		VersionTLS12,
		VersionTLS13,
	}
	allowedCurvePreferencesFIPS = []CurveID{
		CurveP256,
		CurveP384,
		CurveP521,
	}
	allowedSupportedSignatureAlgorithmsFIPS = []SignatureScheme{
		PSSWithSHA256,
		PSSWithSHA384,
		PSSWithSHA512,
		PKCS1WithSHA256,
		ECDSAWithP256AndSHA256,
		PKCS1WithSHA384,
		ECDSAWithP384AndSHA384,
		PKCS1WithSHA512,
		ECDSAWithP521AndSHA512,
	}
	allowedCipherSuitesFIPS = []uint16{
		TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
		TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
	}
	allowedCipherSuitesTLS13FIPS = []uint16{
		TLS_AES_128_GCM_SHA256,
		TLS_AES_256_GCM_SHA384,
	}
)

func isCertificateAllowedFIPS(c *x509.Certificate) bool {
	// The key must be RSA 2048, RSA 3072, RSA 4096,
	// or ECDSA P-256, P-384, P-521.
	switch k := c.PublicKey.(type) {
	case *rsa.PublicKey:
		size := k.N.BitLen()
		return size == 2048 || size == 3072 || size == 4096
	case *ecdsa.PublicKey:
		return k.Curve == elliptic.P256() || k.Curve == elliptic.P384() || k.Curve == elliptic.P521()
	}

	return false
}

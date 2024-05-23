// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"internal/godebug"
	"slices"
	_ "unsafe" // for linkname
)

// Defaults are collected in this file to allow distributions to more easily patch
// them to apply local policies.

var tlskyber = godebug.New("tlskyber")

func defaultCurvePreferences() []CurveID {
	if tlskyber.Value() == "0" {
		return []CurveID{X25519, CurveP256, CurveP384, CurveP521}
	}
	// For now, x25519Kyber768Draft00 must always be followed by X25519.
	return []CurveID{x25519Kyber768Draft00, X25519, CurveP256, CurveP384, CurveP521}
}

// defaultSupportedSignatureAlgorithms contains the signature and hash algorithms that
// the code advertises as supported in a TLS 1.2+ ClientHello and in a TLS 1.2+
// CertificateRequest. The two fields are merged to match with TLS 1.3.
// Note that in TLS 1.2, the ECDSA algorithms are not constrained to P-256, etc.
var defaultSupportedSignatureAlgorithms = []SignatureScheme{
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
	PKCS1WithSHA1,
	ECDSAWithSHA1,
}

var tlsrsakex = godebug.New("tlsrsakex")
var tls3des = godebug.New("tls3des")

func defaultCipherSuites() []uint16 {
	suites := slices.Clone(cipherSuitesPreferenceOrder)
	return slices.DeleteFunc(suites, func(c uint16) bool {
		return disabledCipherSuites[c] ||
			tlsrsakex.Value() != "1" && rsaKexCiphers[c] ||
			tls3des.Value() != "1" && tdesCiphers[c]
	})
}

// defaultCipherSuitesTLS13 is also the preference order, since there are no
// disabled by default TLS 1.3 cipher suites. The same AES vs ChaCha20 logic as
// cipherSuitesPreferenceOrder applies.
//
// defaultCipherSuitesTLS13 should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/quic-go/quic-go
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname defaultCipherSuitesTLS13
var defaultCipherSuitesTLS13 = []uint16{
	TLS_AES_128_GCM_SHA256,
	TLS_AES_256_GCM_SHA384,
	TLS_CHACHA20_POLY1305_SHA256,
}

// defaultCipherSuitesTLS13NoAES should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/quic-go/quic-go
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname defaultCipherSuitesTLS13NoAES
var defaultCipherSuitesTLS13NoAES = []uint16{
	TLS_CHACHA20_POLY1305_SHA256,
	TLS_AES_128_GCM_SHA256,
	TLS_AES_256_GCM_SHA384,
}

var defaultSupportedVersionsFIPS = []uint16{
	VersionTLS12,
}

// defaultCurvePreferencesFIPS are the FIPS-allowed curves,
// in preference order (most preferable first).
var defaultCurvePreferencesFIPS = []CurveID{CurveP256, CurveP384, CurveP521}

// defaultSupportedSignatureAlgorithmsFIPS currently are a subset of
// defaultSupportedSignatureAlgorithms without Ed25519 and SHA-1.
var defaultSupportedSignatureAlgorithmsFIPS = []SignatureScheme{
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

// defaultCipherSuitesFIPS are the FIPS-allowed cipher suites.
var defaultCipherSuitesFIPS = []uint16{
	TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
	TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
	TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
	TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
	TLS_RSA_WITH_AES_128_GCM_SHA256,
	TLS_RSA_WITH_AES_256_GCM_SHA384,
}

// defaultCipherSuitesTLS13FIPS are the FIPS-allowed cipher suites for TLS 1.3.
var defaultCipherSuitesTLS13FIPS = []uint16{
	TLS_AES_128_GCM_SHA256,
	TLS_AES_256_GCM_SHA384,
}

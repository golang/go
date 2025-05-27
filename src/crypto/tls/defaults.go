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

var tlsmlkem = godebug.New("tlsmlkem")

// defaultCurvePreferences is the default set of supported key exchanges, as
// well as the preference order.
func defaultCurvePreferences() []CurveID {
	if tlsmlkem.Value() == "0" {
		return []CurveID{X25519, CurveP256, CurveP384, CurveP521}
	}
	return []CurveID{X25519MLKEM768, X25519, CurveP256, CurveP384, CurveP521}
}

var tlssha1 = godebug.New("tlssha1")

// defaultSupportedSignatureAlgorithms returns the signature and hash algorithms that
// the code advertises and supports in a TLS 1.2+ ClientHello and in a TLS 1.2+
// CertificateRequest. The two fields are merged to match with TLS 1.3.
// Note that in TLS 1.2, the ECDSA algorithms are not constrained to P-256, etc.
func defaultSupportedSignatureAlgorithms() []SignatureScheme {
	if tlssha1.Value() == "1" {
		return []SignatureScheme{
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
	}
	return []SignatureScheme{
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
}

// defaultSupportedSignatureAlgorithmsCert returns the signature algorithms that
// the code advertises as supported for signatures in certificates.
//
// We include all algorithms, including SHA-1 and PKCS#1 v1.5, because it's more
// likely that something on our side will be willing to accept a *-with-SHA1
// certificate (e.g. with a custom VerifyConnection or by a direct match with
// the CertPool), than that the peer would have a better certificate but is just
// choosing not to send it. crypto/x509 will refuse to verify important SHA-1
// signatures anyway.
func defaultSupportedSignatureAlgorithmsCert() []SignatureScheme {
	return []SignatureScheme{
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
}

var tlsrsakex = godebug.New("tlsrsakex")
var tls3des = godebug.New("tls3des")

func supportedCipherSuites(aesGCMPreferred bool) []uint16 {
	if aesGCMPreferred {
		return slices.Clone(cipherSuitesPreferenceOrder)
	} else {
		return slices.Clone(cipherSuitesPreferenceOrderNoAES)
	}
}

func defaultCipherSuites(aesGCMPreferred bool) []uint16 {
	cipherSuites := supportedCipherSuites(aesGCMPreferred)
	return slices.DeleteFunc(cipherSuites, func(c uint16) bool {
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
//   - github.com/sagernet/quic-go
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
//   - github.com/sagernet/quic-go
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

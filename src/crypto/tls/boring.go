// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package tls

import (
	"crypto/internal/boring/fipstls"
)

// needFIPS returns fipstls.Required(); it avoids a new import in common.go.
func needFIPS() bool {
	return fipstls.Required()
}

// fipsMinVersion replaces c.minVersion in FIPS-only mode.
func fipsMinVersion(c *Config) uint16 {
	// FIPS requires TLS 1.2.
	return VersionTLS12
}

// fipsMaxVersion replaces c.maxVersion in FIPS-only mode.
func fipsMaxVersion(c *Config) uint16 {
	// FIPS requires TLS 1.2.
	return VersionTLS12
}

// default defaultFIPSCurvePreferences is the FIPS-allowed curves,
// in preference order (most preferable first).
var defaultFIPSCurvePreferences = []CurveID{CurveP256, CurveP384, CurveP521}

// fipsCurvePreferences replaces c.curvePreferences in FIPS-only mode.
func fipsCurvePreferences(c *Config) []CurveID {
	if c == nil || len(c.CurvePreferences) == 0 {
		return defaultFIPSCurvePreferences
	}
	var list []CurveID
	for _, id := range c.CurvePreferences {
		for _, allowed := range defaultFIPSCurvePreferences {
			if id == allowed {
				list = append(list, id)
				break
			}
		}
	}
	return list
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

// fipsCipherSuites replaces c.cipherSuites in FIPS-only mode.
func fipsCipherSuites(c *Config) []uint16 {
	if c == nil || c.CipherSuites == nil {
		return defaultCipherSuitesFIPS
	}
	list := make([]uint16, 0, len(defaultCipherSuitesFIPS))
	for _, id := range c.CipherSuites {
		for _, allowed := range defaultCipherSuitesFIPS {
			if id == allowed {
				list = append(list, id)
				break
			}
		}
	}
	return list
}

// fipsSupportedSignatureAlgorithms currently are a subset of
// defaultSupportedSignatureAlgorithms without Ed25519 and SHA-1.
var fipsSupportedSignatureAlgorithms = []SignatureScheme{
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

// supportedSignatureAlgorithms returns the supported signature algorithms.
func supportedSignatureAlgorithms() []SignatureScheme {
	if !needFIPS() {
		return defaultSupportedSignatureAlgorithms
	}
	return fipsSupportedSignatureAlgorithms
}

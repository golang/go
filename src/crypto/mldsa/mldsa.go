// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mldsa implements the post-quantum ML-DSA signature scheme specified
// in [FIPS 204].
//
// This package is unavailable if using the [FIPS 140-3 Go Cryptographic Module]
// v1.0.0, in which case [GenerateKey], [NewPrivateKey], [NewPublicKey], and
// [Verify] will return an error. It is available if using v1.26.0 or later.
//
// [FIPS 204]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.204.pdf
// [FIPS 140-3 Go Cryptographic Module]: https://go.dev/doc/security/fips140
package mldsa

import "crypto"

const (
	PrivateKeySize = 32

	MLDSA44PublicKeySize = 1312
	MLDSA65PublicKeySize = 1952
	MLDSA87PublicKeySize = 2592

	MLDSA44SignatureSize = 2420
	MLDSA65SignatureSize = 3309
	MLDSA87SignatureSize = 4627
)

// Parameters represents one of the fixed parameter sets defined in FIPS 204.
//
// Most applications should use [MLDSA44].
//
// Multiple invocations of [MLDSA44], [MLDSA65], or [MLDSA87] will return the
// same respective value, which can be used for equality checks and switch
// statements. The returned value is safe for concurrent use.
type Parameters struct {
	name          string
	publicKeySize int
	signatureSize int
}

// MLDSA44 returns the ML-DSA-44 parameter set defined in FIPS 204.
func MLDSA44() Parameters {
	return Parameters{
		name:          "ML-DSA-44",
		publicKeySize: MLDSA44PublicKeySize,
		signatureSize: MLDSA44SignatureSize,
	}
}

// MLDSA65 returns the ML-DSA-65 parameter set defined in FIPS 204.
func MLDSA65() Parameters {
	return Parameters{
		name:          "ML-DSA-65",
		publicKeySize: MLDSA65PublicKeySize,
		signatureSize: MLDSA65SignatureSize,
	}
}

// MLDSA87 returns the ML-DSA-87 parameter set defined in FIPS 204.
func MLDSA87() Parameters {
	return Parameters{
		name:          "ML-DSA-87",
		publicKeySize: MLDSA87PublicKeySize,
		signatureSize: MLDSA87SignatureSize,
	}
}

// PublicKeySize returns the size of public keys for this parameter set, in bytes.
func (params Parameters) PublicKeySize() int {
	return params.publicKeySize
}

// SignatureSize returns the size of signatures for this parameter set, in bytes.
func (params Parameters) SignatureSize() int {
	return params.signatureSize
}

// String returns the name of the parameter set, e.g. "ML-DSA-44".
func (params Parameters) String() string {
	return params.name
}

// Options contains additional options for signing and verifying ML-DSA signatures.
type Options struct {
	// Context can be used to distinguish signatures created for different
	// purposes. It must be at most 255 bytes long, and it is empty by default.
	//
	// The same context must be used when signing and verifying a signature.
	Context string
}

// HashFunc returns zero, to implement the [crypto.SignerOpts] interface.
func (opts *Options) HashFunc() crypto.Hash {
	return 0
}

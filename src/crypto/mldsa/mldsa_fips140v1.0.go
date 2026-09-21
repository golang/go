// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build fips140v1.0

package mldsa

import (
	"crypto"
	"errors"
	"io"
)

// This file provides stub implementations of the ML-DSA API for building
// against the FIPS 140-3 Go Cryptographic Module v1.0.0, which does not include
// ML-DSA. Top-level functions return an error, and methods are unreachable
// since there is no way to construct a valid PublicKey or PrivateKey.

var errUnavailable = errors.New("mldsa: unavailable in FIPS 140-3 Go Cryptographic Module v1.0.0")

// PrivateKey is an in-memory ML-DSA private key. It implements [crypto.Signer]
// and the informal extended [crypto.PrivateKey] interface.
//
// A PrivateKey is safe for concurrent use.
type PrivateKey struct{}

// GenerateKey generates a new random ML-DSA private key.
func GenerateKey(params Parameters) (*PrivateKey, error) {
	return nil, errUnavailable
}

// NewPrivateKey decodes an ML-DSA private key from the given seed.
//
// The seed must be exactly [PrivateKeySize] bytes long.
func NewPrivateKey(params Parameters, seed []byte) (*PrivateKey, error) {
	return nil, errUnavailable
}

// Public returns the corresponding [PublicKey] for this private key.
//
// It implements the [crypto.Signer] interface.
func (sk *PrivateKey) Public() crypto.PublicKey {
	panic("mldsa: methods are unreachable in FIPS 140-3 Go Cryptographic Module v1.0.0")
}

// Equal reports whether sk and x are the same key (i.e. they are derived from
// the same seed).
//
// If x is not a *PrivateKey, Equal returns false.
func (sk *PrivateKey) Equal(x crypto.PrivateKey) bool {
	panic("mldsa: methods are unreachable in FIPS 140-3 Go Cryptographic Module v1.0.0")
}

// PublicKey returns the corresponding [PublicKey] for this private key.
func (sk *PrivateKey) PublicKey() *PublicKey {
	panic("mldsa: methods are unreachable in FIPS 140-3 Go Cryptographic Module v1.0.0")
}

// Bytes returns the private key seed.
func (sk *PrivateKey) Bytes() []byte {
	panic("mldsa: methods are unreachable in FIPS 140-3 Go Cryptographic Module v1.0.0")
}

// Sign returns a signature of the given message using this private key.
//
// If opts is nil or opts.HashFunc returns zero, the message is signed directly.
// If opts.HashFunc returns [crypto.MLDSAMu], the provided message must be a
// [pre-hashed μ message representative]. opts can be of type *[Options].
// The io.Reader argument is ignored.
//
// [pre-hashed μ message representative]: https://www.rfc-editor.org/rfc/rfc9881.html#externalmu
func (sk *PrivateKey) Sign(_ io.Reader, message []byte, opts crypto.SignerOpts) (signature []byte, err error) {
	panic("mldsa: methods are unreachable in FIPS 140-3 Go Cryptographic Module v1.0.0")
}

// SignDeterministic works like [PrivateKey.Sign], but the signature is
// deterministic.
func (sk *PrivateKey) SignDeterministic(message []byte, opts crypto.SignerOpts) (signature []byte, err error) {
	panic("mldsa: methods are unreachable in FIPS 140-3 Go Cryptographic Module v1.0.0")
}

// PublicKey is an ML-DSA public key. It implements the informal extended
// [crypto.PublicKey] interface.
//
// A PublicKey is safe for concurrent use.
type PublicKey struct{}

// NewPublicKey creates a new ML-DSA public key from the given encoding.
func NewPublicKey(params Parameters, encoding []byte) (*PublicKey, error) {
	return nil, errUnavailable
}

// Bytes returns the public key encoding.
func (pk *PublicKey) Bytes() []byte {
	panic("mldsa: methods are unreachable in FIPS 140-3 Go Cryptographic Module v1.0.0")
}

// Equal reports whether pk and x are the same key (i.e. they have the same
// encoding).
//
// If x is not a *PublicKey, Equal returns false.
func (pk *PublicKey) Equal(x crypto.PublicKey) bool {
	panic("mldsa: methods are unreachable in FIPS 140-3 Go Cryptographic Module v1.0.0")
}

// Parameters returns the parameters associated with this public key.
func (pk *PublicKey) Parameters() Parameters {
	panic("mldsa: methods are unreachable in FIPS 140-3 Go Cryptographic Module v1.0.0")
}

// Verify reports whether signature is a valid signature of message by pk.
// If opts is nil, it's equivalent to the zero value of Options.
func Verify(pk *PublicKey, message []byte, signature []byte, opts *Options) error {
	return errUnavailable
}

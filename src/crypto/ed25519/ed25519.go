// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ed25519 implements the Ed25519 signature algorithm. See
// https://ed25519.cr.yp.to/.
//
// These functions are also compatible with the “Ed25519” function defined in
// RFC 8032. However, unlike RFC 8032's formulation, this package's private key
// representation includes a public key suffix to make multiple signing
// operations with the same key more efficient. This package refers to the RFC
// 8032 private key as the “seed”.
//
// Operations involving private keys are implemented using constant-time
// algorithms.
package ed25519

import (
	"crypto"
	"crypto/internal/fips140/ed25519"
	"crypto/internal/fips140cache"
	"crypto/internal/fips140only"
	"crypto/internal/rand"
	cryptorand "crypto/rand"
	"crypto/subtle"
	"errors"
	"internal/godebug"
	"io"
	"strconv"
)

const (
	// PublicKeySize is the size, in bytes, of public keys as used in this package.
	PublicKeySize = 32
	// PrivateKeySize is the size, in bytes, of private keys as used in this package.
	PrivateKeySize = 64
	// SignatureSize is the size, in bytes, of signatures generated and verified by this package.
	SignatureSize = 64
	// SeedSize is the size, in bytes, of private key seeds. These are the private key representations used by RFC 8032.
	SeedSize = 32
)

// PublicKey is the type of Ed25519 public keys.
type PublicKey []byte

// Any methods implemented on PublicKey might need to also be implemented on
// PrivateKey, as the latter embeds the former and will expose its methods.

// Equal reports whether pub and x have the same value.
func (pub PublicKey) Equal(x crypto.PublicKey) bool {
	xx, ok := x.(PublicKey)
	if !ok {
		return false
	}
	return subtle.ConstantTimeCompare(pub, xx) == 1
}

// PrivateKey is the type of Ed25519 private keys. It implements [crypto.Signer].
type PrivateKey []byte

// Public returns the [PublicKey] corresponding to priv.
func (priv PrivateKey) Public() crypto.PublicKey {
	publicKey := make([]byte, PublicKeySize)
	copy(publicKey, priv[32:])
	return PublicKey(publicKey)
}

// Equal reports whether priv and x have the same value.
func (priv PrivateKey) Equal(x crypto.PrivateKey) bool {
	xx, ok := x.(PrivateKey)
	if !ok {
		return false
	}
	return subtle.ConstantTimeCompare(priv, xx) == 1
}

// Seed returns the private key seed corresponding to priv. It is provided for
// interoperability with RFC 8032. RFC 8032's private keys correspond to seeds
// in this package.
func (priv PrivateKey) Seed() []byte {
	return append(make([]byte, 0, SeedSize), priv[:SeedSize]...)
}

// privateKeyCache uses a pointer to the first byte of underlying storage as a
// key, because [PrivateKey] is a slice header passed around by value.
var privateKeyCache fips140cache.Cache[byte, ed25519.PrivateKey]

// Sign signs the given message with priv. rand is ignored and can be nil.
//
// If opts.HashFunc() is [crypto.SHA512], the pre-hashed variant Ed25519ph is used
// and message is expected to be a SHA-512 hash, otherwise opts.HashFunc() must
// be [crypto.Hash](0) and the message must not be hashed, as Ed25519 performs two
// passes over messages to be signed.
//
// A value of type [Options] can be used as opts, or crypto.Hash(0) or
// crypto.SHA512 directly to select plain Ed25519 or Ed25519ph, respectively.
func (priv PrivateKey) Sign(rand io.Reader, message []byte, opts crypto.SignerOpts) (signature []byte, err error) {
	k, err := privateKeyCache.Get(&priv[0], func() (*ed25519.PrivateKey, error) {
		return ed25519.NewPrivateKey(priv)
	}, func(k *ed25519.PrivateKey) bool {
		return subtle.ConstantTimeCompare(priv, k.Bytes()) == 1
	})
	if err != nil {
		return nil, err
	}
	hash := opts.HashFunc()
	context := ""
	if opts, ok := opts.(*Options); ok {
		context = opts.Context
	}
	switch {
	case hash == crypto.SHA512: // Ed25519ph
		return ed25519.SignPH(k, message, context)
	case hash == crypto.Hash(0) && context != "": // Ed25519ctx
		if fips140only.Enforced() {
			return nil, errors.New("crypto/ed25519: use of Ed25519ctx is not allowed in FIPS 140-only mode")
		}
		return ed25519.SignCtx(k, message, context)
	case hash == crypto.Hash(0): // Ed25519
		return ed25519.Sign(k, message), nil
	default:
		return nil, errors.New("ed25519: expected opts.HashFunc() zero (unhashed message, for standard Ed25519) or SHA-512 (for Ed25519ph)")
	}
}

// Options can be used with [PrivateKey.Sign] or [VerifyWithOptions]
// to select Ed25519 variants.
type Options struct {
	// Hash can be zero for regular Ed25519, or crypto.SHA512 for Ed25519ph.
	Hash crypto.Hash

	// Context, if not empty, selects Ed25519ctx or provides the context string
	// for Ed25519ph. It can be at most 255 bytes in length.
	Context string
}

// HashFunc returns o.Hash.
func (o *Options) HashFunc() crypto.Hash { return o.Hash }

var cryptocustomrand = godebug.New("cryptocustomrand")

// GenerateKey generates a public/private key pair using entropy from random.
//
// If random is nil, a secure random source is used. (Before Go 1.26, a custom
// [crypto/rand.Reader] was used if set by the application. That behavior can be
// restored with GODEBUG=cryptocustomrand=1. This setting will be removed in a
// future Go release. Instead, use [testing/cryptotest.SetGlobalRandom].)
//
// The output of this function is deterministic, and equivalent to reading
// [SeedSize] bytes from random, and passing them to [NewKeyFromSeed].
func GenerateKey(random io.Reader) (PublicKey, PrivateKey, error) {
	if random == nil {
		if cryptocustomrand.Value() == "1" {
			random = cryptorand.Reader
			if !rand.IsDefaultReader(random) {
				cryptocustomrand.IncNonDefault()
			}
		} else {
			random = rand.Reader
		}
	}

	seed := make([]byte, SeedSize)
	if _, err := io.ReadFull(random, seed); err != nil {
		return nil, nil, err
	}

	privateKey := NewKeyFromSeed(seed)
	publicKey := privateKey.Public().(PublicKey)
	return publicKey, privateKey, nil
}

// NewKeyFromSeed calculates a private key from a seed. It will panic if
// len(seed) is not [SeedSize]. This function is provided for interoperability
// with RFC 8032. RFC 8032's private keys correspond to seeds in this
// package.
func NewKeyFromSeed(seed []byte) PrivateKey {
	// Outline the function body so that the returned key can be stack-allocated.
	privateKey := make([]byte, PrivateKeySize)
	newKeyFromSeed(privateKey, seed)
	return privateKey
}

func newKeyFromSeed(privateKey, seed []byte) {
	k, err := ed25519.NewPrivateKeyFromSeed(seed)
	if err != nil {
		// NewPrivateKeyFromSeed only returns an error if the seed length is incorrect.
		panic("ed25519: bad seed length: " + strconv.Itoa(len(seed)))
	}
	copy(privateKey, k.Bytes())
}

// Sign signs the message with privateKey and returns a signature. It will
// panic if len(privateKey) is not [PrivateKeySize].
func Sign(privateKey PrivateKey, message []byte) []byte {
	// Outline the function body so that the returned signature can be
	// stack-allocated.
	signature := make([]byte, SignatureSize)
	sign(signature, privateKey, message)
	return signature
}

func sign(signature []byte, privateKey PrivateKey, message []byte) {
	k, err := privateKeyCache.Get(&privateKey[0], func() (*ed25519.PrivateKey, error) {
		return ed25519.NewPrivateKey(privateKey)
	}, func(k *ed25519.PrivateKey) bool {
		return subtle.ConstantTimeCompare(privateKey, k.Bytes()) == 1
	})
	if err != nil {
		panic("ed25519: bad private key: " + err.Error())
	}
	sig := ed25519.Sign(k, message)
	copy(signature, sig)
}

// Verify reports whether sig is a valid signature of message by publicKey. It
// will panic if len(publicKey) is not [PublicKeySize].
//
// The inputs are not considered confidential, and may leak through timing side
// channels, or if an attacker has control of part of the inputs.
func Verify(publicKey PublicKey, message, sig []byte) bool {
	return VerifyWithOptions(publicKey, message, sig, &Options{Hash: crypto.Hash(0)}) == nil
}

// VerifyWithOptions reports whether sig is a valid signature of message by
// publicKey. A valid signature is indicated by returning a nil error. It will
// panic if len(publicKey) is not [PublicKeySize].
//
// If opts.Hash is [crypto.SHA512], the pre-hashed variant Ed25519ph is used and
// message is expected to be a SHA-512 hash, otherwise opts.Hash must be
// [crypto.Hash](0) and the message must not be hashed, as Ed25519 performs two
// passes over messages to be signed.
//
// The inputs are not considered confidential, and may leak through timing side
// channels, or if an attacker has control of part of the inputs.
func VerifyWithOptions(publicKey PublicKey, message, sig []byte, opts *Options) error {
	if l := len(publicKey); l != PublicKeySize {
		panic("ed25519: bad public key length: " + strconv.Itoa(l))
	}
	k, err := ed25519.NewPublicKey(publicKey)
	if err != nil {
		return err
	}
	switch {
	case opts.Hash == crypto.SHA512: // Ed25519ph
		return ed25519.VerifyPH(k, message, sig, opts.Context)
	case opts.Hash == crypto.Hash(0) && opts.Context != "": // Ed25519ctx
		if fips140only.Enforced() {
			return errors.New("crypto/ed25519: use of Ed25519ctx is not allowed in FIPS 140-only mode")
		}
		return ed25519.VerifyCtx(k, message, sig, opts.Context)
	case opts.Hash == crypto.Hash(0): // Ed25519
		return ed25519.Verify(k, message, sig)
	default:
		return errors.New("ed25519: expected opts.Hash zero (unhashed message, for standard Ed25519) or SHA-512 (for Ed25519ph)")
	}
}

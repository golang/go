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
	"bytes"
	"crypto"
	"crypto/internal/fips/edwards25519"
	cryptorand "crypto/rand"
	"crypto/sha512"
	"crypto/subtle"
	"errors"
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
	return bytes.Clone(priv[:SeedSize])
}

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
	hash := opts.HashFunc()
	context := ""
	if opts, ok := opts.(*Options); ok {
		context = opts.Context
	}
	switch {
	case hash == crypto.SHA512: // Ed25519ph
		if l := len(message); l != sha512.Size {
			return nil, errors.New("ed25519: bad Ed25519ph message hash length: " + strconv.Itoa(l))
		}
		if l := len(context); l > 255 {
			return nil, errors.New("ed25519: bad Ed25519ph context length: " + strconv.Itoa(l))
		}
		signature := make([]byte, SignatureSize)
		sign(signature, priv, message, domPrefixPh, context)
		return signature, nil
	case hash == crypto.Hash(0) && context != "": // Ed25519ctx
		if l := len(context); l > 255 {
			return nil, errors.New("ed25519: bad Ed25519ctx context length: " + strconv.Itoa(l))
		}
		signature := make([]byte, SignatureSize)
		sign(signature, priv, message, domPrefixCtx, context)
		return signature, nil
	case hash == crypto.Hash(0): // Ed25519
		return Sign(priv, message), nil
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

// GenerateKey generates a public/private key pair using entropy from rand.
// If rand is nil, [crypto/rand.Reader] will be used.
//
// The output of this function is deterministic, and equivalent to reading
// [SeedSize] bytes from rand, and passing them to [NewKeyFromSeed].
func GenerateKey(rand io.Reader) (PublicKey, PrivateKey, error) {
	if rand == nil {
		rand = cryptorand.Reader
	}

	seed := make([]byte, SeedSize)
	if _, err := io.ReadFull(rand, seed); err != nil {
		return nil, nil, err
	}

	privateKey := NewKeyFromSeed(seed)
	publicKey := make([]byte, PublicKeySize)
	copy(publicKey, privateKey[32:])

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
	if l := len(seed); l != SeedSize {
		panic("ed25519: bad seed length: " + strconv.Itoa(l))
	}

	h := sha512.Sum512(seed)
	s, err := edwards25519.NewScalar().SetBytesWithClamping(h[:32])
	if err != nil {
		panic("ed25519: internal error: setting scalar failed")
	}
	A := (&edwards25519.Point{}).ScalarBaseMult(s)

	publicKey := A.Bytes()

	copy(privateKey, seed)
	copy(privateKey[32:], publicKey)
}

// Sign signs the message with privateKey and returns a signature. It will
// panic if len(privateKey) is not [PrivateKeySize].
func Sign(privateKey PrivateKey, message []byte) []byte {
	// Outline the function body so that the returned signature can be
	// stack-allocated.
	signature := make([]byte, SignatureSize)
	sign(signature, privateKey, message, domPrefixPure, "")
	return signature
}

// Domain separation prefixes used to disambiguate Ed25519/Ed25519ph/Ed25519ctx.
// See RFC 8032, Section 2 and Section 5.1.
const (
	// domPrefixPure is empty for pure Ed25519.
	domPrefixPure = ""
	// domPrefixPh is dom2(phflag=1) for Ed25519ph. It must be followed by the
	// uint8-length prefixed context.
	domPrefixPh = "SigEd25519 no Ed25519 collisions\x01"
	// domPrefixCtx is dom2(phflag=0) for Ed25519ctx. It must be followed by the
	// uint8-length prefixed context.
	domPrefixCtx = "SigEd25519 no Ed25519 collisions\x00"
)

func sign(signature, privateKey, message []byte, domPrefix, context string) {
	if l := len(privateKey); l != PrivateKeySize {
		panic("ed25519: bad private key length: " + strconv.Itoa(l))
	}
	seed, publicKey := privateKey[:SeedSize], privateKey[SeedSize:]

	h := sha512.Sum512(seed)
	s, err := edwards25519.NewScalar().SetBytesWithClamping(h[:32])
	if err != nil {
		panic("ed25519: internal error: setting scalar failed")
	}
	prefix := h[32:]

	mh := sha512.New()
	if domPrefix != domPrefixPure {
		mh.Write([]byte(domPrefix))
		mh.Write([]byte{byte(len(context))})
		mh.Write([]byte(context))
	}
	mh.Write(prefix)
	mh.Write(message)
	messageDigest := make([]byte, 0, sha512.Size)
	messageDigest = mh.Sum(messageDigest)
	r, err := edwards25519.NewScalar().SetUniformBytes(messageDigest)
	if err != nil {
		panic("ed25519: internal error: setting scalar failed")
	}

	R := (&edwards25519.Point{}).ScalarBaseMult(r)

	kh := sha512.New()
	if domPrefix != domPrefixPure {
		kh.Write([]byte(domPrefix))
		kh.Write([]byte{byte(len(context))})
		kh.Write([]byte(context))
	}
	kh.Write(R.Bytes())
	kh.Write(publicKey)
	kh.Write(message)
	hramDigest := make([]byte, 0, sha512.Size)
	hramDigest = kh.Sum(hramDigest)
	k, err := edwards25519.NewScalar().SetUniformBytes(hramDigest)
	if err != nil {
		panic("ed25519: internal error: setting scalar failed")
	}

	S := edwards25519.NewScalar().MultiplyAdd(k, s, r)

	copy(signature[:32], R.Bytes())
	copy(signature[32:], S.Bytes())
}

// Verify reports whether sig is a valid signature of message by publicKey. It
// will panic if len(publicKey) is not [PublicKeySize].
//
// The inputs are not considered confidential, and may leak through timing side
// channels, or if an attacker has control of part of the inputs.
func Verify(publicKey PublicKey, message, sig []byte) bool {
	return verify(publicKey, message, sig, domPrefixPure, "")
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
	switch {
	case opts.Hash == crypto.SHA512: // Ed25519ph
		if l := len(message); l != sha512.Size {
			return errors.New("ed25519: bad Ed25519ph message hash length: " + strconv.Itoa(l))
		}
		if l := len(opts.Context); l > 255 {
			return errors.New("ed25519: bad Ed25519ph context length: " + strconv.Itoa(l))
		}
		if !verify(publicKey, message, sig, domPrefixPh, opts.Context) {
			return errors.New("ed25519: invalid signature")
		}
		return nil
	case opts.Hash == crypto.Hash(0) && opts.Context != "": // Ed25519ctx
		if l := len(opts.Context); l > 255 {
			return errors.New("ed25519: bad Ed25519ctx context length: " + strconv.Itoa(l))
		}
		if !verify(publicKey, message, sig, domPrefixCtx, opts.Context) {
			return errors.New("ed25519: invalid signature")
		}
		return nil
	case opts.Hash == crypto.Hash(0): // Ed25519
		if !verify(publicKey, message, sig, domPrefixPure, "") {
			return errors.New("ed25519: invalid signature")
		}
		return nil
	default:
		return errors.New("ed25519: expected opts.Hash zero (unhashed message, for standard Ed25519) or SHA-512 (for Ed25519ph)")
	}
}

func verify(publicKey PublicKey, message, sig []byte, domPrefix, context string) bool {
	if l := len(publicKey); l != PublicKeySize {
		panic("ed25519: bad public key length: " + strconv.Itoa(l))
	}

	if len(sig) != SignatureSize || sig[63]&224 != 0 {
		return false
	}

	A, err := (&edwards25519.Point{}).SetBytes(publicKey)
	if err != nil {
		return false
	}

	kh := sha512.New()
	if domPrefix != domPrefixPure {
		kh.Write([]byte(domPrefix))
		kh.Write([]byte{byte(len(context))})
		kh.Write([]byte(context))
	}
	kh.Write(sig[:32])
	kh.Write(publicKey)
	kh.Write(message)
	hramDigest := make([]byte, 0, sha512.Size)
	hramDigest = kh.Sum(hramDigest)
	k, err := edwards25519.NewScalar().SetUniformBytes(hramDigest)
	if err != nil {
		panic("ed25519: internal error: setting scalar failed")
	}

	S, err := edwards25519.NewScalar().SetCanonicalBytes(sig[32:])
	if err != nil {
		return false
	}

	// [S]B = R + [k]A --> [k](-A) + [S]B = R
	minusA := (&edwards25519.Point{}).Negate(A)
	R := (&edwards25519.Point{}).VarTimeDoubleScalarBaseMult(k, minusA, S)

	return bytes.Equal(sig[:32], R.Bytes())
}

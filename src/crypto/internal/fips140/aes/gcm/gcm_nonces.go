// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/alias"
	"crypto/internal/fips140/drbg"
	"crypto/internal/fips140deps/byteorder"
	"errors"
	"math"
)

// SealWithRandomNonce encrypts plaintext to out, and writes a random nonce to
// nonce. nonce must be 12 bytes, and out must be 16 bytes longer than plaintext.
// out and plaintext may overlap exactly or not at all. additionalData and out
// must not overlap.
//
// This complies with FIPS 140-3 IG C.H Scenario 2.
//
// Note that this is NOT a [cipher.AEAD].Seal method.
func SealWithRandomNonce(g *GCM, nonce, out, plaintext, additionalData []byte) {
	if uint64(len(plaintext)) > uint64((1<<32)-2)*gcmBlockSize {
		panic("crypto/cipher: message too large for GCM")
	}
	if len(nonce) != gcmStandardNonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCMWithRandomNonce")
	}
	if len(out) != len(plaintext)+gcmTagSize {
		panic("crypto/cipher: incorrect output length given to GCMWithRandomNonce")
	}
	if alias.InexactOverlap(out, plaintext) {
		panic("crypto/cipher: invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, additionalData) {
		panic("crypto/cipher: invalid buffer overlap of output and additional data")
	}
	fips140.RecordApproved()
	drbg.Read(nonce)
	seal(out, g, nonce, plaintext, additionalData)
}

// NewGCMWithCounterNonce returns a new AEAD that works like GCM, but enforces
// the construction of deterministic nonces. The nonce must be 96 bits, the
// first 32 bits must be an encoding of the module name, and the last 64 bits
// must be a counter. The starting value of the counter is set on the first call
// to Seal, and each subsequent call must increment it as a big-endian uint64.
// If the counter reaches the starting value minus one, Seal will panic.
//
// This complies with FIPS 140-3 IG C.H Scenario 3.
func NewGCMWithCounterNonce(cipher *aes.Block) (*GCMWithCounterNonce, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	return &GCMWithCounterNonce{g: *g}, nil
}

// NewGCMForTLS12 returns a new AEAD that works like GCM, but enforces the
// construction of nonces as specified in RFC 5288, Section 3 and RFC 9325,
// Section 7.2.1.
//
// This complies with FIPS 140-3 IG C.H Scenario 1.a.
func NewGCMForTLS12(cipher *aes.Block) (*GCMWithCounterNonce, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	// TLS 1.2 counters always start at zero.
	return &GCMWithCounterNonce{g: *g, startReady: true}, nil
}

// NewGCMForSSH returns a new AEAD that works like GCM, but enforces the
// construction of nonces as specified in RFC 5647.
//
// This complies with FIPS 140-3 IG C.H Scenario 1.d.
func NewGCMForSSH(cipher *aes.Block) (*GCMWithCounterNonce, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	return &GCMWithCounterNonce{g: *g}, nil
}

type GCMWithCounterNonce struct {
	g           GCM
	prefixReady bool
	prefix      uint32
	startReady  bool
	start       uint64
	next        uint64
}

func (g *GCMWithCounterNonce) NonceSize() int { return gcmStandardNonceSize }

func (g *GCMWithCounterNonce) Overhead() int { return gcmTagSize }

// Seal implements the [cipher.AEAD] interface, checking that the nonce prefix
// is stable and that the counter is strictly increasing.
//
// It is not safe for concurrent use.
func (g *GCMWithCounterNonce) Seal(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != gcmStandardNonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}

	if !g.prefixReady {
		// The first invocation sets the fixed prefix.
		g.prefixReady = true
		g.prefix = byteorder.BEUint32(nonce[:4])
	}
	if g.prefix != byteorder.BEUint32(nonce[:4]) {
		panic("crypto/cipher: GCM nonce prefix changed")
	}

	counter := byteorder.BEUint64(nonce[len(nonce)-8:])
	if !g.startReady {
		// The first invocation sets the starting counter, if not fixed.
		g.startReady = true
		g.start = counter
	}
	counter -= g.start

	// Ensure the counter is strictly increasing.
	if counter == math.MaxUint64 {
		panic("crypto/cipher: counter exhausted")
	}
	if counter < g.next {
		panic("crypto/cipher: counter decreased or remained the same")
	}
	g.next = counter + 1

	fips140.RecordApproved()
	return g.g.sealAfterIndicator(dst, nonce, plaintext, data)
}

func (g *GCMWithCounterNonce) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	fips140.RecordApproved()
	return g.g.Open(dst, nonce, ciphertext, data)
}

// NewGCMWithXORCounterNonce returns a new AEAD that works like GCM, but
// enforces the construction of deterministic nonces. The nonce must be 96 bits,
// the first 32 bits must be an encoding of the module name, and the last 64
// bits must be a counter XOR'd with a fixed value. The module name and XOR mask
// can be set with [GCMWithCounterNonce.SetNoncePrefixAndMask], or they are set
// on the first call to Seal, assuming the counter starts at zero. Each
// subsequent call must increment the counter as a big-endian uint64. If the
// counter reaches 2⁶⁴ minus one, Seal will panic.
//
// This complies with FIPS 140-3 IG C.H Scenario 3.
func NewGCMWithXORCounterNonce(cipher *aes.Block) (*GCMWithXORCounterNonce, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	return &GCMWithXORCounterNonce{g: *g}, nil
}

// NewGCMForTLS13 returns a new AEAD that works like GCM, but enforces the
// construction of nonces as specified in RFC 8446, Section 5.3.
//
// This complies with FIPS 140-3 IG C.H Scenario 1.a.
func NewGCMForTLS13(cipher *aes.Block) (*GCMWithXORCounterNonce, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	return &GCMWithXORCounterNonce{g: *g}, nil
}

// NewGCMForHPKE returns a new AEAD that works like GCM, but enforces the
// construction of nonces as specified in RFC 9180, Section 5.2.
//
// This complies with FIPS 140-3 IG C.H Scenario 5.
func NewGCMForHPKE(cipher *aes.Block) (*GCMWithXORCounterNonce, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	return &GCMWithXORCounterNonce{g: *g}, nil
}

// NewGCMForQUIC returns a new AEAD that works like GCM, but enforces the
// construction of nonces as specified in RFC 9001, Section 5.3.
//
// Unlike in TLS 1.3, the QUIC nonce counter does not always start at zero, as
// the packet number does not reset on key updates, so the XOR mask must be
// provided explicitly instead of being learned on the first Seal call. Note
// that the nonce passed to Seal must already be XOR'd with the IV, the IV is
// provided here only to allow Seal to enforce that the counter is strictly
// increasing.
//
// This complies with FIPS 140-3 IG C.H Scenario 5.
func NewGCMForQUIC(cipher *aes.Block, iv []byte) (*GCMWithXORCounterNonce, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	gcm := &GCMWithXORCounterNonce{g: *g}
	if err := gcm.SetNoncePrefixAndMask(iv); err != nil {
		return nil, err
	}
	return gcm, nil
}

type GCMWithXORCounterNonce struct {
	g      GCM
	ready  bool
	prefix uint32
	mask   uint64
	next   uint64
}

// SetNoncePrefixAndMask sets the fixed prefix and XOR mask for the nonces used
// in Seal. It must be called before the first call to Seal.
//
// The first 32 bits of nonce are used as the fixed prefix, and the last 64 bits
// are used as the XOR mask.
//
// Note that Seal expects the nonce to be already XOR'd with the mask. The mask
// is provided here only to allow Seal to enforce that the counter is strictly
// increasing.
func (g *GCMWithXORCounterNonce) SetNoncePrefixAndMask(nonce []byte) error {
	if len(nonce) != gcmStandardNonceSize {
		return errors.New("crypto/cipher: incorrect nonce length given to SetNoncePrefixAndMask")
	}
	if g.ready {
		return errors.New("crypto/cipher: SetNoncePrefixAndMask called twice or after first Seal")
	}
	g.prefix = byteorder.BEUint32(nonce[:4])
	g.mask = byteorder.BEUint64(nonce[4:])
	g.ready = true
	return nil
}

func (g *GCMWithXORCounterNonce) NonceSize() int { return gcmStandardNonceSize }

func (g *GCMWithXORCounterNonce) Overhead() int { return gcmTagSize }

// Seal implements the [cipher.AEAD] interface, checking that the nonce prefix
// is stable and that the counter is strictly increasing.
//
// It is not safe for concurrent use.
func (g *GCMWithXORCounterNonce) Seal(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != gcmStandardNonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}

	counter := byteorder.BEUint64(nonce[len(nonce)-8:])
	if !g.ready {
		// In the first call, if [GCMWithXORCounterNonce.SetNoncePrefixAndMask]
		// wasn't used, we assume the counter is zero to learn the XOR mask and
		// fixed prefix.
		g.ready = true
		g.mask = counter
		g.prefix = byteorder.BEUint32(nonce[:4])
	}
	if g.prefix != byteorder.BEUint32(nonce[:4]) {
		panic("crypto/cipher: GCM nonce prefix changed")
	}
	counter ^= g.mask

	// Ensure the counter is strictly increasing.
	if counter == math.MaxUint64 {
		panic("crypto/cipher: counter exhausted")
	}
	if counter < g.next {
		panic("crypto/cipher: counter decreased or remained the same")
	}
	g.next = counter + 1

	fips140.RecordApproved()
	return g.g.sealAfterIndicator(dst, nonce, plaintext, data)
}

func (g *GCMWithXORCounterNonce) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	fips140.RecordApproved()
	return g.g.Open(dst, nonce, ciphertext, data)
}

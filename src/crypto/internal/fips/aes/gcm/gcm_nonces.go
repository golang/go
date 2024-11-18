// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm

import (
	"crypto/internal/fips"
	"crypto/internal/fips/aes"
	"crypto/internal/fips/alias"
	"crypto/internal/fips/drbg"
	"internal/byteorder"
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
	fips.RecordApproved()
	drbg.Read(nonce)
	seal(out, g, nonce, plaintext, additionalData)
}

// NewGCMWithCounterNonce returns a new AEAD that works like GCM, but enforces
// the construction of deterministic nonces. The nonce must be 96 bits, the
// first 32 bits must be an encoding of the module name, and the last 64 bits
// must be a counter.
//
// This complies with FIPS 140-3 IG C.H Scenario 3.
func NewGCMWithCounterNonce(cipher *aes.Block) (*GCMWithCounterNonce, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	return &GCMWithCounterNonce{g: *g}, nil
}

type GCMWithCounterNonce struct {
	g         GCM
	ready     bool
	fixedName uint32
	start     uint64
	next      uint64
}

func (g *GCMWithCounterNonce) NonceSize() int { return gcmStandardNonceSize }

func (g *GCMWithCounterNonce) Overhead() int { return gcmTagSize }

func (g *GCMWithCounterNonce) Seal(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != gcmStandardNonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}

	counter := byteorder.BeUint64(nonce[len(nonce)-8:])
	if !g.ready {
		// The first invocation sets the fixed name encoding and start counter.
		g.ready = true
		g.start = counter
		g.fixedName = byteorder.BeUint32(nonce[:4])
	}
	if g.fixedName != byteorder.BeUint32(nonce[:4]) {
		panic("crypto/cipher: incorrect module name given to GCMWithCounterNonce")
	}
	counter -= g.start

	// Ensure the counter is monotonically increasing.
	if counter == math.MaxUint64 {
		panic("crypto/cipher: counter wrapped")
	}
	if counter < g.next {
		panic("crypto/cipher: counter decreased")
	}
	g.next = counter + 1

	fips.RecordApproved()
	return g.g.sealAfterIndicator(dst, nonce, plaintext, data)
}

func (g *GCMWithCounterNonce) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	fips.RecordApproved()
	return g.g.Open(dst, nonce, ciphertext, data)
}

// NewGCMForTLS12 returns a new AEAD that works like GCM, but enforces the
// construction of nonces as specified in RFC 5288, Section 3 and RFC 9325,
// Section 7.2.1.
//
// This complies with FIPS 140-3 IG C.H Scenario 1.a.
func NewGCMForTLS12(cipher *aes.Block) (*GCMForTLS12, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	return &GCMForTLS12{g: *g}, nil
}

type GCMForTLS12 struct {
	g    GCM
	next uint64
}

func (g *GCMForTLS12) NonceSize() int { return gcmStandardNonceSize }

func (g *GCMForTLS12) Overhead() int { return gcmTagSize }

func (g *GCMForTLS12) Seal(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != gcmStandardNonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}

	counter := byteorder.BeUint64(nonce[len(nonce)-8:])

	// Ensure the counter is monotonically increasing.
	if counter == math.MaxUint64 {
		panic("crypto/cipher: counter wrapped")
	}
	if counter < g.next {
		panic("crypto/cipher: counter decreased")
	}
	g.next = counter + 1

	fips.RecordApproved()
	return g.g.sealAfterIndicator(dst, nonce, plaintext, data)
}

func (g *GCMForTLS12) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	fips.RecordApproved()
	return g.g.Open(dst, nonce, ciphertext, data)
}

// NewGCMForTLS13 returns a new AEAD that works like GCM, but enforces the
// construction of nonces as specified in RFC 8446, Section 5.3.
func NewGCMForTLS13(cipher *aes.Block) (*GCMForTLS13, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	return &GCMForTLS13{g: *g}, nil
}

type GCMForTLS13 struct {
	g     GCM
	ready bool
	mask  uint64
	next  uint64
}

func (g *GCMForTLS13) NonceSize() int { return gcmStandardNonceSize }

func (g *GCMForTLS13) Overhead() int { return gcmTagSize }

func (g *GCMForTLS13) Seal(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != gcmStandardNonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}

	counter := byteorder.BeUint64(nonce[len(nonce)-8:])
	if !g.ready {
		// In the first call, the counter is zero, so we learn the XOR mask.
		g.ready = true
		g.mask = counter
	}
	counter ^= g.mask

	// Ensure the counter is monotonically increasing.
	if counter == math.MaxUint64 {
		panic("crypto/cipher: counter wrapped")
	}
	if counter < g.next {
		panic("crypto/cipher: counter decreased")
	}
	g.next = counter + 1

	fips.RecordApproved()
	return g.g.sealAfterIndicator(dst, nonce, plaintext, data)
}

func (g *GCMForTLS13) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	fips.RecordApproved()
	return g.g.Open(dst, nonce, ciphertext, data)
}

// NewGCMForSSH returns a new AEAD that works like GCM, but enforces the
// construction of nonces as specified in RFC 5647.
//
// This complies with FIPS 140-3 IG C.H Scenario 1.d.
func NewGCMForSSH(cipher *aes.Block) (*GCMForSSH, error) {
	g, err := newGCM(&GCM{}, cipher, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	return &GCMForSSH{g: *g}, nil
}

type GCMForSSH struct {
	g     GCM
	ready bool
	start uint64
	next  uint64
}

func (g *GCMForSSH) NonceSize() int { return gcmStandardNonceSize }

func (g *GCMForSSH) Overhead() int { return gcmTagSize }

func (g *GCMForSSH) Seal(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != gcmStandardNonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}

	counter := byteorder.BeUint64(nonce[len(nonce)-8:])
	if !g.ready {
		// In the first call we learn the start value.
		g.ready = true
		g.start = counter
	}
	counter -= g.start

	// Ensure the counter is monotonically increasing.
	if counter == math.MaxUint64 {
		panic("crypto/cipher: counter wrapped")
	}
	if counter < g.next {
		panic("crypto/cipher: counter decreased")
	}
	g.next = counter + 1

	fips.RecordApproved()
	return g.g.sealAfterIndicator(dst, nonce, plaintext, data)
}

func (g *GCMForSSH) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	fips.RecordApproved()
	return g.g.Open(dst, nonce, ciphertext, data)
}

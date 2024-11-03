// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (ppc64le || ppc64) && !purego

package gcm

import (
	"crypto/internal/fips/aes"
	"crypto/internal/fips/subtle"
	"crypto/internal/impl"
	"internal/byteorder"
	"internal/godebug"
	"runtime"
)

// This file implements GCM using an optimized GHASH function.

//go:noescape
func gcmInit(productTable *[256]byte, h []byte)

//go:noescape
func gcmHash(output []byte, productTable *[256]byte, inp []byte, len int)

func counterCryptASM(nr int, out, in []byte, counter *[gcmBlockSize]byte, key *uint32)

// The POWER architecture doesn't have a way to turn off AES-GCM support
// at runtime with GODEBUG=cpu.something=off, so introduce a new GODEBUG
// knob for that. It's intentionally only checked at init() time, to
// avoid the performance overhead of checking it every time.
var supportsAESGCM = godebug.New("#ppc64gcm").Value() == "off"

func init() {
	impl.Register("gcm", "POWER8", &supportsAESGCM)
}

func checkGenericIsExpected() {
	if supportsAESGCM {
		panic("gcm: internal error: using generic implementation despite hardware support")
	}
}

type gcmPlatformData struct {
	productTable [256]byte
}

func initGCM(g *GCM) {
	if !supportsAESGCM {
		return
	}

	hle := make([]byte, gcmBlockSize)
	g.cipher.Encrypt(hle, hle)

	// Reverse the bytes in each 8 byte chunk
	// Load little endian, store big endian
	var h1, h2 uint64
	if runtime.GOARCH == "ppc64le" {
		h1 = byteorder.LeUint64(hle[:8])
		h2 = byteorder.LeUint64(hle[8:])
	} else {
		h1 = byteorder.BeUint64(hle[:8])
		h2 = byteorder.BeUint64(hle[8:])
	}
	byteorder.BePutUint64(hle[:8], h1)
	byteorder.BePutUint64(hle[8:], h2)
	gcmInit(&g.productTable, hle)
}

// deriveCounter computes the initial GCM counter state from the given nonce.
func deriveCounter(counter *[gcmBlockSize]byte, nonce []byte, productTable *[256]byte) {
	if len(nonce) == gcmStandardNonceSize {
		copy(counter[:], nonce)
		counter[gcmBlockSize-1] = 1
	} else {
		var hash [16]byte
		paddedGHASH(&hash, nonce, productTable)
		lens := gcmLengths(0, uint64(len(nonce))*8)
		paddedGHASH(&hash, lens[:], productTable)
		copy(counter[:], hash[:])
	}
}

// counterCrypt encrypts in using AES in counter mode and places the result
// into out. counter is the initial count value and will be updated with the next
// count value. The length of out must be greater than or equal to the length
// of in.
// counterCryptASM implements counterCrypt which then allows the loop to
// be unrolled and optimized.
func counterCrypt(b *aes.Block, out, in []byte, counter *[gcmBlockSize]byte) {
	enc := aes.EncryptionKeySchedule(b)
	rounds := len(enc)/4 - 1
	counterCryptASM(rounds, out, in, counter, &enc[0])
}

// paddedGHASH pads data with zeroes until its length is a multiple of
// 16-bytes. It then calculates a new value for hash using the ghash
// algorithm.
func paddedGHASH(hash *[16]byte, data []byte, productTable *[256]byte) {
	if siz := len(data) - (len(data) % gcmBlockSize); siz > 0 {
		gcmHash(hash[:], productTable, data[:], siz)
		data = data[siz:]
	}
	if len(data) > 0 {
		var s [16]byte
		copy(s[:], data)
		gcmHash(hash[:], productTable, s[:], len(s))
	}
}

// auth calculates GHASH(ciphertext, additionalData), masks the result with
// tagMask and writes the result to out.
func auth(out, ciphertext, aad []byte, tagMask *[gcmTagSize]byte, productTable *[256]byte) {
	var hash [16]byte
	paddedGHASH(&hash, aad, productTable)
	paddedGHASH(&hash, ciphertext, productTable)
	lens := gcmLengths(uint64(len(aad))*8, uint64(len(ciphertext))*8)
	paddedGHASH(&hash, lens[:], productTable)

	copy(out, hash[:])
	for i := range out {
		out[i] ^= tagMask[i]
	}
}

func seal(out []byte, g *GCM, nonce, plaintext, data []byte) {
	if !supportsAESGCM {
		sealGeneric(out, g, nonce, plaintext, data)
		return
	}

	var counter, tagMask [gcmBlockSize]byte
	deriveCounter(&counter, nonce, &g.productTable)

	g.cipher.Encrypt(tagMask[:], counter[:])
	gcmInc32(&counter)

	counterCrypt(&g.cipher, out, plaintext, &counter)
	auth(out[len(plaintext):], out[:len(plaintext)], data, &tagMask, &g.productTable)
}

func open(out []byte, g *GCM, nonce, ciphertext, data []byte) error {
	if !supportsAESGCM {
		return openGeneric(out, g, nonce, ciphertext, data)
	}

	tag := ciphertext[len(ciphertext)-g.tagSize:]
	ciphertext = ciphertext[:len(ciphertext)-g.tagSize]

	var counter, tagMask [gcmBlockSize]byte
	deriveCounter(&counter, nonce, &g.productTable)

	g.cipher.Encrypt(tagMask[:], counter[:])
	gcmInc32(&counter)

	var expectedTag [gcmTagSize]byte
	auth(expectedTag[:], ciphertext, data, &tagMask, &g.productTable)

	if subtle.ConstantTimeCompare(expectedTag[:g.tagSize], tag) != 1 {
		return errOpen
	}

	counterCrypt(&g.cipher, out, ciphertext, &counter)
	return nil
}

func gcmLengths(len0, len1 uint64) [16]byte {
	return [16]byte{
		byte(len0 >> 56),
		byte(len0 >> 48),
		byte(len0 >> 40),
		byte(len0 >> 32),
		byte(len0 >> 24),
		byte(len0 >> 16),
		byte(len0 >> 8),
		byte(len0),
		byte(len1 >> 56),
		byte(len1 >> 48),
		byte(len1 >> 40),
		byte(len1 >> 32),
		byte(len1 >> 24),
		byte(len1 >> 16),
		byte(len1 >> 8),
		byte(len1),
	}
}

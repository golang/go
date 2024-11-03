// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm

import (
	"crypto/internal/fips/aes"
	"crypto/internal/fips/subtle"
	"internal/byteorder"
)

func sealGeneric(out []byte, g *GCM, nonce, plaintext, additionalData []byte) {
	var H, counter, tagMask [gcmBlockSize]byte
	g.cipher.Encrypt(H[:], H[:])
	deriveCounterGeneric(&H, &counter, nonce)
	gcmCounterCryptGeneric(&g.cipher, tagMask[:], tagMask[:], &counter)

	gcmCounterCryptGeneric(&g.cipher, out, plaintext, &counter)

	var tag [gcmTagSize]byte
	gcmAuthGeneric(tag[:], &H, &tagMask, out[:len(plaintext)], additionalData)
	copy(out[len(plaintext):], tag[:])
}

func openGeneric(out []byte, g *GCM, nonce, ciphertext, additionalData []byte) error {
	var H, counter, tagMask [gcmBlockSize]byte
	g.cipher.Encrypt(H[:], H[:])
	deriveCounterGeneric(&H, &counter, nonce)
	gcmCounterCryptGeneric(&g.cipher, tagMask[:], tagMask[:], &counter)

	tag := ciphertext[len(ciphertext)-g.tagSize:]
	ciphertext = ciphertext[:len(ciphertext)-g.tagSize]

	var expectedTag [gcmTagSize]byte
	gcmAuthGeneric(expectedTag[:], &H, &tagMask, ciphertext, additionalData)
	if subtle.ConstantTimeCompare(expectedTag[:g.tagSize], tag) != 1 {
		return errOpen
	}

	gcmCounterCryptGeneric(&g.cipher, out, ciphertext, &counter)

	return nil
}

// deriveCounterGeneric computes the initial GCM counter state from the given nonce.
// See NIST SP 800-38D, section 7.1. This assumes that counter is filled with
// zeros on entry.
func deriveCounterGeneric(H, counter *[gcmBlockSize]byte, nonce []byte) {
	// GCM has two modes of operation with respect to the initial counter
	// state: a "fast path" for 96-bit (12-byte) nonces, and a "slow path"
	// for nonces of other lengths. For a 96-bit nonce, the nonce, along
	// with a four-byte big-endian counter starting at one, is used
	// directly as the starting counter. For other nonce sizes, the counter
	// is computed by passing it through the GHASH function.
	if len(nonce) == gcmStandardNonceSize {
		copy(counter[:], nonce)
		counter[gcmBlockSize-1] = 1
	} else {
		lenBlock := make([]byte, 16)
		byteorder.BePutUint64(lenBlock[8:], uint64(len(nonce))*8)
		ghash(counter, H, nonce, lenBlock)
	}
}

// gcmCounterCryptGeneric encrypts src using AES in counter mode with 32-bit
// wrapping (which is different from AES-CTR) and places the result into out.
// counter is the initial value and will be updated with the next value.
func gcmCounterCryptGeneric(b *aes.Block, out, src []byte, counter *[gcmBlockSize]byte) {
	var mask [gcmBlockSize]byte

	for len(src) >= gcmBlockSize {
		b.Encrypt(mask[:], counter[:])
		gcmInc32(counter)

		subtle.XORBytes(out, src, mask[:])
		out = out[gcmBlockSize:]
		src = src[gcmBlockSize:]
	}

	if len(src) > 0 {
		b.Encrypt(mask[:], counter[:])
		gcmInc32(counter)
		subtle.XORBytes(out, src, mask[:])
	}
}

// gcmInc32 treats the final four bytes of counterBlock as a big-endian value
// and increments it.
func gcmInc32(counterBlock *[gcmBlockSize]byte) {
	ctr := counterBlock[len(counterBlock)-4:]
	byteorder.BePutUint32(ctr, byteorder.BeUint32(ctr)+1)
}

// gcmAuthGeneric calculates GHASH(additionalData, ciphertext), masks the result
// with tagMask and writes the result to out.
func gcmAuthGeneric(out []byte, H, tagMask *[gcmBlockSize]byte, ciphertext, additionalData []byte) {
	checkGenericIsExpected()
	lenBlock := make([]byte, 16)
	byteorder.BePutUint64(lenBlock[:8], uint64(len(additionalData))*8)
	byteorder.BePutUint64(lenBlock[8:], uint64(len(ciphertext))*8)
	var S [gcmBlockSize]byte
	ghash(&S, H, additionalData, ciphertext, lenBlock)
	subtle.XORBytes(out, S[:], tagMask[:])
}

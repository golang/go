// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher

import (
	"crypto/internal/fips/aes"
	"crypto/internal/fips/aes/gcm"
	"crypto/internal/fips/alias"
	"crypto/subtle"
	"errors"
	"internal/byteorder"
)

const (
	gcmBlockSize         = 16
	gcmStandardNonceSize = 12
	gcmTagSize           = 16
	gcmMinimumTagSize    = 12 // NIST SP 800-38D recommends tags with 12 or more bytes.
)

// NewGCM returns the given 128-bit, block cipher wrapped in Galois Counter Mode
// with the standard nonce length.
//
// In general, the GHASH operation performed by this implementation of GCM is not constant-time.
// An exception is when the underlying [Block] was created by aes.NewCipher
// on systems with hardware support for AES. See the [crypto/aes] package documentation for details.
func NewGCM(cipher Block) (AEAD, error) {
	return newGCM(cipher, gcmStandardNonceSize, gcmTagSize)
}

// NewGCMWithNonceSize returns the given 128-bit, block cipher wrapped in Galois
// Counter Mode, which accepts nonces of the given length. The length must not
// be zero.
//
// Only use this function if you require compatibility with an existing
// cryptosystem that uses non-standard nonce lengths. All other users should use
// [NewGCM], which is faster and more resistant to misuse.
func NewGCMWithNonceSize(cipher Block, size int) (AEAD, error) {
	return newGCM(cipher, size, gcmTagSize)
}

// NewGCMWithTagSize returns the given 128-bit, block cipher wrapped in Galois
// Counter Mode, which generates tags with the given length.
//
// Tag sizes between 12 and 16 bytes are allowed.
//
// Only use this function if you require compatibility with an existing
// cryptosystem that uses non-standard tag lengths. All other users should use
// [NewGCM], which is more resistant to misuse.
func NewGCMWithTagSize(cipher Block, tagSize int) (AEAD, error) {
	return newGCM(cipher, gcmStandardNonceSize, tagSize)
}

func newGCM(cipher Block, nonceSize, tagSize int) (AEAD, error) {
	c, ok := cipher.(*aes.Block)
	if !ok {
		return newGCMFallback(cipher, nonceSize, tagSize)
	}
	// We don't return gcm.New directly, because it would always return a non-nil
	// AEAD interface value with type *gcm.GCM even if the *gcm.GCM is nil.
	g, err := gcm.New(c, nonceSize, tagSize)
	if err != nil {
		return nil, err
	}
	return g, nil
}

// gcmAble is an interface implemented by ciphers that have a specific optimized
// implementation of GCM. crypto/aes doesn't use this anymore, and we'd like to
// eventually remove it.
type gcmAble interface {
	NewGCM(nonceSize, tagSize int) (AEAD, error)
}

func newGCMFallback(cipher Block, nonceSize, tagSize int) (AEAD, error) {
	if tagSize < gcmMinimumTagSize || tagSize > gcmBlockSize {
		return nil, errors.New("cipher: incorrect tag size given to GCM")
	}
	if nonceSize <= 0 {
		return nil, errors.New("cipher: the nonce can't have zero length")
	}
	if cipher, ok := cipher.(gcmAble); ok {
		return cipher.NewGCM(nonceSize, tagSize)
	}
	if cipher.BlockSize() != gcmBlockSize {
		return nil, errors.New("cipher: NewGCM requires 128-bit block cipher")
	}
	return &gcmFallback{cipher: cipher, nonceSize: nonceSize, tagSize: tagSize}, nil
}

// gcmFallback is only used for non-AES ciphers, which regrettably we
// theoretically support. It's a copy of the generic implementation from
// crypto/internal/fips/aes/gcm/gcm_generic.go, refer to that file for more details.
type gcmFallback struct {
	cipher    Block
	nonceSize int
	tagSize   int
}

func (g *gcmFallback) NonceSize() int {
	return g.nonceSize
}

func (g *gcmFallback) Overhead() int {
	return g.tagSize
}

func (g *gcmFallback) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	if len(nonce) != g.nonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}
	if g.nonceSize == 0 {
		panic("crypto/cipher: incorrect GCM nonce size")
	}
	if uint64(len(plaintext)) > uint64((1<<32)-2)*gcmBlockSize {
		panic("crypto/cipher: message too large for GCM")
	}

	ret, out := sliceForAppend(dst, len(plaintext)+g.tagSize)
	if alias.InexactOverlap(out, plaintext) {
		panic("crypto/cipher: invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, additionalData) {
		panic("crypto/cipher: invalid buffer overlap of output and additional data")
	}

	var H, counter, tagMask [gcmBlockSize]byte
	g.cipher.Encrypt(H[:], H[:])
	deriveCounter(&H, &counter, nonce)
	gcmCounterCryptGeneric(g.cipher, tagMask[:], tagMask[:], &counter)

	gcmCounterCryptGeneric(g.cipher, out, plaintext, &counter)

	var tag [gcmTagSize]byte
	gcmAuth(tag[:], &H, &tagMask, out[:len(plaintext)], additionalData)
	copy(out[len(plaintext):], tag[:])

	return ret
}

var errOpen = errors.New("cipher: message authentication failed")

func (g *gcmFallback) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	if len(nonce) != g.nonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}
	if g.tagSize < gcmMinimumTagSize {
		panic("crypto/cipher: incorrect GCM tag size")
	}

	if len(ciphertext) < g.tagSize {
		return nil, errOpen
	}
	if uint64(len(ciphertext)) > uint64((1<<32)-2)*gcmBlockSize+uint64(g.tagSize) {
		return nil, errOpen
	}

	ret, out := sliceForAppend(dst, len(ciphertext)-g.tagSize)
	if alias.InexactOverlap(out, ciphertext) {
		panic("crypto/cipher: invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, additionalData) {
		panic("crypto/cipher: invalid buffer overlap of output and additional data")
	}

	var H, counter, tagMask [gcmBlockSize]byte
	g.cipher.Encrypt(H[:], H[:])
	deriveCounter(&H, &counter, nonce)
	gcmCounterCryptGeneric(g.cipher, tagMask[:], tagMask[:], &counter)

	tag := ciphertext[len(ciphertext)-g.tagSize:]
	ciphertext = ciphertext[:len(ciphertext)-g.tagSize]

	var expectedTag [gcmTagSize]byte
	gcmAuth(expectedTag[:], &H, &tagMask, ciphertext, additionalData)
	if subtle.ConstantTimeCompare(expectedTag[:g.tagSize], tag) != 1 {
		// We sometimes decrypt and authenticate concurrently, so we overwrite
		// dst in the event of a tag mismatch. To be consistent across platforms
		// and to avoid releasing unauthenticated plaintext, we clear the buffer
		// in the event of an error.
		clear(out)
		return nil, errOpen
	}

	gcmCounterCryptGeneric(g.cipher, out, ciphertext, &counter)

	return ret, nil
}

func deriveCounter(H, counter *[gcmBlockSize]byte, nonce []byte) {
	if len(nonce) == gcmStandardNonceSize {
		copy(counter[:], nonce)
		counter[gcmBlockSize-1] = 1
	} else {
		lenBlock := make([]byte, 16)
		byteorder.BePutUint64(lenBlock[8:], uint64(len(nonce))*8)
		J := gcm.GHASH(H, nonce, lenBlock)
		copy(counter[:], J)
	}
}

func gcmCounterCryptGeneric(b Block, out, src []byte, counter *[gcmBlockSize]byte) {
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

func gcmInc32(counterBlock *[gcmBlockSize]byte) {
	ctr := counterBlock[len(counterBlock)-4:]
	byteorder.BePutUint32(ctr, byteorder.BeUint32(ctr)+1)
}

func gcmAuth(out []byte, H, tagMask *[gcmBlockSize]byte, ciphertext, additionalData []byte) {
	lenBlock := make([]byte, 16)
	byteorder.BePutUint64(lenBlock[:8], uint64(len(additionalData))*8)
	byteorder.BePutUint64(lenBlock[8:], uint64(len(ciphertext))*8)
	S := gcm.GHASH(H, additionalData, ciphertext, lenBlock)
	subtle.XORBytes(out, S, tagMask[:])
}

// sliceForAppend takes a slice and a requested number of bytes. It returns a
// slice with the contents of the given slice followed by that many bytes and a
// second slice that aliases into it and contains only the extra bytes. If the
// original slice has sufficient capacity then no allocation is performed.
func sliceForAppend(in []byte, n int) (head, tail []byte) {
	if total := len(in) + n; cap(in) >= total {
		head = in[:total]
	} else {
		head = make([]byte, total)
		copy(head, in)
	}
	tail = head[len(in):]
	return
}

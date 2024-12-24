// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher

import (
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/aes/gcm"
	"crypto/internal/fips140/alias"
	"crypto/internal/fips140only"
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
	if fips140only.Enabled {
		return nil, errors.New("crypto/cipher: use of GCM with arbitrary IVs is not allowed in FIPS 140-only mode, use NewGCMWithRandomNonce")
	}
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
	if fips140only.Enabled {
		return nil, errors.New("crypto/cipher: use of GCM with arbitrary IVs is not allowed in FIPS 140-only mode, use NewGCMWithRandomNonce")
	}
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
	if fips140only.Enabled {
		return nil, errors.New("crypto/cipher: use of GCM with arbitrary IVs is not allowed in FIPS 140-only mode, use NewGCMWithRandomNonce")
	}
	return newGCM(cipher, gcmStandardNonceSize, tagSize)
}

func newGCM(cipher Block, nonceSize, tagSize int) (AEAD, error) {
	c, ok := cipher.(*aes.Block)
	if !ok {
		if fips140only.Enabled {
			return nil, errors.New("crypto/cipher: use of GCM with non-AES ciphers is not allowed in FIPS 140-only mode")
		}
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

// NewGCMWithRandomNonce returns the given cipher wrapped in Galois Counter
// Mode, with randomly-generated nonces. The cipher must have been created by
// [aes.NewCipher].
//
// It generates a random 96-bit nonce, which is prepended to the ciphertext by Seal,
// and is extracted from the ciphertext by Open. The NonceSize of the AEAD is zero,
// while the Overhead is 28 bytes (the combination of nonce size and tag size).
//
// A given key MUST NOT be used to encrypt more than 2^32 messages, to limit the
// risk of a random nonce collision to negligible levels.
func NewGCMWithRandomNonce(cipher Block) (AEAD, error) {
	c, ok := cipher.(*aes.Block)
	if !ok {
		return nil, errors.New("cipher: NewGCMWithRandomNonce requires aes.Block")
	}
	g, err := gcm.New(c, gcmStandardNonceSize, gcmTagSize)
	if err != nil {
		return nil, err
	}
	return gcmWithRandomNonce{g}, nil
}

type gcmWithRandomNonce struct {
	*gcm.GCM
}

func (g gcmWithRandomNonce) NonceSize() int {
	return 0
}

func (g gcmWithRandomNonce) Overhead() int {
	return gcmStandardNonceSize + gcmTagSize
}

func (g gcmWithRandomNonce) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	if len(nonce) != 0 {
		panic("crypto/cipher: non-empty nonce passed to GCMWithRandomNonce")
	}

	ret, out := sliceForAppend(dst, gcmStandardNonceSize+len(plaintext)+gcmTagSize)
	if alias.InexactOverlap(out, plaintext) {
		panic("crypto/cipher: invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, additionalData) {
		panic("crypto/cipher: invalid buffer overlap of output and additional data")
	}
	nonce = out[:gcmStandardNonceSize]
	ciphertext := out[gcmStandardNonceSize:]

	// The AEAD interface allows using plaintext[:0] or ciphertext[:0] as dst.
	//
	// This is kind of a problem when trying to prepend or trim a nonce, because the
	// actual AES-GCTR blocks end up overlapping but not exactly.
	//
	// In Open, we write the output *before* the input, so unless we do something
	// weird like working through a chunk of block backwards, it works out.
	//
	// In Seal, we could work through the input backwards or intentionally load
	// ahead before writing.
	//
	// However, the crypto/internal/fips140/aes/gcm APIs also check for exact overlap,
	// so for now we just do a memmove if we detect overlap.
	//
	//     ┌───────────────────────────┬ ─ ─
	//     │PPPPPPPPPPPPPPPPPPPPPPPPPPP│    │
	//     └▽─────────────────────────▲┴ ─ ─
	//       ╲ Seal                    ╲
	//        ╲                    Open ╲
	//     ┌───▼─────────────────────────△──┐
	//     │NN|CCCCCCCCCCCCCCCCCCCCCCCCCCC|T│
	//     └────────────────────────────────┘
	//
	if alias.AnyOverlap(out, plaintext) {
		copy(ciphertext, plaintext)
		plaintext = ciphertext[:len(plaintext)]
	}

	gcm.SealWithRandomNonce(g.GCM, nonce, ciphertext, plaintext, additionalData)
	return ret
}

func (g gcmWithRandomNonce) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	if len(nonce) != 0 {
		panic("crypto/cipher: non-empty nonce passed to GCMWithRandomNonce")
	}
	if len(ciphertext) < gcmStandardNonceSize+gcmTagSize {
		return nil, errOpen
	}

	ret, out := sliceForAppend(dst, len(ciphertext)-gcmStandardNonceSize-gcmTagSize)
	if alias.InexactOverlap(out, ciphertext) {
		panic("crypto/cipher: invalid buffer overlap of output and input")
	}
	if alias.AnyOverlap(out, additionalData) {
		panic("crypto/cipher: invalid buffer overlap of output and additional data")
	}
	// See the discussion in Seal. Note that if there is any overlap at this
	// point, it's because out = ciphertext, so out must have enough capacity
	// even if we sliced the tag off. Also note how [AEAD] specifies that "the
	// contents of dst, up to its capacity, may be overwritten".
	if alias.AnyOverlap(out, ciphertext) {
		nonce = make([]byte, gcmStandardNonceSize)
		copy(nonce, ciphertext)
		copy(out[:len(ciphertext)], ciphertext[gcmStandardNonceSize:])
		ciphertext = out[:len(ciphertext)-gcmStandardNonceSize]
	} else {
		nonce = ciphertext[:gcmStandardNonceSize]
		ciphertext = ciphertext[gcmStandardNonceSize:]
	}

	_, err := g.GCM.Open(out[:0], nonce, ciphertext, additionalData)
	if err != nil {
		return nil, err
	}
	return ret, nil
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
// crypto/internal/fips140/aes/gcm/gcm_generic.go, refer to that file for more details.
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
		byteorder.BEPutUint64(lenBlock[8:], uint64(len(nonce))*8)
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
	byteorder.BEPutUint32(ctr, byteorder.BEUint32(ctr)+1)
}

func gcmAuth(out []byte, H, tagMask *[gcmBlockSize]byte, ciphertext, additionalData []byte) {
	lenBlock := make([]byte, 16)
	byteorder.BEPutUint64(lenBlock[:8], uint64(len(additionalData))*8)
	byteorder.BEPutUint64(lenBlock[8:], uint64(len(ciphertext))*8)
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

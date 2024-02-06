// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package aes

import (
	"crypto/cipher"
	"crypto/internal/alias"
	"crypto/subtle"
	"encoding/binary"
	"errors"
	"internal/cpu"
)

// This file contains two implementations of AES-GCM. The first implementation
// (gcmAsm) uses the KMCTR instruction to encrypt using AES in counter mode and
// the KIMD instruction for GHASH. The second implementation (gcmKMA) uses the
// newer KMA instruction which performs both operations.

// gcmCount represents a 16-byte big-endian count value.
type gcmCount [16]byte

// inc increments the rightmost 32-bits of the count value by 1.
func (x *gcmCount) inc() {
	binary.BigEndian.PutUint32(x[len(x)-4:], binary.BigEndian.Uint32(x[len(x)-4:])+1)
}

// gcmLengths writes len0 || len1 as big-endian values to a 16-byte array.
func gcmLengths(len0, len1 uint64) [16]byte {
	v := [16]byte{}
	binary.BigEndian.PutUint64(v[0:], len0)
	binary.BigEndian.PutUint64(v[8:], len1)
	return v
}

// gcmHashKey represents the 16-byte hash key required by the GHASH algorithm.
type gcmHashKey [16]byte

type gcmAsm struct {
	block     *aesCipherAsm
	hashKey   gcmHashKey
	nonceSize int
	tagSize   int
}

const (
	gcmBlockSize         = 16
	gcmTagSize           = 16
	gcmMinimumTagSize    = 12 // NIST SP 800-38D recommends tags with 12 or more bytes.
	gcmStandardNonceSize = 12
)

var errOpen = errors.New("cipher: message authentication failed")

// Assert that aesCipherAsm implements the gcmAble interface.
var _ gcmAble = (*aesCipherAsm)(nil)

// NewGCM returns the AES cipher wrapped in Galois Counter Mode. This is only
// called by [crypto/cipher.NewGCM] via the gcmAble interface.
func (c *aesCipherAsm) NewGCM(nonceSize, tagSize int) (cipher.AEAD, error) {
	var hk gcmHashKey
	c.Encrypt(hk[:], hk[:])
	g := gcmAsm{
		block:     c,
		hashKey:   hk,
		nonceSize: nonceSize,
		tagSize:   tagSize,
	}
	if cpu.S390X.HasAESGCM {
		g := gcmKMA{g}
		return &g, nil
	}
	return &g, nil
}

func (g *gcmAsm) NonceSize() int {
	return g.nonceSize
}

func (g *gcmAsm) Overhead() int {
	return g.tagSize
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

// ghash uses the GHASH algorithm to hash data with the given key. The initial
// hash value is given by hash which will be updated with the new hash value.
// The length of data must be a multiple of 16-bytes.
//
//go:noescape
func ghash(key *gcmHashKey, hash *[16]byte, data []byte)

// paddedGHASH pads data with zeroes until its length is a multiple of
// 16-bytes. It then calculates a new value for hash using the GHASH algorithm.
func (g *gcmAsm) paddedGHASH(hash *[16]byte, data []byte) {
	siz := len(data) &^ 0xf // align size to 16-bytes
	if siz > 0 {
		ghash(&g.hashKey, hash, data[:siz])
		data = data[siz:]
	}
	if len(data) > 0 {
		var s [16]byte
		copy(s[:], data)
		ghash(&g.hashKey, hash, s[:])
	}
}

// cryptBlocksGCM encrypts src using AES in counter mode using the given
// function code and key. The rightmost 32-bits of the counter are incremented
// between each block as required by the GCM spec. The initial counter value
// is given by cnt, which is updated with the value of the next counter value
// to use.
//
// The lengths of both dst and buf must be greater than or equal to the length
// of src. buf may be partially or completely overwritten during the execution
// of the function.
//
//go:noescape
func cryptBlocksGCM(fn code, key, dst, src, buf []byte, cnt *gcmCount)

// counterCrypt encrypts src using AES in counter mode and places the result
// into dst. cnt is the initial count value and will be updated with the next
// count value. The length of dst must be greater than or equal to the length
// of src.
func (g *gcmAsm) counterCrypt(dst, src []byte, cnt *gcmCount) {
	// Copying src into a buffer improves performance on some models when
	// src and dst point to the same underlying array. We also need a
	// buffer for counter values.
	var ctrbuf, srcbuf [2048]byte
	for len(src) >= 16 {
		siz := len(src)
		if len(src) > len(ctrbuf) {
			siz = len(ctrbuf)
		}
		siz &^= 0xf // align siz to 16-bytes
		copy(srcbuf[:], src[:siz])
		cryptBlocksGCM(g.block.function, g.block.key, dst[:siz], srcbuf[:siz], ctrbuf[:], cnt)
		src = src[siz:]
		dst = dst[siz:]
	}
	if len(src) > 0 {
		var x [16]byte
		g.block.Encrypt(x[:], cnt[:])
		for i := range src {
			dst[i] = src[i] ^ x[i]
		}
		cnt.inc()
	}
}

// deriveCounter computes the initial GCM counter state from the given nonce.
// See NIST SP 800-38D, section 7.1.
func (g *gcmAsm) deriveCounter(nonce []byte) gcmCount {
	// GCM has two modes of operation with respect to the initial counter
	// state: a "fast path" for 96-bit (12-byte) nonces, and a "slow path"
	// for nonces of other lengths. For a 96-bit nonce, the nonce, along
	// with a four-byte big-endian counter starting at one, is used
	// directly as the starting counter. For other nonce sizes, the counter
	// is computed by passing it through the GHASH function.
	var counter gcmCount
	if len(nonce) == gcmStandardNonceSize {
		copy(counter[:], nonce)
		counter[gcmBlockSize-1] = 1
	} else {
		var hash [16]byte
		g.paddedGHASH(&hash, nonce)
		lens := gcmLengths(0, uint64(len(nonce))*8)
		g.paddedGHASH(&hash, lens[:])
		copy(counter[:], hash[:])
	}
	return counter
}

// auth calculates GHASH(ciphertext, additionalData), masks the result with
// tagMask and writes the result to out.
func (g *gcmAsm) auth(out, ciphertext, additionalData []byte, tagMask *[gcmTagSize]byte) {
	var hash [16]byte
	g.paddedGHASH(&hash, additionalData)
	g.paddedGHASH(&hash, ciphertext)
	lens := gcmLengths(uint64(len(additionalData))*8, uint64(len(ciphertext))*8)
	g.paddedGHASH(&hash, lens[:])

	copy(out, hash[:])
	for i := range out {
		out[i] ^= tagMask[i]
	}
}

// Seal encrypts and authenticates plaintext. See the [cipher.AEAD] interface for
// details.
func (g *gcmAsm) Seal(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != g.nonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}
	if uint64(len(plaintext)) > ((1<<32)-2)*BlockSize {
		panic("crypto/cipher: message too large for GCM")
	}

	ret, out := sliceForAppend(dst, len(plaintext)+g.tagSize)
	if alias.InexactOverlap(out[:len(plaintext)], plaintext) {
		panic("crypto/cipher: invalid buffer overlap")
	}

	counter := g.deriveCounter(nonce)

	var tagMask [gcmBlockSize]byte
	g.block.Encrypt(tagMask[:], counter[:])
	counter.inc()

	var tagOut [gcmTagSize]byte
	g.counterCrypt(out, plaintext, &counter)
	g.auth(tagOut[:], out[:len(plaintext)], data, &tagMask)
	copy(out[len(plaintext):], tagOut[:])

	return ret
}

// Open authenticates and decrypts ciphertext. See the [cipher.AEAD] interface
// for details.
func (g *gcmAsm) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	if len(nonce) != g.nonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}
	// Sanity check to prevent the authentication from always succeeding if an implementation
	// leaves tagSize uninitialized, for example.
	if g.tagSize < gcmMinimumTagSize {
		panic("crypto/cipher: incorrect GCM tag size")
	}
	if len(ciphertext) < g.tagSize {
		return nil, errOpen
	}
	if uint64(len(ciphertext)) > ((1<<32)-2)*uint64(BlockSize)+uint64(g.tagSize) {
		return nil, errOpen
	}

	tag := ciphertext[len(ciphertext)-g.tagSize:]
	ciphertext = ciphertext[:len(ciphertext)-g.tagSize]

	counter := g.deriveCounter(nonce)

	var tagMask [gcmBlockSize]byte
	g.block.Encrypt(tagMask[:], counter[:])
	counter.inc()

	var expectedTag [gcmTagSize]byte
	g.auth(expectedTag[:], ciphertext, data, &tagMask)

	ret, out := sliceForAppend(dst, len(ciphertext))
	if alias.InexactOverlap(out, ciphertext) {
		panic("crypto/cipher: invalid buffer overlap")
	}

	if subtle.ConstantTimeCompare(expectedTag[:g.tagSize], tag) != 1 {
		// The AESNI code decrypts and authenticates concurrently, and
		// so overwrites dst in the event of a tag mismatch. That
		// behavior is mimicked here in order to be consistent across
		// platforms.
		for i := range out {
			out[i] = 0
		}
		return nil, errOpen
	}

	g.counterCrypt(out, ciphertext, &counter)
	return ret, nil
}

// gcmKMA implements the cipher.AEAD interface using the KMA instruction. It should
// only be used if hasKMA is true.
type gcmKMA struct {
	gcmAsm
}

// flags for the KMA instruction
const (
	kmaHS      = 1 << 10 // hash subkey supplied
	kmaLAAD    = 1 << 9  // last series of additional authenticated data
	kmaLPC     = 1 << 8  // last series of plaintext or ciphertext blocks
	kmaDecrypt = 1 << 7  // decrypt
)

// kmaGCM executes the encryption or decryption operation given by fn. The tag
// will be calculated and written to tag. cnt should contain the current
// counter state and will be overwritten with the updated counter state.
// TODO(mundaym): could pass in hash subkey
//
//go:noescape
func kmaGCM(fn code, key, dst, src, aad []byte, tag *[16]byte, cnt *gcmCount)

// Seal encrypts and authenticates plaintext. See the [cipher.AEAD] interface for
// details.
func (g *gcmKMA) Seal(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != g.nonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}
	if uint64(len(plaintext)) > ((1<<32)-2)*BlockSize {
		panic("crypto/cipher: message too large for GCM")
	}

	ret, out := sliceForAppend(dst, len(plaintext)+g.tagSize)
	if alias.InexactOverlap(out[:len(plaintext)], plaintext) {
		panic("crypto/cipher: invalid buffer overlap")
	}

	counter := g.deriveCounter(nonce)
	fc := g.block.function | kmaLAAD | kmaLPC

	var tag [gcmTagSize]byte
	kmaGCM(fc, g.block.key, out[:len(plaintext)], plaintext, data, &tag, &counter)
	copy(out[len(plaintext):], tag[:])

	return ret
}

// Open authenticates and decrypts ciphertext. See the [cipher.AEAD] interface
// for details.
func (g *gcmKMA) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	if len(nonce) != g.nonceSize {
		panic("crypto/cipher: incorrect nonce length given to GCM")
	}
	if len(ciphertext) < g.tagSize {
		return nil, errOpen
	}
	if uint64(len(ciphertext)) > ((1<<32)-2)*uint64(BlockSize)+uint64(g.tagSize) {
		return nil, errOpen
	}

	tag := ciphertext[len(ciphertext)-g.tagSize:]
	ciphertext = ciphertext[:len(ciphertext)-g.tagSize]
	ret, out := sliceForAppend(dst, len(ciphertext))
	if alias.InexactOverlap(out, ciphertext) {
		panic("crypto/cipher: invalid buffer overlap")
	}

	if g.tagSize < gcmMinimumTagSize {
		panic("crypto/cipher: incorrect GCM tag size")
	}

	counter := g.deriveCounter(nonce)
	fc := g.block.function | kmaLAAD | kmaLPC | kmaDecrypt

	var expectedTag [gcmTagSize]byte
	kmaGCM(fc, g.block.key, out[:len(ciphertext)], ciphertext, data, &expectedTag, &counter)

	if subtle.ConstantTimeCompare(expectedTag[:g.tagSize], tag) != 1 {
		// The AESNI code decrypts and authenticates concurrently, and
		// so overwrites dst in the event of a tag mismatch. That
		// behavior is mimicked here in order to be consistent across
		// platforms.
		for i := range out {
			out[i] = 0
		}
		return nil, errOpen
	}

	return ret, nil
}

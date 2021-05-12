// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64le
// +build ppc64le

package aes

import (
	"crypto/cipher"
	"crypto/subtle"
	"encoding/binary"
	"errors"
)

// This file implements GCM using an optimized GHASH function.

//go:noescape
func gcmInit(productTable *[256]byte, h []byte)

//go:noescape
func gcmHash(output []byte, productTable *[256]byte, inp []byte, len int)

//go:noescape
func gcmMul(output []byte, productTable *[256]byte)

const (
	gcmCounterSize       = 16
	gcmBlockSize         = 16
	gcmTagSize           = 16
	gcmStandardNonceSize = 12
)

var errOpen = errors.New("cipher: message authentication failed")

// Assert that aesCipherGCM implements the gcmAble interface.
var _ gcmAble = (*aesCipherAsm)(nil)

type gcmAsm struct {
	cipher *aesCipherAsm
	// ks is the key schedule, the length of which depends on the size of
	// the AES key.
	ks []uint32
	// productTable contains pre-computed multiples of the binary-field
	// element used in GHASH.
	productTable [256]byte
	// nonceSize contains the expected size of the nonce, in bytes.
	nonceSize int
	// tagSize contains the size of the tag, in bytes.
	tagSize int
}

// NewGCM returns the AES cipher wrapped in Galois Counter Mode. This is only
// called by crypto/cipher.NewGCM via the gcmAble interface.
func (c *aesCipherAsm) NewGCM(nonceSize, tagSize int) (cipher.AEAD, error) {
	g := &gcmAsm{cipher: c, ks: c.enc, nonceSize: nonceSize, tagSize: tagSize}

	hle := make([]byte, gcmBlockSize)
	c.Encrypt(hle, hle)

	// Reverse the bytes in each 8 byte chunk
	// Load little endian, store big endian
	h1 := binary.LittleEndian.Uint64(hle[:8])
	h2 := binary.LittleEndian.Uint64(hle[8:])
	binary.BigEndian.PutUint64(hle[:8], h1)
	binary.BigEndian.PutUint64(hle[8:], h2)
	gcmInit(&g.productTable, hle)

	return g, nil
}

func (g *gcmAsm) NonceSize() int {
	return g.nonceSize
}

func (g *gcmAsm) Overhead() int {
	return g.tagSize
}

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

// deriveCounter computes the initial GCM counter state from the given nonce.
func (g *gcmAsm) deriveCounter(counter *[gcmBlockSize]byte, nonce []byte) {
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
}

// counterCrypt encrypts in using AES in counter mode and places the result
// into out. counter is the initial count value and will be updated with the next
// count value. The length of out must be greater than or equal to the length
// of in.
func (g *gcmAsm) counterCrypt(out, in []byte, counter *[gcmBlockSize]byte) {
	var mask [gcmBlockSize]byte

	for len(in) >= gcmBlockSize {
		// Hint to avoid bounds check
		_, _ = in[15], out[15]
		g.cipher.Encrypt(mask[:], counter[:])
		gcmInc32(counter)

		// XOR 16 bytes each loop iteration in 8 byte chunks
		in0 := binary.LittleEndian.Uint64(in[0:])
		in1 := binary.LittleEndian.Uint64(in[8:])
		m0 := binary.LittleEndian.Uint64(mask[:8])
		m1 := binary.LittleEndian.Uint64(mask[8:])
		binary.LittleEndian.PutUint64(out[:8], in0^m0)
		binary.LittleEndian.PutUint64(out[8:], in1^m1)
		out = out[16:]
		in = in[16:]
	}

	if len(in) > 0 {
		g.cipher.Encrypt(mask[:], counter[:])
		gcmInc32(counter)
		// XOR leftover bytes
		for i, inb := range in {
			out[i] = inb ^ mask[i]
		}
	}
}

// increments the rightmost 32-bits of the count value by 1.
func gcmInc32(counterBlock *[16]byte) {
	c := counterBlock[len(counterBlock)-4:]
	x := binary.BigEndian.Uint32(c) + 1
	binary.BigEndian.PutUint32(c, x)
}

// paddedGHASH pads data with zeroes until its length is a multiple of
// 16-bytes. It then calculates a new value for hash using the ghash
// algorithm.
func (g *gcmAsm) paddedGHASH(hash *[16]byte, data []byte) {
	if siz := len(data) - (len(data) % gcmBlockSize); siz > 0 {
		gcmHash(hash[:], &g.productTable, data[:], siz)
		data = data[siz:]
	}
	if len(data) > 0 {
		var s [16]byte
		copy(s[:], data)
		gcmHash(hash[:], &g.productTable, s[:], len(s))
	}
}

// auth calculates GHASH(ciphertext, additionalData), masks the result with
// tagMask and writes the result to out.
func (g *gcmAsm) auth(out, ciphertext, aad []byte, tagMask *[gcmTagSize]byte) {
	var hash [16]byte
	g.paddedGHASH(&hash, aad)
	g.paddedGHASH(&hash, ciphertext)
	lens := gcmLengths(uint64(len(aad))*8, uint64(len(ciphertext))*8)
	g.paddedGHASH(&hash, lens[:])

	copy(out, hash[:])
	for i := range out {
		out[i] ^= tagMask[i]
	}
}

// Seal encrypts and authenticates plaintext. See the cipher.AEAD interface for
// details.
func (g *gcmAsm) Seal(dst, nonce, plaintext, data []byte) []byte {
	if len(nonce) != g.nonceSize {
		panic("cipher: incorrect nonce length given to GCM")
	}
	if uint64(len(plaintext)) > ((1<<32)-2)*BlockSize {
		panic("cipher: message too large for GCM")
	}

	ret, out := sliceForAppend(dst, len(plaintext)+g.tagSize)

	var counter, tagMask [gcmBlockSize]byte
	g.deriveCounter(&counter, nonce)

	g.cipher.Encrypt(tagMask[:], counter[:])
	gcmInc32(&counter)

	g.counterCrypt(out, plaintext, &counter)
	g.auth(out[len(plaintext):], out[:len(plaintext)], data, &tagMask)

	return ret
}

// Open authenticates and decrypts ciphertext. See the cipher.AEAD interface
// for details.
func (g *gcmAsm) Open(dst, nonce, ciphertext, data []byte) ([]byte, error) {
	if len(nonce) != g.nonceSize {
		panic("cipher: incorrect nonce length given to GCM")
	}
	if len(ciphertext) < g.tagSize {
		return nil, errOpen
	}
	if uint64(len(ciphertext)) > ((1<<32)-2)*uint64(BlockSize)+uint64(g.tagSize) {
		return nil, errOpen
	}

	tag := ciphertext[len(ciphertext)-g.tagSize:]
	ciphertext = ciphertext[:len(ciphertext)-g.tagSize]

	var counter, tagMask [gcmBlockSize]byte
	g.deriveCounter(&counter, nonce)

	g.cipher.Encrypt(tagMask[:], counter[:])
	gcmInc32(&counter)

	var expectedTag [gcmTagSize]byte
	g.auth(expectedTag[:], ciphertext, data, &tagMask)

	ret, out := sliceForAppend(dst, len(ciphertext))

	if subtle.ConstantTimeCompare(expectedTag[:g.tagSize], tag) != 1 {
		for i := range out {
			out[i] = 0
		}
		return nil, errOpen
	}

	g.counterCrypt(out, ciphertext, &counter)
	return ret, nil
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

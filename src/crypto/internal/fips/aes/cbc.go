// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import (
	"crypto/internal/fips/alias"
	"crypto/internal/fips/subtle"
)

type CBCEncrypter struct {
	b  Block
	iv [BlockSize]byte
}

// NewCBCEncrypter returns a [cipher.BlockMode] which encrypts in cipher block
// chaining mode, using the given Block.
func NewCBCEncrypter(b *Block, iv [BlockSize]byte) *CBCEncrypter {
	return &CBCEncrypter{b: *b, iv: iv}
}

func (c *CBCEncrypter) BlockSize() int { return BlockSize }

func (c *CBCEncrypter) CryptBlocks(dst, src []byte) {
	if len(src)%BlockSize != 0 {
		panic("crypto/cipher: input not full blocks")
	}
	if len(dst) < len(src) {
		panic("crypto/cipher: output smaller than input")
	}
	if alias.InexactOverlap(dst[:len(src)], src) {
		panic("crypto/cipher: invalid buffer overlap")
	}
	if len(src) == 0 {
		return
	}
	cryptBlocksEnc(&c.b, &c.iv, dst, src)
}

func (x *CBCEncrypter) SetIV(iv []byte) {
	if len(iv) != len(x.iv) {
		panic("cipher: incorrect length IV")
	}
	copy(x.iv[:], iv)
}

func cryptBlocksEncGeneric(b *Block, civ *[BlockSize]byte, dst, src []byte) {
	iv := civ[:]
	for len(src) > 0 {
		// Write the xor to dst, then encrypt in place.
		subtle.XORBytes(dst[:BlockSize], src[:BlockSize], iv)
		b.Encrypt(dst[:BlockSize], dst[:BlockSize])

		// Move to the next block with this block as the next iv.
		iv = dst[:BlockSize]
		src = src[BlockSize:]
		dst = dst[BlockSize:]
	}

	// Save the iv for the next CryptBlocks call.
	copy(civ[:], iv)
}

type CBCDecrypter struct {
	b  Block
	iv [BlockSize]byte
}

// NewCBCDecrypter returns a [cipher.BlockMode] which decrypts in cipher block
// chaining mode, using the given Block.
func NewCBCDecrypter(b *Block, iv [BlockSize]byte) *CBCDecrypter {
	return &CBCDecrypter{b: *b, iv: iv}
}

func (c *CBCDecrypter) BlockSize() int { return BlockSize }

func (c *CBCDecrypter) CryptBlocks(dst, src []byte) {
	if len(src)%BlockSize != 0 {
		panic("crypto/cipher: input not full blocks")
	}
	if len(dst) < len(src) {
		panic("crypto/cipher: output smaller than input")
	}
	if alias.InexactOverlap(dst[:len(src)], src) {
		panic("crypto/cipher: invalid buffer overlap")
	}
	if len(src) == 0 {
		return
	}
	cryptBlocksDec(&c.b, &c.iv, dst, src)
}

func (x *CBCDecrypter) SetIV(iv []byte) {
	if len(iv) != len(x.iv) {
		panic("cipher: incorrect length IV")
	}
	copy(x.iv[:], iv)
}

func cryptBlocksDecGeneric(b *Block, civ *[BlockSize]byte, dst, src []byte) {
	// For each block, we need to xor the decrypted data with the previous
	// block's ciphertext (the iv). To avoid making a copy each time, we loop
	// over the blocks backwards.
	end := len(src)
	start := end - BlockSize
	prev := start - BlockSize

	// Copy the last block of ciphertext as the IV of the next call.
	iv := *civ
	copy(civ[:], src[start:end])

	for start >= 0 {
		b.Decrypt(dst[start:end], src[start:end])

		if start > 0 {
			subtle.XORBytes(dst[start:end], dst[start:end], src[prev:start])
		} else {
			// The first block is special because it uses the saved iv.
			subtle.XORBytes(dst[start:end], dst[start:end], iv[:])
		}

		end -= BlockSize
		start -= BlockSize
		prev -= BlockSize
	}
}

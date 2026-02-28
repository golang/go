// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cipher block chaining (CBC) mode.

// CBC provides confidentiality by xoring (chaining) each plaintext block
// with the previous ciphertext block before applying the block cipher.

// See NIST SP 800-38A, pp 10-11

package cipher

import (
	"bytes"
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/alias"
	"crypto/internal/fips140only"
	"crypto/subtle"
)

type cbc struct {
	b         Block
	blockSize int
	iv        []byte
	tmp       []byte
}

func newCBC(b Block, iv []byte) *cbc {
	return &cbc{
		b:         b,
		blockSize: b.BlockSize(),
		iv:        bytes.Clone(iv),
		tmp:       make([]byte, b.BlockSize()),
	}
}

type cbcEncrypter cbc

// cbcEncAble is an interface implemented by ciphers that have a specific
// optimized implementation of CBC encryption. crypto/aes doesn't use this
// anymore, and we'd like to eventually remove it.
type cbcEncAble interface {
	NewCBCEncrypter(iv []byte) BlockMode
}

// NewCBCEncrypter returns a BlockMode which encrypts in cipher block chaining
// mode, using the given Block. The length of iv must be the same as the
// Block's block size.
func NewCBCEncrypter(b Block, iv []byte) BlockMode {
	if len(iv) != b.BlockSize() {
		panic("cipher.NewCBCEncrypter: IV length must equal block size")
	}
	if b, ok := b.(*aes.Block); ok {
		return aes.NewCBCEncrypter(b, [16]byte(iv))
	}
	if fips140only.Enforced() {
		panic("crypto/cipher: use of CBC with non-AES ciphers is not allowed in FIPS 140-only mode")
	}
	if cbc, ok := b.(cbcEncAble); ok {
		return cbc.NewCBCEncrypter(iv)
	}
	return (*cbcEncrypter)(newCBC(b, iv))
}

// newCBCGenericEncrypter returns a BlockMode which encrypts in cipher block chaining
// mode, using the given Block. The length of iv must be the same as the
// Block's block size. This always returns the generic non-asm encrypter for use
// in fuzz testing.
func newCBCGenericEncrypter(b Block, iv []byte) BlockMode {
	if len(iv) != b.BlockSize() {
		panic("cipher.NewCBCEncrypter: IV length must equal block size")
	}
	return (*cbcEncrypter)(newCBC(b, iv))
}

func (x *cbcEncrypter) BlockSize() int { return x.blockSize }

func (x *cbcEncrypter) CryptBlocks(dst, src []byte) {
	if len(src)%x.blockSize != 0 {
		panic("crypto/cipher: input not full blocks")
	}
	if len(dst) < len(src) {
		panic("crypto/cipher: output smaller than input")
	}
	if alias.InexactOverlap(dst[:len(src)], src) {
		panic("crypto/cipher: invalid buffer overlap")
	}
	if _, ok := x.b.(*aes.Block); ok {
		panic("crypto/cipher: internal error: generic CBC used with AES")
	}

	iv := x.iv

	for len(src) > 0 {
		// Write the xor to dst, then encrypt in place.
		subtle.XORBytes(dst[:x.blockSize], src[:x.blockSize], iv)
		x.b.Encrypt(dst[:x.blockSize], dst[:x.blockSize])

		// Move to the next block with this block as the next iv.
		iv = dst[:x.blockSize]
		src = src[x.blockSize:]
		dst = dst[x.blockSize:]
	}

	// Save the iv for the next CryptBlocks call.
	copy(x.iv, iv)
}

func (x *cbcEncrypter) SetIV(iv []byte) {
	if len(iv) != len(x.iv) {
		panic("cipher: incorrect length IV")
	}
	copy(x.iv, iv)
}

type cbcDecrypter cbc

// cbcDecAble is an interface implemented by ciphers that have a specific
// optimized implementation of CBC decryption. crypto/aes doesn't use this
// anymore, and we'd like to eventually remove it.
type cbcDecAble interface {
	NewCBCDecrypter(iv []byte) BlockMode
}

// NewCBCDecrypter returns a BlockMode which decrypts in cipher block chaining
// mode, using the given Block. The length of iv must be the same as the
// Block's block size and must match the iv used to encrypt the data.
func NewCBCDecrypter(b Block, iv []byte) BlockMode {
	if len(iv) != b.BlockSize() {
		panic("cipher.NewCBCDecrypter: IV length must equal block size")
	}
	if b, ok := b.(*aes.Block); ok {
		return aes.NewCBCDecrypter(b, [16]byte(iv))
	}
	if fips140only.Enforced() {
		panic("crypto/cipher: use of CBC with non-AES ciphers is not allowed in FIPS 140-only mode")
	}
	if cbc, ok := b.(cbcDecAble); ok {
		return cbc.NewCBCDecrypter(iv)
	}
	return (*cbcDecrypter)(newCBC(b, iv))
}

// newCBCGenericDecrypter returns a BlockMode which encrypts in cipher block chaining
// mode, using the given Block. The length of iv must be the same as the
// Block's block size. This always returns the generic non-asm decrypter for use in
// fuzz testing.
func newCBCGenericDecrypter(b Block, iv []byte) BlockMode {
	if len(iv) != b.BlockSize() {
		panic("cipher.NewCBCDecrypter: IV length must equal block size")
	}
	return (*cbcDecrypter)(newCBC(b, iv))
}

func (x *cbcDecrypter) BlockSize() int { return x.blockSize }

func (x *cbcDecrypter) CryptBlocks(dst, src []byte) {
	if len(src)%x.blockSize != 0 {
		panic("crypto/cipher: input not full blocks")
	}
	if len(dst) < len(src) {
		panic("crypto/cipher: output smaller than input")
	}
	if alias.InexactOverlap(dst[:len(src)], src) {
		panic("crypto/cipher: invalid buffer overlap")
	}
	if _, ok := x.b.(*aes.Block); ok {
		panic("crypto/cipher: internal error: generic CBC used with AES")
	}
	if len(src) == 0 {
		return
	}

	// For each block, we need to xor the decrypted data with the previous block's ciphertext (the iv).
	// To avoid making a copy each time, we loop over the blocks BACKWARDS.
	end := len(src)
	start := end - x.blockSize
	prev := start - x.blockSize

	// Copy the last block of ciphertext in preparation as the new iv.
	copy(x.tmp, src[start:end])

	// Loop over all but the first block.
	for start > 0 {
		x.b.Decrypt(dst[start:end], src[start:end])
		subtle.XORBytes(dst[start:end], dst[start:end], src[prev:start])

		end = start
		start = prev
		prev -= x.blockSize
	}

	// The first block is special because it uses the saved iv.
	x.b.Decrypt(dst[start:end], src[start:end])
	subtle.XORBytes(dst[start:end], dst[start:end], x.iv)

	// Set the new iv to the first block we copied earlier.
	x.iv, x.tmp = x.tmp, x.iv
}

func (x *cbcDecrypter) SetIV(iv []byte) {
	if len(iv) != len(x.iv) {
		panic("cipher: incorrect length IV")
	}
	copy(x.iv, iv)
}

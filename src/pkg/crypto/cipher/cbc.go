// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cipher block chaining (CBC) mode.

// CBC provides confidentiality by xoring (chaining) each plaintext block
// with the previous ciphertext block before applying the block cipher.

// See NIST SP 800-38A, pp 10-11

package cipher

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
		iv:        dup(iv),
		tmp:       make([]byte, b.BlockSize()),
	}
}

type cbcEncrypter cbc

// NewCBCEncrypter returns a BlockMode which encrypts in cipher block chaining
// mode, using the given Block. The length of iv must be the same as the
// Block's block size.
func NewCBCEncrypter(b Block, iv []byte) BlockMode {
	return (*cbcEncrypter)(newCBC(b, iv))
}

func (x *cbcEncrypter) BlockSize() int { return x.blockSize }

func (x *cbcEncrypter) CryptBlocks(dst, src []byte) {
	for len(src) > 0 {
		for i := 0; i < x.blockSize; i++ {
			x.iv[i] ^= src[i]
		}
		x.b.Encrypt(x.iv, x.iv)
		for i := 0; i < x.blockSize; i++ {
			dst[i] = x.iv[i]
		}
		src = src[x.blockSize:]
		dst = dst[x.blockSize:]
	}
}

type cbcDecrypter cbc

// NewCBCDecrypter returns a BlockMode which decrypts in cipher block chaining
// mode, using the given Block. The length of iv must be the same as the
// Block's block size and must match the iv used to encrypt the data.
func NewCBCDecrypter(b Block, iv []byte) BlockMode {
	return (*cbcDecrypter)(newCBC(b, iv))
}

func (x *cbcDecrypter) BlockSize() int { return x.blockSize }

func (x *cbcDecrypter) CryptBlocks(dst, src []byte) {
	for len(src) > 0 {
		x.b.Decrypt(x.tmp, src[:x.blockSize])
		for i := 0; i < x.blockSize; i++ {
			x.tmp[i] ^= x.iv[i]
			x.iv[i] = src[i]
			dst[i] = x.tmp[i]
		}

		src = src[x.blockSize:]
		dst = dst[x.blockSize:]
	}
}

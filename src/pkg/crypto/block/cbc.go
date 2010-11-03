// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cipher block chaining (CBC) mode.

// CBC provides confidentiality by xoring (chaining) each plaintext block
// with the previous ciphertext block before applying the block cipher.

// See NIST SP 800-38A, pp 10-11

package block

import (
	"io"
)

type cbcCipher struct {
	c         Cipher
	blockSize int
	iv        []byte
	tmp       []byte
}

func newCBC(c Cipher, iv []byte) *cbcCipher {
	n := c.BlockSize()
	x := new(cbcCipher)
	x.c = c
	x.blockSize = n
	x.iv = dup(iv)
	x.tmp = make([]byte, n)
	return x
}

func (x *cbcCipher) BlockSize() int { return x.blockSize }

func (x *cbcCipher) Encrypt(dst, src []byte) {
	for i := 0; i < x.blockSize; i++ {
		x.iv[i] ^= src[i]
	}
	x.c.Encrypt(x.iv, x.iv)
	for i := 0; i < x.blockSize; i++ {
		dst[i] = x.iv[i]
	}
}

func (x *cbcCipher) Decrypt(dst, src []byte) {
	x.c.Decrypt(x.tmp, src)
	for i := 0; i < x.blockSize; i++ {
		x.tmp[i] ^= x.iv[i]
		x.iv[i] = src[i]
		dst[i] = x.tmp[i]
	}
}

// NewCBCDecrypter returns a reader that reads data from r and decrypts it using c
// in cipher block chaining (CBC) mode with the initialization vector iv.
// The returned Reader does not buffer or read ahead except
// as required by the cipher's block size.
func NewCBCDecrypter(c Cipher, iv []byte, r io.Reader) io.Reader {
	return NewECBDecrypter(newCBC(c, iv), r)
}

// NewCBCEncrypter returns a writer that encrypts data using c
// in cipher block chaining (CBC) mode with the initialization vector iv
// and writes the encrypted data to w.
// The returned Writer does no buffering except as required
// by the cipher's block size, so there is no need for a Flush method.
func NewCBCEncrypter(c Cipher, iv []byte, w io.Writer) io.Writer {
	return NewECBEncrypter(newCBC(c, iv), w)
}

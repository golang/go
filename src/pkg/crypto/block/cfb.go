// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cipher feedback (CFB) mode.

// CFB provides confidentiality by feeding a fraction of
// the previous ciphertext in as the plaintext for the next
// block operation.

// See NIST SP 800-38A, pp 11-13

package block

import (
	"io"
)

type cfbCipher struct {
	c          Cipher
	blockSize  int // our block size (s/8)
	cipherSize int // underlying cipher block size
	iv         []byte
	tmp        []byte
}

func newCFB(c Cipher, s int, iv []byte) *cfbCipher {
	if s == 0 || s%8 != 0 {
		panic("crypto/block: invalid CFB mode")
	}
	b := c.BlockSize()
	x := new(cfbCipher)
	x.c = c
	x.blockSize = s / 8
	x.cipherSize = b
	x.iv = dup(iv)
	x.tmp = make([]byte, b)
	return x
}

func (x *cfbCipher) BlockSize() int { return x.blockSize }

func (x *cfbCipher) Encrypt(dst, src []byte) {
	// Encrypt old IV and xor prefix with src to make dst.
	x.c.Encrypt(x.tmp, x.iv)
	for i := 0; i < x.blockSize; i++ {
		dst[i] = src[i] ^ x.tmp[i]
	}

	// Slide unused IV pieces down and insert dst at end.
	for i := 0; i < x.cipherSize-x.blockSize; i++ {
		x.iv[i] = x.iv[i+x.blockSize]
	}
	off := x.cipherSize - x.blockSize
	for i := off; i < x.cipherSize; i++ {
		x.iv[i] = dst[i-off]
	}
}

func (x *cfbCipher) Decrypt(dst, src []byte) {
	// Encrypt [sic] old IV and xor prefix with src to make dst.
	x.c.Encrypt(x.tmp, x.iv)
	for i := 0; i < x.blockSize; i++ {
		dst[i] = src[i] ^ x.tmp[i]
	}

	// Slide unused IV pieces down and insert src at top.
	for i := 0; i < x.cipherSize-x.blockSize; i++ {
		x.iv[i] = x.iv[i+x.blockSize]
	}
	off := x.cipherSize - x.blockSize
	for i := off; i < x.cipherSize; i++ {
		// Reconstruct src = dst ^ x.tmp
		// in case we overwrote src (src == dst).
		x.iv[i] = dst[i-off] ^ x.tmp[i-off]
	}
}

// NewCFBDecrypter returns a reader that reads data from r and decrypts it using c
// in s-bit cipher feedback (CFB) mode with the initialization vector iv.
// The returned Reader does not buffer or read ahead except
// as required by the cipher's block size.
// Modes for s not a multiple of 8 are unimplemented.
func NewCFBDecrypter(c Cipher, s int, iv []byte, r io.Reader) io.Reader {
	return NewECBDecrypter(newCFB(c, s, iv), r)
}

// NewCFBEncrypter returns a writer that encrypts data using c
// in s-bit cipher feedback (CFB) mode with the initialization vector iv
// and writes the encrypted data to w.
// The returned Writer does no buffering except as required
// by the cipher's block size, so there is no need for a Flush method.
// Modes for s not a multiple of 8 are unimplemented.
func NewCFBEncrypter(c Cipher, s int, iv []byte, w io.Writer) io.Writer {
	return NewECBEncrypter(newCFB(c, s, iv), w)
}

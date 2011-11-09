// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import "strconv"

// The AES block size in bytes.
const BlockSize = 16

// A Cipher is an instance of AES encryption using a particular key.
type Cipher struct {
	enc []uint32
	dec []uint32
}

type KeySizeError int

func (k KeySizeError) Error() string {
	return "crypto/aes: invalid key size " + strconv.Itoa(int(k))
}

// NewCipher creates and returns a new Cipher.
// The key argument should be the AES key,
// either 16, 24, or 32 bytes to select
// AES-128, AES-192, or AES-256.
func NewCipher(key []byte) (*Cipher, error) {
	k := len(key)
	switch k {
	default:
		return nil, KeySizeError(k)
	case 16, 24, 32:
		break
	}

	n := k + 28
	c := &Cipher{make([]uint32, n), make([]uint32, n)}
	expandKey(key, c.enc, c.dec)
	return c, nil
}

// BlockSize returns the AES block size, 16 bytes.
// It is necessary to satisfy the Block interface in the
// package "crypto/cipher".
func (c *Cipher) BlockSize() int { return BlockSize }

// Encrypt encrypts the 16-byte buffer src using the key k
// and stores the result in dst.
// Note that for amounts of data larger than a block,
// it is not safe to just call Encrypt on successive blocks;
// instead, use an encryption mode like CBC (see crypto/cipher/cbc.go).
func (c *Cipher) Encrypt(dst, src []byte) { encryptBlock(c.enc, dst, src) }

// Decrypt decrypts the 16-byte buffer src using the key k
// and stores the result in dst.
func (c *Cipher) Decrypt(dst, src []byte) { decryptBlock(c.dec, dst, src) }

// Reset zeros the key data, so that it will no longer
// appear in the process's memory.
func (c *Cipher) Reset() {
	for i := 0; i < len(c.enc); i++ {
		c.enc[i] = 0
	}
	for i := 0; i < len(c.dec); i++ {
		c.dec[i] = 0
	}
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package des

import "strconv"

// The DES block size in bytes.
const BlockSize = 8

type KeySizeError int

func (k KeySizeError) Error() string {
	return "crypto/des: invalid key size " + strconv.Itoa(int(k))
}

// Cipher is an instance of DES encryption.
type Cipher struct {
	subkeys [16]uint64
}

// NewCipher creates and returns a new Cipher.
func NewCipher(key []byte) (*Cipher, error) {
	if len(key) != 8 {
		return nil, KeySizeError(len(key))
	}

	c := new(Cipher)
	c.generateSubkeys(key)
	return c, nil
}

// BlockSize returns the DES block size, 8 bytes.
func (c *Cipher) BlockSize() int { return BlockSize }

// Encrypts the 8-byte buffer src and stores the result in dst.
// Note that for amounts of data larger than a block,
// it is not safe to just call Encrypt on successive blocks;
// instead, use an encryption mode like CBC (see crypto/cipher/cbc.go).
func (c *Cipher) Encrypt(dst, src []byte) { encryptBlock(c.subkeys[:], dst, src) }

// Decrypts the 8-byte buffer src and stores the result in dst.
func (c *Cipher) Decrypt(dst, src []byte) { decryptBlock(c.subkeys[:], dst, src) }

// Reset zeros the key data, so that it will no longer
// appear in the process's memory.
func (c *Cipher) Reset() {
	for i := 0; i < len(c.subkeys); i++ {
		c.subkeys[i] = 0
	}
}

// A TripleDESCipher is an instance of TripleDES encryption.
type TripleDESCipher struct {
	cipher1, cipher2, cipher3 Cipher
}

// NewCipher creates and returns a new Cipher.
func NewTripleDESCipher(key []byte) (*TripleDESCipher, error) {
	if len(key) != 24 {
		return nil, KeySizeError(len(key))
	}

	c := new(TripleDESCipher)
	c.cipher1.generateSubkeys(key[:8])
	c.cipher2.generateSubkeys(key[8:16])
	c.cipher3.generateSubkeys(key[16:])
	return c, nil
}

// BlockSize returns the TripleDES block size, 8 bytes.
// It is necessary to satisfy the Block interface in the
// package "crypto/cipher".
func (c *TripleDESCipher) BlockSize() int { return BlockSize }

// Encrypts the 8-byte buffer src and stores the result in dst.
// Note that for amounts of data larger than a block,
// it is not safe to just call Encrypt on successive blocks;
// instead, use an encryption mode like CBC (see crypto/cipher/cbc.go).
func (c *TripleDESCipher) Encrypt(dst, src []byte) {
	c.cipher1.Encrypt(dst, src)
	c.cipher2.Decrypt(dst, dst)
	c.cipher3.Encrypt(dst, dst)
}

// Decrypts the 8-byte buffer src and stores the result in dst.
func (c *TripleDESCipher) Decrypt(dst, src []byte) {
	c.cipher3.Decrypt(dst, src)
	c.cipher2.Encrypt(dst, dst)
	c.cipher1.Decrypt(dst, dst)
}

// Reset zeros the key data, so that it will no longer
// appear in the process's memory.
func (c *TripleDESCipher) Reset() {
	c.cipher1.Reset()
	c.cipher2.Reset()
	c.cipher3.Reset()
}

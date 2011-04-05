// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package des

import (
	"os"
	"strconv"
)

// The DES block size in bytes.
const BlockSize = 8

type KeySizeError int

func (k KeySizeError) String() string {
	return "crypto/des: invalid key size " + strconv.Itoa(int(k))
}

// A DESCipher is an instance of DES encryption.
type DESCipher struct {
	key     []byte
	subkeys [16]uint64
}

// NewCipher creates and returns a new Cipher.
func NewDESCipher(key []byte) (*DESCipher, os.Error) {
	k := len(key)
	if k != 8 {
		return nil, KeySizeError(k)
	}

	c := &DESCipher{key, [16]uint64{}}
	ksGenerateSubkeys(c)
	return c, nil
}

// BlockSize returns the DES block size, 8 bytes.
func (c *DESCipher) BlockSize() int { return BlockSize }

// Encrypts the 8-byte buffer src and stores the result in dst.
// Note that for amounts of data larger than a block,
// it is not safe to just call Encrypt on successive blocks;
// instead, use an encryption mode like CBC (see crypto/cipher/cbc.go).
func (c *DESCipher) Encrypt(dst, src []byte) { encryptBlock(c.subkeys, dst, src) }

// Decrypts the 8-byte buffer src and stores the result in dst.
func (c *DESCipher) Decrypt(dst, src []byte) { decryptBlock(c.subkeys, dst, src) }

// Reset zeros the key data, so that it will no longer
// appear in the process's memory.
func (c *DESCipher) Reset() {
	for i := 0; i < len(c.key); i++ {
		c.key[i] = 0
	}
	for i := 0; i < len(c.subkeys); i++ {
		c.subkeys[i] = 0
	}
}

// A TripleDESCipher is an instance of TripleDES encryption.
type TripleDESCipher struct {
	key                       []byte
	cipher1, cipher2, cipher3 *DESCipher
}

// NewCipher creates and returns a new Cipher.
func NewTripleDESCipher(key []byte) (*TripleDESCipher, os.Error) {
	k := len(key)
	if k != 24 {
		return nil, KeySizeError(k)
	}

	cipher1, _ := NewDESCipher(key[0:8])
	cipher2, _ := NewDESCipher(key[8:16])
	cipher3, _ := NewDESCipher(key[16:])
	c := &TripleDESCipher{key, cipher1, cipher2, cipher3}
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
	for i := 0; i < len(c.key); i++ {
		c.key[i] = 0
	}
	c.cipher1.Reset()
	c.cipher2.Reset()
	c.cipher3.Reset()
}

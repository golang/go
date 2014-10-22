// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package des

import (
	"crypto/cipher"
	"strconv"
)

// The DES block size in bytes.
const BlockSize = 8

type KeySizeError int

func (k KeySizeError) Error() string {
	return "crypto/des: invalid key size " + strconv.Itoa(int(k))
}

// desCipher is an instance of DES encryption.
type desCipher struct {
	subkeys [16]uint64
}

// NewCipher creates and returns a new cipher.Block.
func NewCipher(key []byte) (cipher.Block, error) {
	if len(key) != 8 {
		return nil, KeySizeError(len(key))
	}

	c := new(desCipher)
	c.generateSubkeys(key)
	return c, nil
}

func (c *desCipher) BlockSize() int { return BlockSize }

func (c *desCipher) Encrypt(dst, src []byte) { encryptBlock(c.subkeys[:], dst, src) }

func (c *desCipher) Decrypt(dst, src []byte) { decryptBlock(c.subkeys[:], dst, src) }

// A tripleDESCipher is an instance of TripleDES encryption.
type tripleDESCipher struct {
	cipher1, cipher2, cipher3 desCipher
}

// NewTripleDESCipher creates and returns a new cipher.Block.
func NewTripleDESCipher(key []byte) (cipher.Block, error) {
	if len(key) != 24 {
		return nil, KeySizeError(len(key))
	}

	c := new(tripleDESCipher)
	c.cipher1.generateSubkeys(key[:8])
	c.cipher2.generateSubkeys(key[8:16])
	c.cipher3.generateSubkeys(key[16:])
	return c, nil
}

func (c *tripleDESCipher) BlockSize() int { return BlockSize }

func (c *tripleDESCipher) Encrypt(dst, src []byte) {
	c.cipher1.Encrypt(dst, src)
	c.cipher2.Decrypt(dst, dst)
	c.cipher3.Encrypt(dst, dst)
}

func (c *tripleDESCipher) Decrypt(dst, src []byte) {
	c.cipher3.Decrypt(dst, src)
	c.cipher2.Encrypt(dst, dst)
	c.cipher1.Decrypt(dst, dst)
}

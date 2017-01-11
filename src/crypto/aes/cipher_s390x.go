// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import (
	"crypto/cipher"
	"crypto/internal/cipherhw"
)

type code int

// Function codes for the cipher message family of instructions.
const (
	aes128 code = 18
	aes192      = 19
	aes256      = 20
)

type aesCipherAsm struct {
	function code      // code for cipher message instruction
	key      []byte    // key (128, 192 or 256 bytes)
	storage  [256]byte // array backing key slice
}

// cryptBlocks invokes the cipher message (KM) instruction with
// the given function code. This is equivalent to AES in ECB
// mode. The length must be a multiple of BlockSize (16).
//go:noescape
func cryptBlocks(c code, key, dst, src *byte, length int)

var useAsm = cipherhw.AESGCMSupport()

func newCipher(key []byte) (cipher.Block, error) {
	if !useAsm {
		return newCipherGeneric(key)
	}

	var function code
	switch len(key) {
	case 128 / 8:
		function = aes128
	case 192 / 8:
		function = aes192
	case 256 / 8:
		function = aes256
	default:
		return nil, KeySizeError(len(key))
	}

	var c aesCipherAsm
	c.function = function
	c.key = c.storage[:len(key)]
	copy(c.key, key)
	return &c, nil
}

func (c *aesCipherAsm) BlockSize() int { return BlockSize }

func (c *aesCipherAsm) Encrypt(dst, src []byte) {
	if len(src) < BlockSize {
		panic("crypto/aes: input not full block")
	}
	if len(dst) < BlockSize {
		panic("crypto/aes: output not full block")
	}
	cryptBlocks(c.function, &c.key[0], &dst[0], &src[0], BlockSize)
}

func (c *aesCipherAsm) Decrypt(dst, src []byte) {
	if len(src) < BlockSize {
		panic("crypto/aes: input not full block")
	}
	if len(dst) < BlockSize {
		panic("crypto/aes: output not full block")
	}
	// The decrypt function code is equal to the function code + 128.
	cryptBlocks(c.function+128, &c.key[0], &dst[0], &src[0], BlockSize)
}

// expandKey is used by BenchmarkExpand. cipher message (KM) does not need key
// expansion so there is no assembly equivalent.
func expandKey(key []byte, enc, dec []uint32) {
	expandKeyGo(key, enc, dec)
}

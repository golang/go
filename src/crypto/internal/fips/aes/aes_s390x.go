// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package aes

import (
	"crypto/internal/impl"
	"internal/cpu"
)

type code int

// Function codes for the cipher message family of instructions.
const (
	aes128 code = 18
	aes192 code = 19
	aes256 code = 20
)

type block struct {
	function code     // code for cipher message instruction
	key      []byte   // key (128, 192 or 256 bits)
	storage  [32]byte // array backing key slice

	fallback *blockExpanded
}

// cryptBlocks invokes the cipher message (KM) instruction with
// the given function code. This is equivalent to AES in ECB
// mode. The length must be a multiple of BlockSize (16).
//
//go:noescape
func cryptBlocks(c code, key, dst, src *byte, length int)

var supportsAES = cpu.S390X.HasAES && cpu.S390X.HasAESCBC

func init() {
	// CP Assist for Cryptographic Functions (CPACF)
	// https://www.ibm.com/docs/en/zos/3.1.0?topic=icsf-cp-assist-cryptographic-functions-cpacf
	impl.Register("aes", "CPACF", &supportsAES)
}

func checkGenericIsExpected() {
	if supportsAES {
		panic("crypto/aes: internal error: using generic implementation despite hardware support")
	}
}

func newBlock(c *Block, key []byte) *Block {
	if !supportsAES {
		c.fallback = &blockExpanded{}
		newBlockExpanded(c.fallback, key)
		return c
	}

	switch len(key) {
	case aes128KeySize:
		c.function = aes128
	case aes192KeySize:
		c.function = aes192
	case aes256KeySize:
		c.function = aes256
	}
	c.key = c.storage[:len(key)]
	copy(c.key, key)
	return c
}

// BlockFunction returns the function code for the block cipher.
// It is used by the GCM implementation to invoke the KMA instruction.
func BlockFunction(c *Block) int {
	return int(c.function)
}

// BlockKey returns the key for the block cipher.
// It is used by the GCM implementation to invoke the KMA instruction.
func BlockKey(c *Block) []byte {
	return c.key
}

func encryptBlock(c *Block, dst, src []byte) {
	if c.fallback != nil {
		encryptBlockGeneric(c.fallback, dst, src)
	} else {
		cryptBlocks(c.function, &c.key[0], &dst[0], &src[0], BlockSize)
	}
}

func decryptBlock(c *Block, dst, src []byte) {
	if c.fallback != nil {
		decryptBlockGeneric(c.fallback, dst, src)
	} else {
		// The decrypt function code is equal to the function code + 128.
		cryptBlocks(c.function+128, &c.key[0], &dst[0], &src[0], BlockSize)
	}
}

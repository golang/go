// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package aes

import (
	"crypto/internal/fips140deps/byteorder"
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

func rotwLE(t uint32) uint32 {
	return (t >> 8) | (t << 24)
}

var supportsAES = cpu.RISCV64HasV && cpu.RISCV64HasZvkned &&
	cpu.RISCV64HasZbb && cpu.RISCV64VLENB >= 16

func init() {
	impl.Register("aes", "Zvkned", &supportsAES)
}

type block struct {
	blockExpanded
}

//go:noescape
func expandKeyAsm(nr int, key *byte, enc *uint32, dec *uint32)

//go:noescape
func encryptBlockAsm(nr int, xk *uint32, dst, src *byte)

//go:noescape
func decryptBlockAsm(nr int, xk *uint32, dst, src *byte)

func checkGenericIsExpected() {
	if supportsAES {
		panic("crypto/aes: internal error: using generic implementation despite hardware support")
	}
}

// expandKeyAES192LE expands an AES192 key for use with RISC-V
// Zvkned instructions. These keys are in little endian form and
// the decryption keys are simply reversed (and do not have
// MixColumns applied).
func expandKeyAES192LE(c *blockExpanded, key []byte) {
	const nk = 6
	const n = 52

	for i := 0; i < nk; i++ {
		c.enc[i] = byteorder.LEUint32(key[4*i:])
	}
	for i := nk; i < n; i++ {
		t := c.enc[i-1]
		if i%nk == 0 {
			t = subw(rotwLE(t)) ^ uint32(powx[i/nk-1])
		}
		c.enc[i] = c.enc[i-nk] ^ t
	}
	for i := 0; i < n; i += 4 {
		ei := n - i - 4
		for j := 0; j < 4; j++ {
			c.dec[i+j] = c.enc[ei+j]
		}
	}
}

func newBlock(c *Block, key []byte) *Block {
	switch len(key) {
	case aes128KeySize:
		c.rounds = aes128Rounds
	case aes192KeySize:
		c.rounds = aes192Rounds
	case aes256KeySize:
		c.rounds = aes256Rounds
	default:
		panic("invalid AES key size")
	}

	if !supportsAES {
		expandKeyGeneric(&c.blockExpanded, key)
	} else if c.rounds == aes192Rounds {
		expandKeyAES192LE(&c.blockExpanded, key)
	} else {
		expandKeyAsm(c.rounds, &key[0], &c.enc[0], &c.dec[0])
	}
	return c
}

func EncryptionKeySchedule(c *Block) []uint32 {
	return c.enc[:c.roundKeysSize()]
}

func encryptBlock(c *Block, dst, src []byte) {
	if supportsAES {
		encryptBlockAsm(c.rounds, &c.enc[0], &dst[0], &src[0])
	} else {
		encryptBlockGeneric(&c.blockExpanded, dst, src)
	}
}

func decryptBlock(c *Block, dst, src []byte) {
	if supportsAES {
		decryptBlockAsm(c.rounds, &c.dec[0], &dst[0], &src[0])
	} else {
		decryptBlockGeneric(&c.blockExpanded, dst, src)
	}
}

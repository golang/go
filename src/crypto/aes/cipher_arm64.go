// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import (
	"crypto/cipher"
	"crypto/internal/subtle"
	"internal/cpu"
	"math/bits"
)

// defined in asm_arm64.s
//go:noescape
func encryptBlockAsm(nr int, xk *uint32, dst, src *byte)

//go:noescape
func decryptBlockAsm(nr int, xk *uint32, dst, src *byte)

type aesCipherAsm struct {
	aesCipher
}

func newCipher(key []byte) (cipher.Block, error) {
	if !cpu.ARM64.HasAES {
		return newCipherGeneric(key)
	}
	n := len(key) + 28
	c := aesCipherAsm{aesCipher{make([]uint32, n), make([]uint32, n)}}
	arm64ExpandKey(key, c.enc, c.dec)
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
	if subtle.InexactOverlap(dst[:BlockSize], src[:BlockSize]) {
		panic("crypto/aes: invalid buffer overlap")
	}
	encryptBlockAsm(len(c.enc)/4-1, &c.enc[0], &dst[0], &src[0])
}

func (c *aesCipherAsm) Decrypt(dst, src []byte) {
	if len(src) < BlockSize {
		panic("crypto/aes: input not full block")
	}
	if len(dst) < BlockSize {
		panic("crypto/aes: output not full block")
	}
	if subtle.InexactOverlap(dst[:BlockSize], src[:BlockSize]) {
		panic("crypto/aes: invalid buffer overlap")
	}
	decryptBlockAsm(len(c.dec)/4-1, &c.dec[0], &dst[0], &src[0])
}

func arm64ExpandKey(key []byte, enc, dec []uint32) {
	expandKeyGo(key, enc, dec)
	nk := len(enc)
	for i := 0; i < nk; i++ {
		enc[i] = bits.ReverseBytes32(enc[i])
		dec[i] = bits.ReverseBytes32(dec[i])
	}
}

// expandKey is used by BenchmarkExpand to ensure that the asm implementation
// of key expansion is used for the benchmark when it is available.
func expandKey(key []byte, enc, dec []uint32) {
	if cpu.ARM64.HasAES {
		arm64ExpandKey(key, enc, dec)
	} else {
		expandKeyGo(key, enc, dec)
	}
}

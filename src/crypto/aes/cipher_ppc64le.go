// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import (
	"crypto/cipher"
	"crypto/internal/subtle"
)

// defined in asm_ppc64le.s

//go:noescape
func setEncryptKeyAsm(key *byte, keylen int, enc *uint32) int

//go:noescape
func setDecryptKeyAsm(key *byte, keylen int, dec *uint32) int

//go:noescape
func doEncryptKeyAsm(key *byte, keylen int, dec *uint32) int

//go:noescape
func encryptBlockAsm(dst, src *byte, enc *uint32)

//go:noescape
func decryptBlockAsm(dst, src *byte, dec *uint32)

type aesCipherAsm struct {
	aesCipher
}

func newCipher(key []byte) (cipher.Block, error) {
	n := 64 // size is fixed for all and round value is stored inside it too
	c := aesCipherAsm{aesCipher{make([]uint32, n), make([]uint32, n)}}
	k := len(key)

	ret := 0
	ret += setEncryptKeyAsm(&key[0], k*8, &c.enc[0])
	ret += setDecryptKeyAsm(&key[0], k*8, &c.dec[0])

	if ret > 0 {
		return nil, KeySizeError(k)
	}

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
	encryptBlockAsm(&dst[0], &src[0], &c.enc[0])
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
	decryptBlockAsm(&dst[0], &src[0], &c.dec[0])
}

// expandKey is used by BenchmarkExpand to ensure that the asm implementation
// of key expansion is used for the benchmark when it is available.
func expandKey(key []byte, enc, dec []uint32) {
	setEncryptKeyAsm(&key[0], len(key)*8, &enc[0])
	setDecryptKeyAsm(&key[0], len(key)*8, &dec[0])
}

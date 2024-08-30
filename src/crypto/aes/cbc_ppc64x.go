// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (ppc64 || ppc64le) && !purego

package aes

import (
	"crypto/cipher"
	"crypto/internal/alias"
)

// Assert that aesCipherAsm implements the cbcEncAble and cbcDecAble interfaces.
var _ cbcEncAble = (*aesCipherAsm)(nil)
var _ cbcDecAble = (*aesCipherAsm)(nil)

const cbcEncrypt = 1
const cbcDecrypt = 0

type cbc struct {
	b   *aesCipherAsm
	enc int
	iv  [BlockSize]byte
}

func (b *aesCipherAsm) NewCBCEncrypter(iv []byte) cipher.BlockMode {
	var c cbc
	c.b = b
	c.enc = cbcEncrypt
	copy(c.iv[:], iv)
	return &c
}

func (b *aesCipherAsm) NewCBCDecrypter(iv []byte) cipher.BlockMode {
	var c cbc
	c.b = b
	c.enc = cbcDecrypt
	copy(c.iv[:], iv)
	return &c
}

func (x *cbc) BlockSize() int { return BlockSize }

// cryptBlocksChain invokes the cipher message identifying encrypt or decrypt.
//
//go:noescape
func cryptBlocksChain(src, dst *byte, length int, key *uint32, iv *byte, enc int, nr int)

func (x *cbc) CryptBlocks(dst, src []byte) {
	if len(src)%BlockSize != 0 {
		panic("crypto/cipher: input not full blocks")
	}
	if len(dst) < len(src) {
		panic("crypto/cipher: output smaller than input")
	}
	if alias.InexactOverlap(dst[:len(src)], src) {
		panic("crypto/cipher: invalid buffer overlap")
	}
	if len(src) > 0 {
		if x.enc == cbcEncrypt {
			cryptBlocksChain(&src[0], &dst[0], len(src), &x.b.enc[0], &x.iv[0], x.enc, int(x.b.l)/4-1)
		} else {
			cryptBlocksChain(&src[0], &dst[0], len(src), &x.b.dec[0], &x.iv[0], x.enc, int(x.b.l)/4-1)
		}
	}
}

func (x *cbc) SetIV(iv []byte) {
	if len(iv) != BlockSize {
		panic("cipher: incorrect length IV")
	}
	copy(x.iv[:], iv)
}

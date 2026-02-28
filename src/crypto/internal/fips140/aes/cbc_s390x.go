// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package aes

// cryptBlocksChain invokes the cipher message with chaining (KMC) instruction
// with the given function code. The length must be a multiple of BlockSize (16).
//
//go:noescape
func cryptBlocksChain(c code, iv, key, dst, src *byte, length int)

func cryptBlocksEnc(b *Block, civ *[BlockSize]byte, dst, src []byte) {
	if b.fallback != nil {
		cryptBlocksEncGeneric(b, civ, dst, src)
		return
	}
	cryptBlocksChain(b.function, &civ[0], &b.key[0], &dst[0], &src[0], len(src))
}

func cryptBlocksDec(b *Block, civ *[BlockSize]byte, dst, src []byte) {
	if b.fallback != nil {
		cryptBlocksDecGeneric(b, civ, dst, src)
		return
	}
	// Decrypt function code is encrypt + 128.
	cryptBlocksChain(b.function+128, &civ[0], &b.key[0], &dst[0], &src[0], len(src))
}

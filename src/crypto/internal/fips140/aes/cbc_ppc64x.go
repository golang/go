// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (ppc64 || ppc64le) && !purego

package aes

// cryptBlocksChain invokes the cipher message identifying encrypt or decrypt.
//
//go:noescape
func cryptBlocksChain(src, dst *byte, length int, key *uint32, iv *byte, enc int, nr int)

const cbcEncrypt = 1
const cbcDecrypt = 0

func cryptBlocksEnc(b *Block, civ *[BlockSize]byte, dst, src []byte) {
	if !supportsAES {
		cryptBlocksEncGeneric(b, civ, dst, src)
	} else {
		cryptBlocksChain(&src[0], &dst[0], len(src), &b.enc[0], &civ[0], cbcEncrypt, b.rounds)
	}
}

func cryptBlocksDec(b *Block, civ *[BlockSize]byte, dst, src []byte) {
	if !supportsAES {
		cryptBlocksDecGeneric(b, civ, dst, src)
	} else {
		cryptBlocksChain(&src[0], &dst[0], len(src), &b.dec[0], &civ[0], cbcDecrypt, b.rounds)
	}
}

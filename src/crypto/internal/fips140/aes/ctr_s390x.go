// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package aes

import (
	"crypto/internal/fips140/subtle"
	"crypto/internal/fips140deps/byteorder"
)

func ctrBlocks1(b *Block, dst, src *[BlockSize]byte, ivlo, ivhi uint64) {
	ctrBlocksS390x(b, dst[:], src[:], ivlo, ivhi)
}

func ctrBlocks2(b *Block, dst, src *[2 * BlockSize]byte, ivlo, ivhi uint64) {
	ctrBlocksS390x(b, dst[:], src[:], ivlo, ivhi)
}

func ctrBlocks4(b *Block, dst, src *[4 * BlockSize]byte, ivlo, ivhi uint64) {
	ctrBlocksS390x(b, dst[:], src[:], ivlo, ivhi)
}

func ctrBlocks8(b *Block, dst, src *[8 * BlockSize]byte, ivlo, ivhi uint64) {
	ctrBlocksS390x(b, dst[:], src[:], ivlo, ivhi)
}

func ctrBlocksS390x(b *Block, dst, src []byte, ivlo, ivhi uint64) {
	if b.fallback != nil {
		ctrBlocks(b, dst, src, ivlo, ivhi)
	}

	buf := make([]byte, len(src), 8*BlockSize)
	for i := 0; i < len(buf); i += BlockSize {
		byteorder.BEPutUint64(buf[i:], ivhi)
		byteorder.BEPutUint64(buf[i+8:], ivlo)
		ivlo, ivhi = add128(ivlo, ivhi, 1)
	}

	// Encrypt the buffer using AES in ECB mode.
	cryptBlocks(b.function, &b.key[0], &buf[0], &buf[0], len(buf))

	// XOR into buf first, in case src and dst overlap (see ctrBlocks).
	subtle.XORBytes(buf, src, buf)
	copy(dst, buf)
}

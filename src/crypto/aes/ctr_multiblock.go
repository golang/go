// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64

package aes

import (
	"crypto/cipher"
	"crypto/internal/alias"
)

//go:generate sh -c "go run ./ctr_multiblock_amd64_gen.go | asmfmt > ctr_multiblock_amd64.s"
//go:generate sh -c "go run ./ctr_multiblock_arm64_gen.go | asmfmt > ctr_multiblock_arm64.s"

// defined in ctr_multiblock_*.s

//go:noescape
func rev16Asm(iv *byte)

//go:noescape
func ctrBlocks1Asm(nr int, xk *uint32, dst, src, ivRev *byte, blockIndex uint64)

//go:noescape
func ctrBlocks2Asm(nr int, xk *uint32, dst, src, ivRev *byte, blockIndex uint64)

//go:noescape
func ctrBlocks4Asm(nr int, xk *uint32, dst, src, ivRev *byte, blockIndex uint64)

//go:noescape
func ctrBlocks8Asm(nr int, xk *uint32, dst, src, ivRev *byte, blockIndex uint64)

type aesCtrWithIV struct {
	enc    []uint32
	rounds int
	ivRev  [BlockSize]byte
	offset uint64
}

// NewCTR implements crypto/cipher.ctrAble so that crypto/cipher.NewCTR
// will use the optimised implementation in this file when possible.
func (c *aesCipherAsm) NewCTR(iv []byte) cipher.Stream {
	if len(iv) != BlockSize {
		panic("bad IV length")
	}

	// Reverse IV once, because it is needed in reversed form
	// in all subsequent ASM calls.
	var ivRev [BlockSize]byte
	copy(ivRev[:], iv)
	rev16Asm(&ivRev[0])

	return &aesCtrWithIV{
		enc:    c.enc,
		rounds: len(c.enc)/4 - 1,
		ivRev:  ivRev,
		offset: 0,
	}
}

func (c *aesCtrWithIV) XORKeyStream(dst, src []byte) {
	c.XORKeyStreamAt(dst, src, c.offset)
	c.offset += uint64(len(src))
}

func (c *aesCtrWithIV) XORKeyStreamAt(dst, src []byte, offset uint64) {
	if len(dst) < len(src) {
		panic("len(dst) < len(src)")
	}
	dst = dst[:len(src)]

	if alias.InexactOverlap(dst, src) {
		panic("crypto/aes: invalid buffer overlap")
	}

	offsetMod16 := offset % BlockSize

	if offsetMod16 != 0 {
		// We have a partial block in the beginning.
		plaintext := make([]byte, BlockSize)
		copy(plaintext[offsetMod16:BlockSize], src)
		ciphertext := make([]byte, BlockSize)
		ctrBlocks1Asm(c.rounds, &c.enc[0], &ciphertext[0], &plaintext[0], &c.ivRev[0], offset/BlockSize)
		progress := BlockSize - offsetMod16
		if progress > uint64(len(src)) {
			progress = uint64(len(src))
		}
		copy(dst[:progress], ciphertext[offsetMod16:BlockSize])
		src = src[progress:]
		dst = dst[progress:]
		offset += progress
	}

	for len(src) >= 8*BlockSize {
		ctrBlocks8Asm(c.rounds, &c.enc[0], &dst[0], &src[0], &c.ivRev[0], offset/BlockSize)
		src = src[8*BlockSize:]
		dst = dst[8*BlockSize:]
		offset += 8 * BlockSize
	}
	// 4, 2, and 1 blocks in the end can happen max 1 times, so if, not for.
	if len(src) >= 4*BlockSize {
		ctrBlocks4Asm(c.rounds, &c.enc[0], &dst[0], &src[0], &c.ivRev[0], offset/BlockSize)
		src = src[4*BlockSize:]
		dst = dst[4*BlockSize:]
		offset += 4 * BlockSize
	}
	if len(src) >= 2*BlockSize {
		ctrBlocks2Asm(c.rounds, &c.enc[0], &dst[0], &src[0], &c.ivRev[0], offset/BlockSize)
		src = src[2*BlockSize:]
		dst = dst[2*BlockSize:]
		offset += 2 * BlockSize
	}
	if len(src) >= 1*BlockSize {
		ctrBlocks1Asm(c.rounds, &c.enc[0], &dst[0], &src[0], &c.ivRev[0], offset/BlockSize)
		src = src[1*BlockSize:]
		dst = dst[1*BlockSize:]
		offset += 1 * BlockSize
	}

	if len(src) != 0 {
		// We have a partial block in the end.
		plaintext := make([]byte, BlockSize)
		copy(plaintext, src)
		ciphertext := make([]byte, BlockSize)
		ctrBlocks1Asm(c.rounds, &c.enc[0], &ciphertext[0], &plaintext[0], &c.ivRev[0], offset/BlockSize)
		copy(dst, ciphertext)
	}
}

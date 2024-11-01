// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || arm64) && !purego

package aes

//go:generate sh -c "go run ./ctr_arm64_gen.go | asmfmt > ctr_arm64.s"

//go:noescape
func ctrBlocks1Asm(nr int, xk *[60]uint32, dst, src *[BlockSize]byte, ivlo, ivhi uint64)

//go:noescape
func ctrBlocks2Asm(nr int, xk *[60]uint32, dst, src *[2 * BlockSize]byte, ivlo, ivhi uint64)

//go:noescape
func ctrBlocks4Asm(nr int, xk *[60]uint32, dst, src *[4 * BlockSize]byte, ivlo, ivhi uint64)

//go:noescape
func ctrBlocks8Asm(nr int, xk *[60]uint32, dst, src *[8 * BlockSize]byte, ivlo, ivhi uint64)

func ctrBlocks1(b *Block, dst, src *[BlockSize]byte, ivlo, ivhi uint64) {
	if !supportsAES {
		ctrBlocks(b, dst[:], src[:], ivlo, ivhi)
	} else {
		ctrBlocks1Asm(b.rounds, &b.enc, dst, src, ivlo, ivhi)
	}
}

func ctrBlocks2(b *Block, dst, src *[2 * BlockSize]byte, ivlo, ivhi uint64) {
	if !supportsAES {
		ctrBlocks(b, dst[:], src[:], ivlo, ivhi)
	} else {
		ctrBlocks2Asm(b.rounds, &b.enc, dst, src, ivlo, ivhi)
	}
}

func ctrBlocks4(b *Block, dst, src *[4 * BlockSize]byte, ivlo, ivhi uint64) {
	if !supportsAES {
		ctrBlocks(b, dst[:], src[:], ivlo, ivhi)
	} else {
		ctrBlocks4Asm(b.rounds, &b.enc, dst, src, ivlo, ivhi)
	}
}

func ctrBlocks8(b *Block, dst, src *[8 * BlockSize]byte, ivlo, ivhi uint64) {
	if !supportsAES {
		ctrBlocks(b, dst[:], src[:], ivlo, ivhi)
	} else {
		ctrBlocks8Asm(b.rounds, &b.enc, dst, src, ivlo, ivhi)
	}
}

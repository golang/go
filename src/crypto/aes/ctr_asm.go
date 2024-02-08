// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || arm64) && !purego

package aes

import (
	"crypto/cipher"
	"crypto/internal/fips/alias"
	"internal/byteorder"
	"math/bits"
)

// Each ctrBlocksNAsm function XORs src with N blocks of counter keystream, and
// stores it in dst. src is loaded in full before storing dst, so they can
// overlap even inexactly. The starting counter value is passed in as a pair of
// little-endian 64-bit integers.

//go:generate sh -c "go run ./ctr_arm64_gen.go | asmfmt > ctr_arm64.s"

//go:noescape
func ctrBlocks1Asm(nr int, xk *[60]uint32, dst, src *[BlockSize]byte, ivlo, ivhi uint64)

//go:noescape
func ctrBlocks2Asm(nr int, xk *[60]uint32, dst, src *[2 * BlockSize]byte, ivlo, ivhi uint64)

//go:noescape
func ctrBlocks4Asm(nr int, xk *[60]uint32, dst, src *[4 * BlockSize]byte, ivlo, ivhi uint64)

//go:noescape
func ctrBlocks8Asm(nr int, xk *[60]uint32, dst, src *[8 * BlockSize]byte, ivlo, ivhi uint64)

type aesCtrWithIV struct {
	enc        [60]uint32
	rounds     int    // 10 for AES-128, 12 for AES-192, 14 for AES-256
	ivlo, ivhi uint64 // start counter as 64-bit limbs
	offset     uint64 // for XORKeyStream only
}

var _ ctrAble = (*aesCipherAsm)(nil)

func (c *aesCipherAsm) NewCTR(iv []byte) cipher.Stream {
	if len(iv) != BlockSize {
		panic("bad IV length")
	}

	return &aesCtrWithIV{
		enc:    c.enc,
		rounds: int(c.l/4 - 1),
		ivlo:   byteorder.BeUint64(iv[8:16]),
		ivhi:   byteorder.BeUint64(iv[0:8]),
		offset: 0,
	}
}

func (c *aesCtrWithIV) XORKeyStream(dst, src []byte) {
	c.XORKeyStreamAt(dst, src, c.offset)

	var carry uint64
	c.offset, carry = bits.Add64(c.offset, uint64(len(src)), 0)
	if carry != 0 {
		panic("crypto/aes: counter overflow")
	}
}

// XORKeyStreamAt behaves like XORKeyStream but keeps no state, and instead
// seeks into the keystream by the given bytes offset from the start (ignoring
// any XORKetStream calls). This allows for random access into the keystream, up
// to 16 EiB from the start.
func (c *aesCtrWithIV) XORKeyStreamAt(dst, src []byte, offset uint64) {
	if len(dst) < len(src) {
		panic("crypto/aes: len(dst) < len(src)")
	}
	dst = dst[:len(src)]
	if alias.InexactOverlap(dst, src) {
		panic("crypto/aes: invalid buffer overlap")
	}

	ivlo, ivhi := add128(c.ivlo, c.ivhi, offset/BlockSize)

	if blockOffset := offset % BlockSize; blockOffset != 0 {
		// We have a partial block at the beginning.
		var in, out [BlockSize]byte
		copy(in[blockOffset:], src)
		ctrBlocks1Asm(c.rounds, &c.enc, &out, &in, ivlo, ivhi)
		n := copy(dst, out[blockOffset:])
		src = src[n:]
		dst = dst[n:]
		ivlo, ivhi = add128(ivlo, ivhi, 1)
	}

	for len(src) >= 8*BlockSize {
		ctrBlocks8Asm(c.rounds, &c.enc, (*[8 * BlockSize]byte)(dst), (*[8 * BlockSize]byte)(src), ivlo, ivhi)
		src = src[8*BlockSize:]
		dst = dst[8*BlockSize:]
		ivlo, ivhi = add128(ivlo, ivhi, 8)
	}

	// The tail can have at most 7 = 4 + 2 + 1 blocks.
	if len(src) >= 4*BlockSize {
		ctrBlocks4Asm(c.rounds, &c.enc, (*[4 * BlockSize]byte)(dst), (*[4 * BlockSize]byte)(src), ivlo, ivhi)
		src = src[4*BlockSize:]
		dst = dst[4*BlockSize:]
		ivlo, ivhi = add128(ivlo, ivhi, 4)
	}
	if len(src) >= 2*BlockSize {
		ctrBlocks2Asm(c.rounds, &c.enc, (*[2 * BlockSize]byte)(dst), (*[2 * BlockSize]byte)(src), ivlo, ivhi)
		src = src[2*BlockSize:]
		dst = dst[2*BlockSize:]
		ivlo, ivhi = add128(ivlo, ivhi, 2)
	}
	if len(src) >= 1*BlockSize {
		ctrBlocks1Asm(c.rounds, &c.enc, (*[1 * BlockSize]byte)(dst), (*[1 * BlockSize]byte)(src), ivlo, ivhi)
		src = src[1*BlockSize:]
		dst = dst[1*BlockSize:]
		ivlo, ivhi = add128(ivlo, ivhi, 1)
	}

	if len(src) != 0 {
		// We have a partial block at the end.
		var in, out [BlockSize]byte
		copy(in[:], src)
		ctrBlocks1Asm(c.rounds, &c.enc, &out, &in, ivlo, ivhi)
		copy(dst, out[:])
	}
}

func add128(lo, hi uint64, x uint64) (uint64, uint64) {
	lo, c := bits.Add64(lo, x, 0)
	hi, _ = bits.Add64(hi, 0, c)
	return lo, hi
}

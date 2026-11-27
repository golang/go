// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/alias"
	"crypto/internal/fips140/subtle"
	"crypto/internal/fips140deps/byteorder"
	"math/bits"
)

type CTR struct {
	b          Block
	ivlo, ivhi uint64          // start counter as 64-bit limbs
	offset     uint64          // for XORKeyStream only
	keybuf     [BlockSize]byte // keystream from offset to the end of its block
}

func NewCTR(b *Block, iv []byte) *CTR {
	// Allocate the CTR here, in an easily inlineable function, so
	// the allocation can be done in the caller's stack frame
	// instead of the heap.  See issue 70499.
	c := newCTR(b, iv)
	return &c
}

func newCTR(b *Block, iv []byte) CTR {
	if len(iv) != BlockSize {
		panic("bad IV length")
	}

	return CTR{
		b:      *b,
		ivlo:   byteorder.BEUint64(iv[8:16]),
		ivhi:   byteorder.BEUint64(iv[0:8]),
		offset: 0,
	}
}

func (c *CTR) XORKeyStream(dst, src []byte) {
	c.xorKeyStreamAt(dst, src, c.offset, &c.keybuf)

	var carry uint64
	c.offset, carry = bits.Add64(c.offset, uint64(len(src)), 0)
	if carry != 0 {
		panic("crypto/aes: counter overflow")
	}
}

// RoundToBlock is used by CTR_DRBG, which discards the rightmost unused bits at
// each request. It rounds the offset up to the next block boundary.
func RoundToBlock(c *CTR) {
	if remainder := c.offset % BlockSize; remainder != 0 {
		var carry uint64
		c.offset, carry = bits.Add64(c.offset, BlockSize-remainder, 0)
		if carry != 0 {
			panic("crypto/aes: counter overflow")
		}
	}
}

// XORKeyStreamAt behaves like XORKeyStream but keeps no state, and instead
// seeks into the keystream by the given bytes offset from the start (ignoring
// any XORKetStream calls). This allows for random access into the keystream, up
// to 16 EiB from the start.
func (c *CTR) XORKeyStreamAt(dst, src []byte, offset uint64) {
	var keybuf [BlockSize]byte
	if offset%BlockSize != 0 {
		ivlo, ivhi := add128(c.ivlo, c.ivhi, offset/BlockSize)
		ctrBlocks1(&c.b, &keybuf, &keybuf, ivlo, ivhi)
	}
	c.xorKeyStreamAt(dst, src, offset, &keybuf)
}

// xorKeyStreamAt implements XORKeyStreamAt. If offset is not a multiple of
// BlockSize, keybuf[offset%BlockSize:] must hold the keystream from offset to
// the end of its block. If the input ends mid-block, keybuf is overwritten so
// that the same holds for the advanced offset.
func (c *CTR) xorKeyStreamAt(dst, src []byte, offset uint64, keybuf *[BlockSize]byte) {
	if len(dst) < len(src) {
		panic("crypto/aes: len(dst) < len(src)")
	}
	dst = dst[:len(src)]
	if alias.InexactOverlap(dst, src) {
		panic("crypto/aes: invalid buffer overlap")
	}
	fips140.RecordApproved()

	ivlo, ivhi := add128(c.ivlo, c.ivhi, offset/BlockSize)

	if blockOffset := offset % BlockSize; blockOffset != 0 {
		// We have a partial block at the beginning.
		n := min(len(src), BlockSize-int(blockOffset))
		d, s, k := dst[:n], src[:n], keybuf[blockOffset:][:n]
		for i := range k {
			d[i] = s[i] ^ k[i]
		}
		src = src[n:]
		dst = dst[n:]
		ivlo, ivhi = add128(ivlo, ivhi, 1)
	}

	if n := uint64(len(src)) / BlockSize; ctrBulk(n, ivlo) {
		// On arm64, all full blocks go to one assembly call, unless the low
		// 64 bits of the counter would wrap within them.
		if supportsAESEOR3 {
			ctrBlocksBulkEOR3(&c.b, &dst[0], &src[0], n, ivlo, ivhi)
		} else {
			ctrBlocksBulk(&c.b, &dst[0], &src[0], n, ivlo, ivhi)
		}
		src = src[n*BlockSize:]
		dst = dst[n*BlockSize:]
		ivlo, ivhi = add128(ivlo, ivhi, n)
	} else {
		for len(src) >= 8*BlockSize {
			ctrBlocks8(&c.b, (*[8 * BlockSize]byte)(dst), (*[8 * BlockSize]byte)(src), ivlo, ivhi)
			src = src[8*BlockSize:]
			dst = dst[8*BlockSize:]
			ivlo, ivhi = add128(ivlo, ivhi, 8)
		}

		// The tail can have at most 7 = 4 + 2 + 1 blocks.
		if len(src) >= 4*BlockSize {
			ctrBlocks4(&c.b, (*[4 * BlockSize]byte)(dst), (*[4 * BlockSize]byte)(src), ivlo, ivhi)
			src = src[4*BlockSize:]
			dst = dst[4*BlockSize:]
			ivlo, ivhi = add128(ivlo, ivhi, 4)
		}
		if len(src) >= 2*BlockSize {
			ctrBlocks2(&c.b, (*[2 * BlockSize]byte)(dst), (*[2 * BlockSize]byte)(src), ivlo, ivhi)
			src = src[2*BlockSize:]
			dst = dst[2*BlockSize:]
			ivlo, ivhi = add128(ivlo, ivhi, 2)
		}
		if len(src) >= 1*BlockSize {
			ctrBlocks1(&c.b, (*[1 * BlockSize]byte)(dst), (*[1 * BlockSize]byte)(src), ivlo, ivhi)
			src = src[1*BlockSize:]
			dst = dst[1*BlockSize:]
			ivlo, ivhi = add128(ivlo, ivhi, 1)
		}
	}

	if len(src) != 0 {
		// We have a partial block at the end.
		var in [BlockSize]byte
		copy(in[:], src)
		ctrBlocks1(&c.b, keybuf, &in, ivlo, ivhi)
		copy(dst, keybuf[:])
	}
}

// Each of ctrBlocks1, ctrBlocks2, ctrBlocks4 and ctrBlocks8 XORs src with that
// many blocks of counter keystream, and stores it in dst. src is loaded in full
// before storing dst, so they can overlap even inexactly. (ctrBlocksBulk and
// ctrBlocksBulkEOR3, used on arm64, need exact overlap or none.) The starting
// counter value is passed in as a pair of little-endian 64-bit integers.

func ctrBlocks(b *Block, dst, src []byte, ivlo, ivhi uint64) {
	buf := make([]byte, len(src), 8*BlockSize)
	for i := 0; i < len(buf); i += BlockSize {
		byteorder.BEPutUint64(buf[i:], ivhi)
		byteorder.BEPutUint64(buf[i+8:], ivlo)
		ivlo, ivhi = add128(ivlo, ivhi, 1)
		encryptBlock(b, buf[i:], buf[i:])
	}
	// XOR into buf first, in case src and dst overlap (see above).
	subtle.XORBytes(buf, src, buf)
	copy(dst, buf)
}

func add128(lo, hi uint64, x uint64) (uint64, uint64) {
	lo, c := bits.Add64(lo, x, 0)
	hi, _ = bits.Add64(hi, 0, c)
	return lo, hi
}

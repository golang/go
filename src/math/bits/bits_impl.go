// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides basic implementations of the bits functions.

package bits

const uintSize = 32 << (^uint(0) >> 32 & 1) // 32 or 64

func ntz(x uint) (n int) {
	if UintSize == 32 {
		return ntz32(uint32(x))
	}
	return ntz64(uint64(x))
}

// See http://supertech.csail.mit.edu/papers/debruijn.pdf
const deBruijn32 = 0x077CB531

var deBruijn32tab = [32]byte{
	0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
	31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9,
}

func ntz8(x uint8) (n int) {
	if x == 0 {
		return 8
	}
	// see comment in ntz64
	return int(deBruijn32tab[uint32(x&-x)*deBruijn32>>(32-5)])
}

func ntz16(x uint16) (n int) {
	if x == 0 {
		return 16
	}
	// see comment in ntz64
	return int(deBruijn32tab[uint32(x&-x)*deBruijn32>>(32-5)])
}

func ntz32(x uint32) int {
	if x == 0 {
		return 32
	}
	// see comment in ntz64
	return int(deBruijn32tab[(x&-x)*deBruijn32>>(32-5)])
}

const deBruijn64 = 0x03f79d71b4ca8b09

var deBruijn64tab = [64]byte{
	0, 1, 56, 2, 57, 49, 28, 3, 61, 58, 42, 50, 38, 29, 17, 4,
	62, 47, 59, 36, 45, 43, 51, 22, 53, 39, 33, 30, 24, 18, 12, 5,
	63, 55, 48, 27, 60, 41, 37, 16, 46, 35, 44, 21, 52, 32, 23, 11,
	54, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6,
}

func ntz64(x uint64) int {
	if x == 0 {
		return 64
	}
	// If popcount is fast, replace code below with return popcount(^x & (x - 1)).
	//
	// x & -x leaves only the right-most bit set in the word. Let k be the
	// index of that bit. Since only a single bit is set, the value is two
	// to the power of k. Multiplying by a power of two is equivalent to
	// left shifting, in this case by k bits. The de Bruijn (64 bit) constant
	// is such that all six bit, consecutive substrings are distinct.
	// Therefore, if we have a left shifted version of this constant we can
	// find by how many bits it was shifted by looking at which six bit
	// substring ended up at the top of the word.
	// (Knuth, volume 4, section 7.3.1)
	return int(deBruijn64tab[(x&-x)*deBruijn64>>(64-6)])
}

func blen(x uint64) (i int) {
	for ; x >= 1<<(16-1); x >>= 16 {
		i += 16
	}
	if x >= 1<<(8-1) {
		x >>= 8
		i += 8
	}
	if x >= 1<<(4-1) {
		x >>= 4
		i += 4
	}
	if x >= 1<<(2-1) {
		x >>= 2
		i += 2
	}
	if x >= 1<<(1-1) {
		i++
	}
	return
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package des

import (
	"encoding/binary"
	"sync"
)

func cryptBlock(subkeys []uint64, dst, src []byte, decrypt bool) {
	b := binary.BigEndian.Uint64(src)
	b = permuteInitialBlock(b)
	left, right := uint32(b>>32), uint32(b)

	left = (left << 1) | (left >> 31)
	right = (right << 1) | (right >> 31)

	if decrypt {
		for i := 0; i < 8; i++ {
			left, right = feistel(left, right, subkeys[15-2*i], subkeys[15-(2*i+1)])
		}
	} else {
		for i := 0; i < 8; i++ {
			left, right = feistel(left, right, subkeys[2*i], subkeys[2*i+1])
		}
	}

	left = (left << 31) | (left >> 1)
	right = (right << 31) | (right >> 1)

	// switch left & right and perform final permutation
	preOutput := (uint64(right) << 32) | uint64(left)
	binary.BigEndian.PutUint64(dst, permuteFinalBlock(preOutput))
}

// DES Feistel function. feistelBox must be initialized via
// feistelBoxOnce.Do(initFeistelBox) first.
func feistel(l, r uint32, k0, k1 uint64) (lout, rout uint32) {
	var t uint32

	t = r ^ uint32(k0>>32)
	l ^= feistelBox[7][t&0x3f] ^
		feistelBox[5][(t>>8)&0x3f] ^
		feistelBox[3][(t>>16)&0x3f] ^
		feistelBox[1][(t>>24)&0x3f]

	t = ((r << 28) | (r >> 4)) ^ uint32(k0)
	l ^= feistelBox[6][(t)&0x3f] ^
		feistelBox[4][(t>>8)&0x3f] ^
		feistelBox[2][(t>>16)&0x3f] ^
		feistelBox[0][(t>>24)&0x3f]

	t = l ^ uint32(k1>>32)
	r ^= feistelBox[7][t&0x3f] ^
		feistelBox[5][(t>>8)&0x3f] ^
		feistelBox[3][(t>>16)&0x3f] ^
		feistelBox[1][(t>>24)&0x3f]

	t = ((l << 28) | (l >> 4)) ^ uint32(k1)
	r ^= feistelBox[6][(t)&0x3f] ^
		feistelBox[4][(t>>8)&0x3f] ^
		feistelBox[2][(t>>16)&0x3f] ^
		feistelBox[0][(t>>24)&0x3f]

	return l, r
}

// feistelBox[s][16*i+j] contains the output of permutationFunction
// for sBoxes[s][i][j] << 4*(7-s)
var feistelBox [8][64]uint32

var feistelBoxOnce sync.Once

// general purpose function to perform DES block permutations.
func permuteBlock(src uint64, permutation []uint8) (block uint64) {
	for position, n := range permutation {
		bit := (src >> n) & 1
		block |= bit << uint((len(permutation)-1)-position)
	}
	return
}

func initFeistelBox() {
	for s := range sBoxes {
		for i := 0; i < 4; i++ {
			for j := 0; j < 16; j++ {
				f := uint64(sBoxes[s][i][j]) << (4 * (7 - uint(s)))
				f = permuteBlock(f, permutationFunction[:])

				// Row is determined by the 1st and 6th bit.
				// Column is the middle four bits.
				row := uint8(((i & 2) << 4) | i&1)
				col := uint8(j << 1)
				t := row | col

				// The rotation was performed in the feistel rounds, being factored out and now mixed into the feistelBox.
				f = (f << 1) | (f >> 31)

				feistelBox[s][t] = uint32(f)
			}
		}
	}
}

// permuteInitialBlock is equivalent to the permutation defined
// by initialPermutation.
func permuteInitialBlock(block uint64) uint64 {
	// block = b7 b6 b5 b4 b3 b2 b1 b0 (8 bytes)
	b1 := block >> 48
	b2 := block << 48
	block ^= b1 ^ b2 ^ b1<<48 ^ b2>>48

	// block = b1 b0 b5 b4 b3 b2 b7 b6
	b1 = block >> 32 & 0xff00ff
	b2 = (block & 0xff00ff00)
	block ^= b1<<32 ^ b2 ^ b1<<8 ^ b2<<24 // exchange b0 b4 with b3 b7

	// block is now b1 b3 b5 b7 b0 b2 b4 b6, the permutation:
	//                  ...  8
	//                  ... 24
	//                  ... 40
	//                  ... 56
	//  7  6  5  4  3  2  1  0
	// 23 22 21 20 19 18 17 16
	//                  ... 32
	//                  ... 48

	// exchange 4,5,6,7 with 32,33,34,35 etc.
	b1 = block & 0x0f0f00000f0f0000
	b2 = block & 0x0000f0f00000f0f0
	block ^= b1 ^ b2 ^ b1>>12 ^ b2<<12

	// block is the permutation:
	//
	//   [+8]         [+40]
	//
	//  7  6  5  4
	// 23 22 21 20
	//  3  2  1  0
	// 19 18 17 16    [+32]

	// exchange 0,1,4,5 with 18,19,22,23
	b1 = block & 0x3300330033003300
	b2 = block & 0x00cc00cc00cc00cc
	block ^= b1 ^ b2 ^ b1>>6 ^ b2<<6

	// block is the permutation:
	// 15 14
	// 13 12
	// 11 10
	//  9  8
	//  7  6
	//  5  4
	//  3  2
	//  1  0 [+16] [+32] [+64]

	// exchange 0,2,4,6 with 9,11,13,15:
	b1 = block & 0xaaaaaaaa55555555
	block ^= b1 ^ b1>>33 ^ b1<<33

	// block is the permutation:
	// 6 14 22 30 38 46 54 62
	// 4 12 20 28 36 44 52 60
	// 2 10 18 26 34 42 50 58
	// 0  8 16 24 32 40 48 56
	// 7 15 23 31 39 47 55 63
	// 5 13 21 29 37 45 53 61
	// 3 11 19 27 35 43 51 59
	// 1  9 17 25 33 41 49 57
	return block
}

// permuteFinalBlock is equivalent to the permutation defined
// by finalPermutation.
func permuteFinalBlock(block uint64) uint64 {
	// Perform the same bit exchanges as permuteInitialBlock
	// but in reverse order.
	b1 := block & 0xaaaaaaaa55555555
	block ^= b1 ^ b1>>33 ^ b1<<33

	b1 = block & 0x3300330033003300
	b2 := block & 0x00cc00cc00cc00cc
	block ^= b1 ^ b2 ^ b1>>6 ^ b2<<6

	b1 = block & 0x0f0f00000f0f0000
	b2 = block & 0x0000f0f00000f0f0
	block ^= b1 ^ b2 ^ b1>>12 ^ b2<<12

	b1 = block >> 32 & 0xff00ff
	b2 = (block & 0xff00ff00)
	block ^= b1<<32 ^ b2 ^ b1<<8 ^ b2<<24

	b1 = block >> 48
	b2 = block << 48
	block ^= b1 ^ b2 ^ b1<<48 ^ b2>>48
	return block
}

// creates 16 28-bit blocks rotated according
// to the rotation schedule.
func ksRotate(in uint32) (out []uint32) {
	out = make([]uint32, 16)
	last := in
	for i := 0; i < 16; i++ {
		// 28-bit circular left shift
		left := (last << (4 + ksRotations[i])) >> 4
		right := (last << 4) >> (32 - ksRotations[i])
		out[i] = left | right
		last = out[i]
	}
	return
}

// creates 16 56-bit subkeys from the original key.
func (c *desCipher) generateSubkeys(keyBytes []byte) {
	feistelBoxOnce.Do(initFeistelBox)

	// apply PC1 permutation to key
	key := binary.BigEndian.Uint64(keyBytes)
	permutedKey := permuteBlock(key, permutedChoice1[:])

	// rotate halves of permuted key according to the rotation schedule
	leftRotations := ksRotate(uint32(permutedKey >> 28))
	rightRotations := ksRotate(uint32(permutedKey<<4) >> 4)

	// generate subkeys
	for i := 0; i < 16; i++ {
		// combine halves to form 56-bit input to PC2
		pc2Input := uint64(leftRotations[i])<<28 | uint64(rightRotations[i])
		// apply PC2 permutation to 7 byte input
		c.subkeys[i] = unpack(permuteBlock(pc2Input, permutedChoice2[:]))
	}
}

// Expand 48-bit input to 64-bit, with each 6-bit block padded by extra two bits at the top.
// By doing so, we can have the input blocks (four bits each), and the key blocks (six bits each) well-aligned without
// extra shifts/rotations for alignments.
func unpack(x uint64) uint64 {
	return ((x>>(6*1))&0xff)<<(8*0) |
		((x>>(6*3))&0xff)<<(8*1) |
		((x>>(6*5))&0xff)<<(8*2) |
		((x>>(6*7))&0xff)<<(8*3) |
		((x>>(6*0))&0xff)<<(8*4) |
		((x>>(6*2))&0xff)<<(8*5) |
		((x>>(6*4))&0xff)<<(8*6) |
		((x>>(6*6))&0xff)<<(8*7)
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package des

import (
	"encoding/binary"
)

func cryptBlock(subkeys []uint64, dst, src []byte, decrypt bool) {
	b := binary.BigEndian.Uint64(src)
	b = permuteBlock(b, initialPermutation[:])
	left, right := uint32(b>>32), uint32(b)

	var subkey uint64
	for i := 0; i < 16; i++ {
		if decrypt {
			subkey = subkeys[15-i]
		} else {
			subkey = subkeys[i]
		}

		left, right = right, left^feistel(right, subkey)
	}
	// switch left & right and perform final permutation
	preOutput := (uint64(right) << 32) | uint64(left)
	binary.BigEndian.PutUint64(dst, permuteBlock(preOutput, finalPermutation[:]))
}

// Encrypt one block from src into dst, using the subkeys.
func encryptBlock(subkeys []uint64, dst, src []byte) {
	cryptBlock(subkeys, dst, src, false)
}

// Decrypt one block from src into dst, using the subkeys.
func decryptBlock(subkeys []uint64, dst, src []byte) {
	cryptBlock(subkeys, dst, src, true)
}

// DES Feistel function
func feistel(right uint32, key uint64) (result uint32) {
	sBoxLocations := key ^ permuteBlock(uint64(right), expansionFunction[:])
	var sBoxResult uint32
	for i := uint8(0); i < 8; i++ {
		sBoxLocation := uint8(sBoxLocations>>42) & 0x3f
		sBoxLocations <<= 6
		// row determined by 1st and 6th bit
		row := (sBoxLocation & 0x1) | ((sBoxLocation & 0x20) >> 4)
		// column is middle four bits
		column := (sBoxLocation >> 1) & 0xf
		sBoxResult |= uint32(sBoxes[i][row][column]) << (4 * (7 - i))
	}
	return uint32(permuteBlock(uint64(sBoxResult), permutationFunction[:]))
}

// general purpose function to perform DES block permutations
func permuteBlock(src uint64, permutation []uint8) (block uint64) {
	for position, n := range permutation {
		bit := (src >> n) & 1
		block |= bit << uint((len(permutation)-1)-position)
	}
	return
}

// creates 16 28-bit blocks rotated according
// to the rotation schedule
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

// creates 16 56-bit subkeys from the original key
func (c *Cipher) generateSubkeys(keyBytes []byte) {
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
		c.subkeys[i] = permuteBlock(pc2Input, permutedChoice2[:])
	}
}

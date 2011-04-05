// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package des

// Encrypt one block from src into dst, using the subkeys.
func encryptBlock(subkeys [16]uint64, dst, src []byte) {
	// perform initial permutation
	permutedSrc := permuteBlock(src, initialPermutation[0:])

	// split into left and right halves
	left := uint32(permutedSrc >> 32)
	right := uint32(permutedSrc)

	// process left and right with feistel function
	for i := 0; i < 16; i++ {
		previousRight := right
		right = left ^ feistel(right, subkeys[i])
		left = previousRight
	}
	// switch left & right and perform final permutation
	preOutput := (uint64(right) << 32) | uint64(left)
	final := uint64ToBytes(permuteBlock(uint64ToBytes(preOutput), finalPermutation[0:]))

	// copy bytes to destination
	copy(dst, final)
}

// Decrypt one block from src into dst, using the subkeys.
func decryptBlock(subkeys [16]uint64, dst, src []byte) {
	// perform initial permutation
	permutedSrc := permuteBlock(src, initialPermutation[0:])

	// split into left and right halves
	left := uint32(permutedSrc >> 32)
	right := uint32(permutedSrc)

	// process left and right with feistel function
	for i := 0; i < 16; i++ {
		previousRight := right
		// decryption reverses order of subkeys
		right = left ^ feistel(right, subkeys[15-i])
		left = previousRight
	}
	// switch left & right and perform final permutation
	preOutput := (uint64(right) << 32) | uint64(left)
	final := uint64ToBytes(permuteBlock(uint64ToBytes(preOutput), finalPermutation[0:]))

	// copy bytes to destination
	copy(dst, final)
}

// DES Feistel function
func feistel(right uint32, key uint64) (result uint32) {
	rightExpanded := permuteBlock(uint32ToBytes(right), expansionFunction[:])
	xorResult := key ^ rightExpanded
	var sBoxResult uint32
	for i := uint8(0); i < 8; i++ {
		sBoxCoordValue := uint8((xorResult << (16 + (6 * i))) >> 58)
		// determine the proper S-box row and column from the 6-bits of
		// sBoxCoordValue row is determined by 1st and 6th bit
		row := (sBoxCoordValue & 0x1) | ((sBoxCoordValue & 0x20) >> 4)
		// column is middle four bits
		column := (sBoxCoordValue << 3) >> 4
		sBoxResult |= uint32(sBoxes[i][row][column]) << (4 * (7 - i))
	}
	return uint32(permuteBlock(uint32ToBytes(sBoxResult), permutationFunction[0:]))
}

// general purpose function to perform DES block permutations
func permuteBlock(src []byte, permutation []uint8) (block uint64) {
	for finalPosition, bitNumber := range permutation {
		bitIndex := bitNumber - 1
		byteIndex := bitIndex >> 3
		bitNumberInByte := bitIndex % 8
		bitValue := (src[byteIndex] << bitNumberInByte) >> 7
		block |= uint64(bitValue) << uint64((uint8(len(permutation)-1))-uint8(finalPosition))
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
		part1 := (last << (4 + uint32(ksRotations[i]))) >> 4
		part2 := (last << 4) >> (32 - ksRotations[i])
		out[i] = part1 | part2
		last = out[i]
	}
	return
}

// creates 16 56-bit subkeys from the original key
func ksGenerateSubkeys(cipher *DESCipher) {
	// apply PC1 permutation to key
	permutedKey := permuteBlock(cipher.key, permutedChoice1[0:])

	// rotate halves of permuted key according to the rotation schedule
	leftRotations := ksRotate(uint32(permutedKey >> 28))
	rightRotations := ksRotate(uint32(permutedKey<<4) >> 4)

	// generate subkeys
	for i := 0; i < 16; i++ {
		// combine halves to form 56-bit input to PC2
		pc2Input := uint64(leftRotations[i])<<28 | uint64(rightRotations[i])
		// apply PC2 permutation to 7 byte input
		cipher.subkeys[i] = permuteBlock(uint64ToBytes(pc2Input)[1:], permutedChoice2[0:])
	}
}

// generates a byte array from uint32 input
func uint32ToBytes(block uint32) []byte {
	return []byte{
		byte(block >> 24), byte(block >> 16),
		byte(block >> 8), byte(block)}
}

// generates a byte array from uint64 input
func uint64ToBytes(block uint64) []byte {
	return []byte{
		byte(block >> 56), byte(block >> 48),
		byte(block >> 40), byte(block >> 32),
		byte(block >> 24), byte(block >> 16),
		byte(block >> 8), byte(block)}
}

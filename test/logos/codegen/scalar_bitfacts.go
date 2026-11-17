// asmcheck -O=3

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math/bits"

func alignedTrailingCount(x uint64) int {
	// amd64:"MOVL [$]5, AX" -"BSFQ" -"TZCNTQ"
	// arm64:"MOVD [$]5, R0" -"RBIT" -"CLZ"
	return bits.TrailingZeros64((x &^ 255) + 32)
}

func rotatedHeaderLength(key uint16) int {
	word := bits.RotateLeft64(uint64(key)|1<<16, 24)
	// amd64:"MOVL [$]41, AX" -"BSRQ" -"LZCNTQ" -"ROLQ" -"RORX"
	// arm64:"MOVD [$]41, R0" -"CLZ" -"ROR"
	return bits.Len64(word)
}

func sparsePopulationPrefix(x uint64) int {
	word := (x & 0x15555555) | 0xaaaaaaaa00000000
	// amd64/v2:"MOVL [$]1, AX" -"POPCNTQ" -"CALL"
	// amd64/v3:"MOVL [$]1, AX" -"POPCNTQ" -"CALL"
	// amd64/v4:"MOVL [$]1, AX" -"POPCNTQ" -"CALL"
	// arm64:"MOVD [$]1, R0" -"VCNT" -"VUADDLV"
	return bits.OnesCount64(word) >> 4
}

func dynamicTrailingCount(x uint64) int {
	// amd64:"BSFQ|TZCNTQ"
	// arm64:"RBIT" "CLZ"
	return bits.TrailingZeros64(x)
}

func dynamicHeaderLength(x uint64) int {
	// amd64:"BSRQ|LZCNTQ"
	// arm64:"CLZ"
	return bits.Len64(x)
}

func dynamicPopulationPrefix(x uint64) int {
	// amd64/v2:"POPCNTQ"
	// amd64/v3:"POPCNTQ"
	// amd64/v4:"POPCNTQ"
	// arm64:"VCNT" "VUADDLV"
	return bits.OnesCount64(x) >> 4
}
